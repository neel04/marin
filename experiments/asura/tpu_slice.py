# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create, set up, and delete the Spot multi-host TPU slice named by an asura Fast Track config.

A multi-host slice is a GCE managed instance group of TPU VMs bound by a workload policy. ``setup`` ships
this checkout, the asura cluster config, and ``$WANDB_API_KEY`` to every host, then installs the environment.

Every slice stops billing on its own: ``slice_watchdog.sh`` powers each host off at the ``--stop-at`` time given
to ``create``, or once no training worker has run on it for a while.

    uv run python -m experiments.asura.tpu_slice create --config <config.yaml> --stop-at <RFC 3339 time>
    uv run python -m experiments.asura.tpu_slice {setup,delete} --config <config.yaml>
"""

import argparse
import os
import subprocess
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

from experiments.asura.run_dclm import GcpWorkerSettings, ssh_command, tpu_hosts
from experiments.asura.run_fast_track import worker_settings

# Accelerator -> (machine type, accelerator topology, hosts).
SLICES = {
    "v5p-32": ("ct5p-hightpu-4t", "2x2x4", 4),
    "v6e-16": ("ct6e-standard-4t", "4x4", 4),
}
IMAGE_FAMILY = "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e"
IMAGE_PROJECT = "ubuntu-os-accelerator-images"
READY_TIMEOUT = 30 * 60
REPO = Path(__file__).resolve().parents[2]
WATCHDOG = Path(__file__).with_name("slice_watchdog.sh")
CLUSTER_CONFIG = Path.home() / ".config/marin/clusters/asura.yaml"


def gcloud(settings: GcpWorkerSettings, *args: str) -> None:
    subprocess.run(["gcloud", *args, f"--project={settings.project}", "--quiet"], check=True)


def names(settings: GcpWorkerSettings) -> tuple[str, str, str]:
    """Region, workload-policy name, and instance-template name for the slice."""
    return settings.zone.rsplit("-", 1)[0], f"{settings.tpu_vm}-policy", f"{settings.tpu_vm}-tpl"


def create(settings: GcpWorkerSettings, stop_at: datetime) -> None:
    if stop_at <= datetime.now(UTC):
        raise ValueError(f"Stop time {stop_at.isoformat()} is not in the future")

    machine_type, topology, host_count = SLICES[settings.accelerator]
    region, policy, template = names(settings)

    gcloud(
        settings,
        *("compute", "resource-policies", "create", "workload-policy", policy),
        f"--region={region}",
        "--type=high-throughput",
        f"--accelerator-topology={topology}",
    )

    # Spot templates in a managed instance group must stop, not delete, on preemption, and cannot carry a GCE
    # termination time, so the watchdog stops each host. The do-nothing failure policy keeps stopped hosts stopped.
    gcloud(
        settings,
        *("alpha", "compute", "instance-templates", "create", template),
        f"--machine-type={machine_type}",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
        f"--metadata=stop-at={stop_at.isoformat()}",
        f"--metadata-from-file=startup-script={WATCHDOG}",
        f"--image-family={IMAGE_FAMILY}",
        f"--image-project={IMAGE_PROJECT}",
        "--boot-disk-size=100GB",
        "--scopes=cloud-platform",
    )

    gcloud(
        settings,
        *("alpha", "compute", "instance-groups", "managed", "create", settings.tpu_vm),
        f"--zone={settings.zone}",
        f"--size={host_count}",
        "--target-size-policy-mode=bulk",
        f"--template={template}",
        "--default-action-on-vm-failure=do-nothing",
        f"--workload-policy=projects/{settings.project}/regions/{region}/resourcePolicies/{policy}",
    )

    hosts = wait_until_ready(settings, host_count, READY_TIMEOUT)
    print(f"{settings.tpu_vm} ready: {', '.join(hosts)}", flush=True)


def wait_until_ready(settings: GcpWorkerSettings, host_count: int, timeout: float) -> list[str]:
    """The slice's hosts, once at least ``host_count`` of them run and accept SSH, within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        hosts = tpu_hosts(settings)
        if len(hosts) >= host_count and all(_reachable(settings, host) for host in hosts):
            return hosts

        if time.monotonic() > deadline:
            raise TimeoutError(f"{settings.tpu_vm} not ready after {timeout}s; delete it to stop billing")
        time.sleep(20)


def _reachable(settings: GcpWorkerSettings, host: str) -> bool:
    return subprocess.run([*ssh_command(settings, host), "--command=true"], capture_output=True).returncode == 0


def setup(settings: GcpWorkerSettings, archive: Path) -> None:
    """Install ``archive``, a checkout from ``archive_checkout``, with the credentials on every host."""
    hosts = tpu_hosts(settings)
    if not hosts:
        raise ValueError(f"No running hosts for {settings.tpu_vm}; run create first")
    credentials = {
        "~/.config/marin/clusters/asura.yaml": CLUSTER_CONFIG.read_text(),
        "~/.config/wandb/api-key": os.environ["WANDB_API_KEY"],
    }
    archive_uri = f"{settings.bucket}/tmp/ttl=1d/asura-source/{uuid.uuid4().hex}.tar.gz"
    subprocess.run(["gcloud", "storage", "cp", str(archive), archive_uri], check=True, capture_output=True)

    def install(host: str) -> None:
        ssh = ssh_command(settings, host)
        # Credentials travel over stdin so they never appear in a command line or on disk locally.
        for path, content in credentials.items():
            subprocess.run(
                [*ssh, f"--command=install -D -m 600 /dev/stdin {path}"], input=content, text=True, check=True
            )
        install_command = (
            f"gcloud storage cp {archive_uri} /tmp/marin-source.tar.gz --quiet && "
            "tar -xzOf /tmp/marin-source.tar.gz experiments/asura/setup_worker.sh | bash -s /tmp/marin-source.tar.gz"
        )
        subprocess.run([*ssh, f"--command={install_command}"], check=True)
        print(f"{host}: set up", flush=True)

    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        list(pool.map(install, hosts))


def archive_checkout(archive: Path) -> None:
    """Tar the tracked and untracked, non-ignored files of this checkout, including uncommitted edits."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    with tarfile.open(archive, "w:gz") as tar:
        for path in listing.stdout.split("\0"):
            if path and (REPO / path).is_file():
                tar.add(REPO / path, arcname=path)


def delete(settings: GcpWorkerSettings) -> None:
    """Delete whichever parts of the slice exist."""
    region, policy, template = names(settings)
    # The instance group goes first: it uses the template and the policy.
    deletions = [
        ("instance-groups", "managed", "delete", settings.tpu_vm, f"--zone={settings.zone}"),
        ("instance-templates", "delete", template),
        ("resource-policies", "delete", policy, f"--region={region}"),
    ]

    for deletion in deletions:
        command = ["gcloud", "compute", *deletion, f"--project={settings.project}", "--quiet"]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0 and "was not found" not in result.stderr:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "setup", "delete"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-at", type=datetime.fromisoformat, help="RFC 3339 time at which create's hosts stop")
    parser.add_argument("--slice", help="Slice name, to run beside another queue; defaults to the config's tpu_vm")
    args = parser.parse_args()
    settings = worker_settings(yaml.safe_load(args.config.read_text()), args.slice)

    if args.action == "setup":
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.tar.gz"
            archive_checkout(archive)
            setup(settings, archive)
    elif args.action == "delete":
        delete(settings)
    elif args.stop_at is None or args.stop_at.tzinfo is None:
        parser.error("create needs --stop-at with a UTC offset, so every slice stops billing on its own")
    else:
        create(settings, args.stop_at)


if __name__ == "__main__":
    main()
