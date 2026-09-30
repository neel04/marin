# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create, set up, and delete the Spot multi-host TPU slice named by an asura Fast Track config.

A multi-host slice is a GCE managed instance group of TPU VMs bound by a workload policy. ``setup`` ships
this checkout, the asura cluster config, and ``$WANDB_API_KEY`` to every host, then installs the environment.

    uv run python -m experiments.asura.tpu_slice {create,setup,delete} --config <config.yaml>
"""

import argparse
import os
import subprocess
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
CLUSTER_CONFIG = Path.home() / ".config/marin/clusters/asura.yaml"


def gcloud(settings: GcpWorkerSettings, *args: str) -> None:
    subprocess.run(["gcloud", *args, f"--project={settings.project}", "--quiet"], check=True)


def names(settings: GcpWorkerSettings) -> tuple[str, str, str]:
    """Region, workload-policy name, and instance-template name for the slice."""
    return settings.zone.rsplit("-", 1)[0], f"{settings.tpu_vm}-policy", f"{settings.tpu_vm}-tpl"


def create(settings: GcpWorkerSettings) -> None:
    machine_type, topology, host_count = SLICES[settings.accelerator]
    region, policy, template = names(settings)
    gcloud(
        settings,
        *("compute", "resource-policies", "create", "workload-policy", policy),
        f"--region={region}",
        "--type=high-throughput",
        f"--accelerator-topology={topology}",
    )
    # Spot templates in a managed instance group must stop, not delete, on preemption.
    gcloud(
        settings,
        *("alpha", "compute", "instance-templates", "create", template),
        f"--machine-type={machine_type}",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
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
    deadline = time.monotonic() + READY_TIMEOUT
    while len(hosts := tpu_hosts(settings)) < host_count or not all(_reachable(settings, h) for h in hosts):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{settings.tpu_vm} not ready after {READY_TIMEOUT}s; delete it to stop billing")
        time.sleep(20)
    print(f"{settings.tpu_vm} ready: {', '.join(hosts)}", flush=True)


def _reachable(settings: GcpWorkerSettings, host: str) -> bool:
    return subprocess.run([*ssh_command(settings, host), "--command=true"], capture_output=True).returncode == 0


def setup(settings: GcpWorkerSettings) -> None:
    hosts = tpu_hosts(settings)
    if not hosts:
        raise ValueError(f"No running hosts for {settings.tpu_vm}; run create first")
    credentials = {
        "~/.config/marin/clusters/asura.yaml": CLUSTER_CONFIG.read_text(),
        "~/.config/wandb/api-key": os.environ["WANDB_API_KEY"],
    }
    archive_uri = f"{settings.bucket}/tmp/ttl=1d/asura-source/{uuid.uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.tar.gz"
        _archive_checkout(archive)
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


def _archive_checkout(archive: Path) -> None:
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
    region, policy, template = names(settings)
    gcloud(settings, "compute", "instance-groups", "managed", "delete", settings.tpu_vm, f"--zone={settings.zone}")
    gcloud(settings, "compute", "instance-templates", "delete", template)
    gcloud(settings, "compute", "resource-policies", "delete", policy, f"--region={region}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "setup", "delete"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = worker_settings(yaml.safe_load(args.config.read_text()))
    {"create": create, "setup": setup, "delete": delete}[args.action](settings)


if __name__ == "__main__":
    main()
