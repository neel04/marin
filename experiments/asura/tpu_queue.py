# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Fast Track rungs back to back on one Spot TPU slice that stops billing on its own.

The queue estimates how long its runs take, pads the estimate, and creates the slice with that stop time. It
reprovisions the slice when a preemption leaves hosts missing, retries each run, skips runs that could not
finish before the stop time, and deletes the slice at the end.

The queue snapshots this checkout when it starts. Every slice setup installs the snapshot, and each run's controller
builds its configuration from the snapshot's ``experiments``, so edits made while the queue runs reach none of its
runs. The controller still imports the libraries under ``lib/`` from the live checkout.

    uv run python -m experiments.asura.tpu_queue --config <config.yaml> --size d768 --variant <name> ...
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from experiments.asura.run_dclm import GcpWorkerSettings, tpu_hosts
from experiments.asura.run_fast_track import worker_settings
from experiments.asura.tpu_slice import SLICES, archive_checkout, create, delete, setup, wait_until_ready
from experiments.grug.looped.model import VARIANTS

logger = logging.getLogger(__name__)

# Measured controller wall time of one data-matched rung on v5p-32, including compile and eval.
RUN_MINUTES = {"d768": 9, "d1024": 30}
# Slice creation and setup, including a short wait for Spot capacity.
PROVISION_MINUTES = 10

# The slice stops at PADDING times the queue's estimated length; a single run stops at RUN_TIMEOUT_FACTOR times its own.
PADDING = 1.5
RUN_TIMEOUT_FACTOR = 2
ATTEMPTS = 3
# A host the instance group restarts, after a preemption or a host error, runs before it accepts SSH again; this long
# for it to boot before the slice is replaced. Its disk, and the setup on it, survive the restart.
RESTART_GRACE = timedelta(minutes=5)

DENSE = "dense"


@dataclass(frozen=True)
class Run:
    run_id: str
    args: tuple[str, ...] = ()
    """Extra ``run_fast_track`` arguments."""


def run_queue(
    config: Path, size: str, runs: Sequence[Run], run_minutes: float, slice_name: str | None = None
) -> list[Run]:
    """Train ``runs`` in order on a fresh slice and return the ones that finished.

    ``run_minutes`` is one run's expected controller wall time, which sets each run's timeout and the slice's stop time.
    ``slice_name`` replaces the config's ``tpu_vm``, so that two queues can run side by side on separate slices.
    """
    settings = worker_settings(yaml.safe_load(config.read_text()), slice_name)
    run_time = timedelta(minutes=run_minutes)
    stop_at = datetime.now(UTC) + PADDING * (timedelta(minutes=PROVISION_MINUTES) + len(runs) * run_time)
    logger.info("Queue of %d %s runs; the slice stops at %s", len(runs), size, stop_at.isoformat())

    finished = []
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = _Snapshot.take(Path(tmp), config)
        launch = [
            sys.executable,
            "-m",
            "experiments.asura.run_fast_track",
            f"--config={snapshot.config}",
            f"--size={size}",
        ]
        if slice_name:
            launch.append(f"--slice={slice_name}")
        try:
            for run in runs:
                if _train_with_retries(run, launch, snapshot, settings, run_time, stop_at):
                    finished.append(run)
        finally:
            delete(settings)

    return finished


@dataclass(frozen=True)
class _Snapshot:
    """This checkout as the queue started: an archive for the hosts, its extraction for the controller, and a copy of
    the queue's config, which the controller reads by its absolute path, wherever the original lives."""

    archive: Path
    root: Path
    config: Path

    @staticmethod
    def take(directory: Path, config: Path) -> "_Snapshot":
        archive = directory / "source.tar.gz"
        archive_checkout(archive)

        root = directory / "source"
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter=_extracted_now)

        copied = directory / "queue-config.yaml"
        shutil.copyfile(config, copied)
        return _Snapshot(archive, root, copied.resolve())


def _extracted_now(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
    """The ``data`` filter, stamping each file with the time of extraction. macOS empties files older than three days
    from the temporary directory every night, and a checkout's unchanged files keep their old times in the archive."""
    return tarfile.data_filter(member, destination).replace(mtime=time.time())


def _train_with_retries(
    run: Run,
    launch: list[str],
    snapshot: _Snapshot,
    settings: GcpWorkerSettings,
    run_time: timedelta,
    stop_at: datetime,
) -> bool:
    """Whether ``run`` finished within ``ATTEMPTS`` attempts that each fit before the slice stops."""
    for attempt in range(1, ATTEMPTS + 1):
        if datetime.now(UTC) + run_time > stop_at:
            logger.warning("Skipping %s: it would not finish before the slice stops", run.run_id)
            return False

        logger.info("%s: attempt %d", run.run_id, attempt)
        if not _ensure_slice(settings, snapshot, stop_at):
            continue

        # A lost connection hangs the controller, so stop a run at twice its usual time or at the stop time.
        timeout = min(RUN_TIMEOUT_FACTOR * run_time, stop_at - datetime.now(UTC))
        if _train(run, launch, snapshot, timeout):
            return True

    return False


def _ensure_slice(settings: GcpWorkerSettings, snapshot: _Snapshot, stop_at: datetime) -> bool:
    """Whether the slice has all its hosts reachable, provisioning a fresh one when a preemption took some or a
    restarted host does not come back."""
    host_count = SLICES[settings.accelerator][2]
    try:
        if _hosts_ready(settings, host_count):
            return True

        logger.info("Provisioning the slice")
        delete(settings)
        create(settings, stop_at)
        setup(settings, snapshot.archive)
    except (subprocess.CalledProcessError, TimeoutError):
        # Spot capacity and networks come and go; a failed provisioning costs one attempt.
        logger.exception("Provisioning failed")
        return False

    return True


def _hosts_ready(settings: GcpWorkerSettings, host_count: int) -> bool:
    """Whether all ``host_count`` hosts run and accept SSH, waiting ``RESTART_GRACE`` for a restarted host to boot."""
    if len(tpu_hosts(settings)) < host_count:
        return False

    try:
        wait_until_ready(settings, host_count, RESTART_GRACE.total_seconds())
    except TimeoutError:
        logger.warning("A host still refuses SSH after %s", RESTART_GRACE)
        return False

    return True


def _train(run: Run, launch: list[str], snapshot: _Snapshot, timeout: timedelta) -> bool:
    """Whether ``run`` trained to completion within ``timeout``, controlled from the snapshot."""
    command = [*launch, f"--run-id={run.run_id}", *run.args]
    try:
        # ``python -m`` imports from its working directory first, so the controller loads the snapshot's experiments.
        completed = subprocess.run(command, cwd=snapshot.root, timeout=timeout.total_seconds())
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("%s: stopped after %s", run.run_id, timeout)
        return False


def _variant_run(
    variant: str, size: str, accelerator: str, date: str, tag: str, seed: int, extra_args: tuple[str, ...]
) -> Run:
    # Seed 0 keeps the untagged run id, as in tune_fast_track.
    name = "-".join(part for part in (variant, tag, f"s{seed}" if seed else "") if part)
    run_id = f"fasttrack-{name}-{size}-dclmbpe-{accelerator}-{date}"
    extra_args = (*extra_args, f"--seed={seed}")
    if variant == DENSE:
        return Run(run_id, extra_args)
    return Run(run_id, (f"--variant={variant}", *extra_args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--size", required=True, choices=sorted(RUN_MINUTES))
    parser.add_argument(
        "--variant", action="append", required=True, choices=[DENSE, *sorted(VARIANTS)], help=f"{DENSE}: the reference"
    )
    parser.add_argument("--date", default=f"{datetime.now():%Y%m%d}", help="Date tag in each run id")
    parser.add_argument("--optimizer-trial", help="run_fast_track's --optimizer-trial JSON, for every run")
    parser.add_argument("--tag", default="", help="Added after the variant in every run id, e.g. to name the optimizer")
    parser.add_argument("--seed", type=int, default=0, help="run_fast_track's --seed, for every run")
    parser.add_argument(
        "--save-checkpoints", action="store_true", help="run_fast_track's --save-checkpoints, for every run"
    )
    parser.add_argument("--slice", help="Slice name, to run beside another queue; defaults to the config's tpu_vm")
    parser.add_argument(
        "--run-minutes",
        type=float,
        help="One run's expected wall time, for variants slower than the measured rung; defaults to RUN_MINUTES",
    )
    args = parser.parse_args()

    accelerator = worker_settings(yaml.safe_load(args.config.read_text())).accelerator.replace("-", "")
    extra_args = (f"--optimizer-trial={args.optimizer_trial}",) if args.optimizer_trial else ()
    if args.save_checkpoints:
        extra_args = (*extra_args, "--save-checkpoints")
    runs = [
        _variant_run(variant, args.size, accelerator, args.date, args.tag, args.seed, extra_args)
        for variant in args.variant
    ]

    run_minutes = args.run_minutes or RUN_MINUTES[args.size]
    finished = run_queue(args.config, args.size, runs, run_minutes, args.slice)
    logger.info("Finished %d of %d runs", len(finished), len(runs))
    if len(finished) < len(runs):
        sys.exit(1)


if __name__ == "__main__":
    main()
