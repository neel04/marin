# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Fast Track rungs back to back on one Spot TPU slice that stops billing on its own.

The queue estimates how long its runs take, pads the estimate, and creates the slice with that stop time. It
reprovisions the slice when a preemption leaves hosts missing, retries each run, skips runs that could not
finish before the stop time, and deletes the slice at the end.

    uv run python -m experiments.asura.tpu_queue --config <config.yaml> --size d768 --variant <name> ...
"""

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from experiments.asura.run_dclm import GcpWorkerSettings, tpu_hosts
from experiments.asura.run_fast_track import worker_settings
from experiments.asura.tpu_slice import SLICES, create, delete, setup
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

DENSE = "dense"


@dataclass(frozen=True)
class Run:
    run_id: str
    args: tuple[str, ...] = ()
    """Extra ``run_fast_track`` arguments."""


def run_queue(config: Path, size: str, runs: Sequence[Run]) -> list[Run]:
    """Train ``runs`` in order on a fresh slice and return the ones that finished."""
    if size not in RUN_MINUTES:
        raise ValueError(f"No measured run time for {size}; add one to RUN_MINUTES")

    settings = worker_settings(yaml.safe_load(config.read_text()))
    run_time = timedelta(minutes=RUN_MINUTES[size])
    stop_at = datetime.now(UTC) + PADDING * (timedelta(minutes=PROVISION_MINUTES) + len(runs) * run_time)
    logger.info("Queue of %d %s runs; the slice stops at %s", len(runs), size, stop_at.isoformat())

    launch = [sys.executable, "-m", "experiments.asura.run_fast_track", f"--config={config}", f"--size={size}"]
    finished = []
    try:
        for run in runs:
            if _train_with_retries(run, launch, settings, run_time, stop_at):
                finished.append(run)
    finally:
        delete(settings)

    return finished


def _train_with_retries(
    run: Run, launch: list[str], settings: GcpWorkerSettings, run_time: timedelta, stop_at: datetime
) -> bool:
    """Whether ``run`` finished within ``ATTEMPTS`` attempts that each fit before the slice stops."""
    for attempt in range(1, ATTEMPTS + 1):
        if datetime.now(UTC) + run_time > stop_at:
            logger.warning("Skipping %s: it would not finish before the slice stops", run.run_id)
            return False

        logger.info("%s: attempt %d", run.run_id, attempt)
        if not _ensure_slice(settings, stop_at):
            continue

        # A lost connection hangs the controller, so stop a run at twice its usual time or at the stop time.
        timeout = min(RUN_TIMEOUT_FACTOR * run_time, stop_at - datetime.now(UTC))
        if _train(run, launch, timeout):
            return True

    return False


def _ensure_slice(settings: GcpWorkerSettings, stop_at: datetime) -> bool:
    """Whether the slice has all its hosts, provisioning a fresh one when a preemption took some."""
    host_count = SLICES[settings.accelerator][2]
    try:
        if len(tpu_hosts(settings)) < host_count:
            logger.info("Provisioning the slice")
            delete(settings)
            create(settings, stop_at)
            setup(settings)
    except (subprocess.CalledProcessError, TimeoutError):
        # Spot capacity and networks come and go; a failed provisioning costs one attempt.
        logger.exception("Provisioning failed")
        return False

    return True


def _train(run: Run, launch: list[str], timeout: timedelta) -> bool:
    """Whether ``run`` trained to completion within ``timeout``."""
    command = [*launch, f"--run-id={run.run_id}", *run.args]
    try:
        return subprocess.run(command, timeout=timeout.total_seconds()).returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("%s: stopped after %s", run.run_id, timeout)
        return False


def _variant_run(variant: str, size: str, accelerator: str, date: str) -> Run:
    run_id = f"fasttrack-{variant}-{size}-dclmbpe-{accelerator}-{date}"
    if variant == DENSE:
        return Run(run_id)
    return Run(run_id, (f"--variant={variant}",))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--size", required=True, choices=sorted(RUN_MINUTES))
    parser.add_argument(
        "--variant", action="append", required=True, choices=[DENSE, *sorted(VARIANTS)], help=f"{DENSE}: the reference"
    )
    parser.add_argument("--date", default=f"{datetime.now():%Y%m%d}", help="Date tag in each run id")
    args = parser.parse_args()

    accelerator = worker_settings(yaml.safe_load(args.config.read_text())).accelerator.replace("-", "")
    runs = [_variant_run(variant, args.size, accelerator, args.date) for variant in args.variant]

    finished = run_queue(args.config, args.size, runs)
    logger.info("Finished %d of %d runs", len(finished), len(runs))
    if len(finished) < len(runs):
        sys.exit(1)


if __name__ == "__main__":
    main()
