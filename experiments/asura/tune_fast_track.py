# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune a d768 Fast Track rung's optimizer with Vizier on the asura TPU slice.

Each round asks Vizier's default designer (GP-UCB-PE, batched Gaussian-process Bayesian optimization) for a batch of
``OptimizerTrial`` settings, trains the batch as one TPU queue, and reports each run's Paloma macro loss back. The
rung's run with the heuristic optimizer seeds the study. The study lives in a local SQLite file, so rerunning the
command resumes it: Vizier hands the client its unfinished trials again, and runs that already finished are read,
not retrained. With ``--seeds``, it instead retrains the best trial with other seeds to measure the loss's spread.

    uv run --with-requirements experiments/asura/vizier-requirements.txt \\
        python -m experiments.asura.tune_fast_track --config <config.yaml> --study <name> --db <path.sqlite> \\
        [--variant <looped variant> --reference-run <its heuristic run id>]
"""

import argparse
import dataclasses
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import wandb
import yaml
from vizier.service import clients
from vizier.service import pyvizier as vz

from experiments.asura.run_fast_track import OptimizerTrial
from experiments.asura.tpu_queue import RUN_MINUTES, Run, run_queue
from experiments.grug.looped.model import VARIANTS

logger = logging.getLogger(__name__)

SIZE = "d768"
METRIC = "eval/paloma/macro_loss"
DENSE_REFERENCE_RUN = "fasttrack-dense-d768-dclmbpe-v5p32-20260928"
# A new study's first trial is the reference run.
REFERENCE_TRIAL_ID = 1
# The heuristic optimizer, as a trial; beta2 is MoeHeuristic's at 128 x 4096 tokens per batch.
HEURISTIC = OptimizerTrial(
    lr_multiplier=1.0,
    adam_lr_multiplier=1.0,
    momentum=0.95,
    beta1=0.9,
    beta2=0.999**4,
    warmup=0.01,
    decay=1.0,
    min_lr_ratio=0.05,
)
SEARCH_SPACE = {
    "lr_multiplier": (0.25, 4.0, vz.ScaleType.LOG),
    "adam_lr_multiplier": (0.25, 4.0, vz.ScaleType.LOG),
    "momentum": (0.85, 0.98, vz.ScaleType.LINEAR),
    "beta1": (0.8, 0.98, vz.ScaleType.LINEAR),
    "beta2": (0.95, 0.9995, vz.ScaleType.REVERSE_LOG),
    "warmup": (0.0, 0.1, vz.ScaleType.LINEAR),
    "decay": (0.1, 1.0, vz.ScaleType.LINEAR),
    "min_lr_ratio": (0.0, 0.3, vz.ScaleType.LINEAR),
}
assert SEARCH_SPACE.keys() == {field.name for field in dataclasses.fields(OptimizerTrial)}


@dataclass(frozen=True)
class Sweep:
    """What a study trains: a d768 rung on the asura ladder config, optionally as a looped variant."""

    config: Path
    variant: str | None
    reference_run: str
    """The rung's W&B run with the heuristic optimizer, which seeds the study as its first trial."""
    accelerator: str
    wandb_path: str

    @staticmethod
    def load(config: Path, variant: str | None, reference_run: str) -> "Sweep":
        settings = yaml.safe_load(config.read_text())
        accelerator = settings["accelerator"].replace("-", "")
        wandb_path = f"{settings['wandb_entity']}/{settings['wandb_project']}"
        return Sweep(config, variant, reference_run, accelerator, wandb_path)

    def run(self, client_id: str, trial: clients.Trial, seed: int = 0) -> Run:
        """The TPU run of ``trial`` with ``seed``; seed 0 is the run that measured the trial."""
        if trial.id == REFERENCE_TRIAL_ID and seed == 0:
            return Run(self.reference_run)

        seed_tag = f"-s{seed}" if seed else ""
        run_id = f"fasttrack-{client_id}-t{trial.id:03d}{seed_tag}-{SIZE}-dclmbpe-{self.accelerator}"

        args = [f"--optimizer-trial={json.dumps(dict(trial.parameters))}", f"--seed={seed}"]
        if self.variant:
            args.append(f"--variant={self.variant}")

        return Run(run_id, tuple(args))

    def final_loss(self, run_id: str) -> float | None:
        """The run's final Paloma macro loss, or None if it has not finished."""
        runs = wandb.Api().runs(self.wandb_path, filters={"display_name": run_id, "state": "finished"})
        if not runs:
            return None
        return runs[0].summary[METRIC]

    def train_unfinished(self, runs: list[Run]) -> None:
        """Train, as one TPU queue, the runs that have not finished in W&B."""
        unfinished = [run for run in runs if self.final_loss(run.run_id) is None]
        run_queue([self.config], SIZE, unfinished, RUN_MINUTES[SIZE])


def open_study(study_id: str, db: Path, sweep: Sweep) -> clients.Study:
    """Load the study from ``db``, creating it and seeding it with the sweep's reference run if it is new."""
    clients.environment_variables.servicer_kwargs["database_url"] = f"sqlite:///{db}"

    config = vz.StudyConfig(algorithm="DEFAULT")
    for name, (low, high, scale) in SEARCH_SPACE.items():
        config.search_space.root.add_float_param(name, low, high, scale_type=scale)
    config.metric_information.append(vz.MetricInformation(METRIC, goal=vz.ObjectiveMetricGoal.MINIMIZE))

    study = clients.Study.from_study_config(config, owner="marin", study_id=study_id)
    if not list(study.trials()):
        reference_loss = sweep.final_loss(sweep.reference_run)
        if reference_loss is None:
            raise ValueError(f"{sweep.reference_run} has not finished, so it cannot seed the study")

        reference = vz.Trial(parameters=dataclasses.asdict(HEURISTIC))
        reference.complete(vz.Measurement({METRIC: reference_loss}))
        added = study.add_trial(reference)
        assert added.id == REFERENCE_TRIAL_ID

    return study


def tune(sweep: Sweep, study: clients.Study, client_id: str, trials: int, round_size: int) -> None:
    """Run rounds of suggested trials until ``trials`` have completed."""
    while _completed_trials(study) < trials:
        count = min(round_size, trials - _completed_trials(study))
        suggestions = study.suggest(count=count, client_id=client_id)
        runs = {trial.id: sweep.run(client_id, trial) for trial in suggestions}
        sweep.train_unfinished(list(runs.values()))

        finished = 0
        for trial in suggestions:
            loss = sweep.final_loss(runs[trial.id].run_id)
            if loss is None:
                # Left active, so Vizier suggests it to this client again next round.
                continue

            _complete(trial, loss)
            finished += 1

        if finished == 0:
            raise RuntimeError("No trial in this round finished; check the slice before resuming")


def replicate(sweep: Sweep, study: clients.Study, client_id: str, seeds: list[int]) -> None:
    """Retrain the study's best trial with each of ``seeds`` and log its loss across seeds, seed 0 included."""
    best = next(iter(study.optimal_trials()))
    runs = {seed: sweep.run(client_id, best, seed) for seed in sorted({0, *seeds})}
    sweep.train_unfinished(list(runs.values()))

    losses = {seed: sweep.final_loss(run.run_id) for seed, run in runs.items()}
    finished = [loss for loss in losses.values() if loss is not None]
    logger.info(
        "Trial %d over %d seeds: %s = %s; mean %.4f, standard deviation %.4f",
        best.id,
        len(finished),
        METRIC,
        losses,
        statistics.mean(finished),
        statistics.stdev(finished),
    )


def _completed_trials(study: clients.Study) -> int:
    completed = study.trials(vz.TrialFilter(status=[vz.TrialStatus.COMPLETED]))
    return len(list(completed))


def _complete(trial: clients.Trial, loss: float) -> None:
    """Report ``loss`` to Vizier; a diverged run is an infeasible trial."""
    if math.isfinite(loss):
        trial.complete(vz.Measurement({METRIC: loss}))
    else:
        trial.complete(infeasible_reason=f"{METRIC} is {loss}")

    logger.info("Trial %d: %s = %s with %s", trial.id, METRIC, loss, dict(trial.parameters))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Vizier's designers log every step of their own optimizers through absl.
    logging.getLogger("absl").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", required=True, help="Study name; also the client id and run-id prefix")
    parser.add_argument("--db", type=Path, required=True, help="Local SQLite file holding the study")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="Looped variant to tune. Omit for the dense rung.")
    parser.add_argument(
        "--reference-run",
        default=DENSE_REFERENCE_RUN,
        help="The rung's W&B run with the heuristic optimizer, which seeds a new study",
    )
    parser.add_argument("--trials", type=int, default=28, help="Completed trials to reach, the reference included")
    parser.add_argument("--round-size", type=int, default=9, help="Trials per TPU queue")
    parser.add_argument("--seeds", type=int, nargs="+", help="Instead of tuning, retrain the best trial with these")
    args = parser.parse_args()

    if args.variant and args.reference_run == DENSE_REFERENCE_RUN:
        parser.error("--variant needs --reference-run: the variant's own run with the heuristic optimizer")

    sweep = Sweep.load(args.config, args.variant, args.reference_run)
    study = open_study(args.study, args.db, sweep)

    if args.seeds:
        replicate(sweep, study, args.study, args.seeds)
        return

    tune(sweep, study, args.study, args.trials, args.round_size)
    for trial in study.optimal_trials():
        best = trial.materialize()
        logger.info("Best: %s = %s with %s", METRIC, best.final_measurement.metrics[METRIC].value, best.parameters)


if __name__ == "__main__":
    main()
