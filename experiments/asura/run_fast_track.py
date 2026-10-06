# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one Fast Track dense rung on the asura TPU setup from a local controller.

Create and set up the slice first with ``experiments.asura.tpu_slice``. The reference setup and its
results are in docs/experiments/ledger.md.
"""

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml
from fray.current_client import set_current_client
from fray.types import ResourceConfig
from marin.datakit.download.uncheatable_eval import UncheatableEvalDownloadConfig, download_latest_uncheatable_eval
from marin.execution.lazy import ArtifactStep, resolve
from marin.experiment.data import raw_download, tokenized

from experiments.asura.run_dclm import GcpSshClient, GcpWorkerSettings, Provisioning
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS
from experiments.grug.fast_track.launch import H100_LADDER_SIZES, build_h100_ladder_run
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.looped.model import VARIANTS

# Spot slices can be preempted, so keep recovery checkpoints close together.
RESUME_SAVE_INTERVAL = timedelta(minutes=15)


@dataclass(frozen=True)
class OptimizerTrial:
    """One tuning trial's optimizer: multipliers on the heuristic's learning rates, and absolute settings."""

    lr_multiplier: float
    adam_lr_multiplier: float
    momentum: float
    beta1: float
    beta2: float
    warmup: float
    decay: float
    min_lr_ratio: float

    @staticmethod
    def from_json(text: str) -> "OptimizerTrial":
        return OptimizerTrial(**json.loads(text))

    def apply(self, config: GrugMoeMuonHConfig) -> GrugMoeMuonHConfig:
        """The heuristic's ``config`` with this trial's settings: its learning rates scaled, the rest replaced."""
        return dataclasses.replace(
            config,
            learning_rate=config.learning_rate * self.lr_multiplier,
            adam_lr=config.adam_lr * self.adam_lr_multiplier,
            momentum=self.momentum,
            beta1=self.beta1,
            beta2=self.beta2,
            warmup=self.warmup,
            decay=self.decay,
            min_lr_ratio=self.min_lr_ratio,
        )


def validation_sets(tokenizer: str, config: dict) -> list[ArtifactStep]:
    """WikiText-103, every Paloma subset, and the Uncheatable Eval dump pinned to one commit."""
    version = config["version"]
    commit = config["uncheatable_commit"]
    uncheatable_raw = raw_download(
        f"raw/uncheatable_eval-{commit[:7]}",
        fn=download_latest_uncheatable_eval,
        build_config=lambda ctx: UncheatableEvalDownloadConfig(
            output_path=ctx.output_path,
            repo_owner="ziqing-huang",
            repo_name="uncheatable_eval",
            data_path="data",
            branch=commit,
        ),
        version=version,
    )
    return [
        tokenized(
            f"fasttrack-public-wikitext103-{tokenizer}",
            version=version,
            paths=[config["wikitext_raw"]],
            tokenizer=tokenizer,
            validation=True,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="20g"),
        ),
        *paloma_datasets(tokenizer=tokenizer, tag=tokenizer, raw_prefix=config["paloma_raw"], version=version).values(),
        *(
            tokenized(
                f"uncheatable_eval/{subset}-{tokenizer}",
                tokenizer=tokenizer,
                version=version,
                raw=uncheatable_raw,
                glob=glob,
                validation=True,
            )
            for subset, glob in UNCHEATABLE_SUBSETS.items()
        ),
    ]


def worker_settings(config: dict, slice_name: str | None = None) -> GcpWorkerSettings:
    """The config's worker settings, on the slice named ``slice_name`` instead of the config's ``tpu_vm`` if given."""
    return GcpWorkerSettings(
        project=config["project"],
        zone=config["zone"],
        bucket=config["bucket"],
        prefix=config["prefix"],
        cpu_vm=config["cpu_vm"],
        tpu_vm=slice_name or config["tpu_vm"],
        accelerator=config["accelerator"],
        wandb_entity=config["wandb_entity"],
        provisioning=Provisioning(config.get("provisioning", Provisioning.SPOT)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--size", required=True, choices=H100_LADDER_SIZES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Short probe run without permanent checkpoints. Omit for the data-matched rung.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Looped model variant from experiments.grug.looped.model. Omit for the dense reference.",
    )
    parser.add_argument(
        "--optimizer-trial",
        type=OptimizerTrial.from_json,
        help="OptimizerTrial fields as JSON. Omit for the heuristic optimizer.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Model initialization and data order")
    parser.add_argument("--slice", help="Slice name, to run beside another queue; defaults to the config's tpu_vm")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Keep a permanent final checkpoint. Off by default: a rung is cheap to rerun when one is needed.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    os.environ["MARIN_CLUSTER"] = "asura"
    os.environ["MARIN_PREFIX"] = config["prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = config["bucket"]

    run = build_h100_ladder_run(
        run_id=args.run_id,
        size=args.size,
        dense=True,
        model_variant=VARIANTS[args.variant] if args.variant else None,
        optimizer_variant=args.optimizer_trial.apply if args.optimizer_trial else None,
        seed=args.seed,
        accelerator=config["accelerator"],
        tokenizer=config["tokenizer"],
        train_caches={"dclm": config["train_cache"]},
        validation=validation_sets(config["tokenizer"], config["eval"]),
        num_steps=args.num_steps,
        version=config["version"],
        wandb_entity=config["wandb_entity"],
        wandb_project=config["wandb_project"],
        save_checkpoints=args.save_checkpoints,
        resume_save_interval=RESUME_SAVE_INTERVAL,
    )

    print(f"Launching {args.run_id}: {args.size} {args.variant or 'reference'} on {config['accelerator']}", flush=True)
    with set_current_client(GcpSshClient(worker_settings(config, args.slice))):
        resolve(run, max_concurrent=1)


if __name__ == "__main__":
    main()
