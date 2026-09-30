# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one Fast Track dense rung on the asura TPU setup from a local controller.

Create and set up the slice first with ``experiments.asura.tpu_slice``. The reference setup and its
results are in docs/experiments/ledger.md.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import yaml
from fray.current_client import set_current_client
from fray.types import ResourceConfig
from marin.datakit.download.uncheatable_eval import UncheatableEvalDownloadConfig, download_latest_uncheatable_eval
from marin.execution.lazy import ArtifactStep, resolve
from marin.experiment.data import raw_download, tokenized

from experiments.asura.run_dclm import GcpSshClient, GcpWorkerSettings
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS
from experiments.grug.fast_track.launch import H100_LADDER_SIZES, build_h100_ladder_run

# Spot slices can be preempted, so keep recovery checkpoints close together.
RESUME_SAVE_INTERVAL = timedelta(minutes=15)


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


def worker_settings(config: dict) -> GcpWorkerSettings:
    return GcpWorkerSettings(
        project=config["project"],
        zone=config["zone"],
        bucket=config["bucket"],
        prefix=config["prefix"],
        cpu_vm=config["cpu_vm"],
        tpu_vm=config["tpu_vm"],
        accelerator=config["accelerator"],
        wandb_entity=config["wandb_entity"],
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
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    os.environ["MARIN_CLUSTER"] = "asura"
    os.environ["MARIN_PREFIX"] = config["prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = config["bucket"]
    run = build_h100_ladder_run(
        run_id=args.run_id,
        size=args.size,
        dense=True,
        accelerator=config["accelerator"],
        tokenizer=config["tokenizer"],
        train_caches={"dclm": config["train_cache"]},
        validation=validation_sets(config["tokenizer"], config["eval"]),
        num_steps=args.num_steps,
        version=config["version"],
        wandb_entity=config["wandb_entity"],
        wandb_project=config["wandb_project"],
        save_checkpoints=args.num_steps is None,
        resume_save_interval=RESUME_SAVE_INTERVAL,
    )
    print(f"Launching {args.run_id}: {args.size} on {config['accelerator']}", flush=True)
    with set_current_client(GcpSshClient(worker_settings(config))):
        resolve(run, max_concurrent=1)


if __name__ == "__main__":
    main()
