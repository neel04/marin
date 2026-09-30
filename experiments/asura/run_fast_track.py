# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the public-source Fast Track dense rung from a local GCP controller."""

import argparse
import os
from pathlib import Path

import yaml
from fray.current_client import set_current_client
from fray.types import ResourceConfig
from marin.execution.lazy import resolve
from marin.experiment.data import tokenized
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats

from experiments.asura.run_dclm import GcpSshClient, GcpWorkerSettings
from experiments.grug.fast_track.launch import SEQ_LEN, build_h100_ladder_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    os.environ["MARIN_CLUSTER"] = "asura"
    os.environ["MARIN_PREFIX"] = config["prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = config["bucket"]
    settings = GcpWorkerSettings(
        project=config["project"],
        zone=config["zone"],
        bucket=config["bucket"],
        prefix=config["prefix"],
        cpu_vm=config["cpu_vm"],
        tpu_vm=config["tpu_vm"],
        accelerator=config["accelerator"],
        wandb_entity=config["wandb_entity"],
    )

    required_tokens = config["batch_size"] * config["num_steps"] * SEQ_LEN
    cache = config["train_cache"].rstrip("/")
    tokens = read_tokenized_cache_stats(cache, "train").total_tokens
    if tokens < required_tokens:
        raise ValueError(f"DCLM cache has only {tokens:,} distinct tokens")
    train_caches = {"dclm": f"{cache}/train"}
    print(f"DCLM: {tokens:,} tokens at {cache}/train", flush=True)

    validation = tokenized(
        "fasttrack-public-wikitext103-16k",
        version=config["version"],
        paths=[config["validation_path"]],
        tokenizer=config["tokenizer"],
        validation=True,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="20g"),
    )
    run = build_h100_ladder_run(
        run_id=config["run_id"],
        size="d1280",
        dense=True,
        accelerator=config["accelerator"],
        tokenizer=config["tokenizer"],
        train_caches=train_caches,
        validation=[validation],
        batch_size=config["batch_size"],
        num_steps=config["num_steps"],
        version=config["version"],
        wandb_entity=config["wandb_entity"],
        wandb_project=config["wandb_project"],
        save_checkpoints=True,
    )
    print(f"Launching {config['run_id']} for {required_tokens:,} tokens on {config['accelerator']}", flush=True)
    with set_current_client(GcpSshClient(settings)):
        resolve(run, max_concurrent=1)


if __name__ == "__main__":
    main()
