# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the pinned public DCLM shards for the Asura Fast Track run."""

import argparse
import os
from pathlib import Path

import yaml
from fray.types import ResourceConfig
from huggingface_hub import HfApi
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.log_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    os.environ["MARIN_CLUSTER"] = "asura"
    os.environ["MARIN_PREFIX"] = config["prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = config["bucket"]

    revision = HfApi().model_info(config["tokenizer"]).sha
    if revision != config["tokenizer_revision"]:
        raise ValueError(f"Tokenizer revision changed: {revision}")

    options = config["tokenize"]
    tokenize(
        TokenizeConfig(
            train_paths=config["train_input_paths"],
            validation_paths=[],
            cache_path=config["train_cache"],
            tokenizer=config["tokenizer"],
            max_workers=options["workers"],
            num_shards=options["shards"],
            levanter_batch_size=options["batch_size"],
            worker_resources=ResourceConfig(cpu=2, ram="6g", disk="5g"),
        )
    )
    stats = read_tokenized_cache_stats(config["train_cache"], "train")
    print(f"DCLM tokenization complete: {stats.total_tokens:,} tokens", flush=True)


if __name__ == "__main__":
    configure_logging()
    main()
