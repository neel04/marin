# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Train an optimized 300M config on MiniPile using an existing v6e-32 TPU slice.

This keeps the same Ray/TPU launch path as the bignode tutorial, but replaces the
large FineWeb cache with a much smaller MiniPile smoke test dataset.
"""

import math
import os

from fray.v2 import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize, default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_300m
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

PROFILE_ENABLED = os.environ.get("MARIN_PROFILE", "").lower() in {"1", "true", "yes", "on"}
PROFILE_START_STEP = int(os.environ.get("MARIN_PROFILE_START_STEP", "5"))
PROFILE_NUM_STEPS = int(os.environ.get("MARIN_PROFILE_NUM_STEPS", "25"))
PROFILE_PERFETTO_LINK = os.environ.get("MARIN_PROFILE_PERFETTO_LINK", "").lower() in {"1", "true", "yes", "on"}

EPOCHS = 2
SEQ_LEN = 512
BATCH_SIZE = 1024
MINIPILE_HF_ID = "JeanKaddour/minipile"
MINIPILE_TRAIN_TOKENS = 1_434_081_494
NUM_TRAIN_STEPS = math.ceil(EPOCHS * MINIPILE_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))

minipile_tokenized = default_tokenize(
    name=MINIPILE_HF_ID,
    dataset=MINIPILE_HF_ID,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
)

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu(
        "v6e-32",
        slice_count=1,
        cpu=32,
        ram="128g",
        disk="50g",
    ),
    train_seq_len=SEQ_LEN,
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=3e-4,
    weight_decay=1e-3,
    max_eval_batches=8,
    skip_bad_steps=True,
    z_loss_weight=1e-4,
    profiler=ProfilerConfig(
        enabled=PROFILE_ENABLED,
        start_step=PROFILE_START_STEP,
        num_steps=PROFILE_NUM_STEPS,
        perfetto_link=PROFILE_PERFETTO_LINK,
    ),
)

llama_300m_minipile_model = default_train(
    name="prof_llama_300M_minipile_bsz_1024",
    tokenized=minipile_tokenized,
    model_config=llama_300m,
    train_config=train_config,
    tags=["opt", "llama", "300m", "minipile", "bignode"],
    eval_harness_tasks=CORE_TASKS,
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[llama_300m_minipile_model])
