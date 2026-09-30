# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the reusable asura DCLM recipe with a local Marin controller and GCP workers."""

import argparse
import math
import os
import shlex
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import yaml
from fray.current_client import set_current_client
from fray.types import JobRequest, JobStatus, ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import resolve
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_1_4b


@dataclass(frozen=True)
class DataSettings:
    cache_name: str
    tokenizer: str
    version: str


@dataclass(frozen=True)
class TrainingSettings:
    target_tokens: int
    batch_size: int
    seq_len: int
    learning_rate: float
    weight_decay: float
    warmup: int
    min_lr_ratio: float
    z_loss_weight: float
    tensor_parallel_size: int
    accelerator: str
    wandb_entity: str
    wandb_project: str


@dataclass(frozen=True)
class RunSettings:
    project: str
    zone: str
    bucket: str
    prefix: str
    cpu_vm: str
    tpu_vm: str
    data: DataSettings
    training: TrainingSettings


@dataclass(frozen=True)
class GcpWorkerSettings:
    project: str
    zone: str
    bucket: str
    prefix: str
    cpu_vm: str
    tpu_vm: str
    """A single-host TPU VM, or a managed instance group whose hosts are named ``<tpu_vm>-<4 chars>``."""
    accelerator: str
    wandb_entity: str


def load_settings(path: Path) -> RunSettings:
    raw = yaml.safe_load(path.read_text())
    data = raw["data"]
    training = raw["training"]
    if training["model"] != "llama_1_4b":
        raise ValueError(f"Unsupported model: {training['model']}")
    return RunSettings(
        project=raw["project"],
        zone=raw["zone"],
        bucket=raw["bucket"],
        prefix=raw["prefix"],
        cpu_vm=raw["cpu_vm"],
        tpu_vm=raw["tpu_vm"],
        data=DataSettings(cache_name=data["cache_name"], tokenizer=data["tokenizer"], version=data["version"]),
        training=TrainingSettings(**{key: value for key, value in training.items() if key != "model"}),
    )


@dataclass
class SshJobHandle:
    job_id: str
    returncode: int
    log_path: Path

    def status(self) -> JobStatus:
        return JobStatus.SUCCEEDED if self.returncode == 0 else JobStatus.FAILED

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        del timeout
        if self.returncode and raise_on_failure:
            lines = self.log_path.read_text(errors="replace").splitlines()[-80:]
            raise RuntimeError(f"Remote job {self.job_id} failed; log {self.log_path}:\n" + "\n".join(lines))
        return self.status()

    def logs(self, max_lines: int = 0) -> tuple[str, ...]:
        lines = self.log_path.read_text(errors="replace").splitlines()
        return tuple(lines[-max_lines:] if max_lines else lines)

    def terminate(self) -> None:
        raise RuntimeError("Stop the active gcloud SSH command after checking the remote training process.")


class GcpSshClient:
    """Dispatch Marin's CPU and TPU steps from this machine to dedicated GCP VMs."""

    def __init__(self, settings: GcpWorkerSettings):
        self.settings = settings

    def submit(self, request: JobRequest, adopt_existing: bool = True) -> SshJobHandle:
        del adopt_existing
        if request.entrypoint.callable_entrypoint is None:
            raise ValueError("The asura dispatcher requires a callable Marin job")
        device = request.resources.device
        if device.kind == "tpu" and device.variant != self.settings.accelerator:
            raise ValueError(f"Unexpected TPU request: {device.variant}")
        if device.kind not in {"cpu", "tpu"}:
            raise ValueError(f"Unsupported worker type: {device.kind}")

        task_id = uuid.uuid4().hex
        local_payload = Path(tempfile.gettempdir()) / f"marin-asura-{task_id}.pkl"
        remote_payload = f"/tmp/marin-asura-{task_id}.pkl"
        payload_uri = f"{self.settings.bucket}/tmp/ttl=1d/dispatch/{task_id}.pkl"
        try:
            with local_payload.open("wb") as destination:
                cloudpickle.dump(request.entrypoint.callable_entrypoint, destination)
            local_payload.chmod(0o600)
            subprocess.run(["gcloud", "storage", "cp", str(local_payload), payload_uri], check=True, capture_output=True)
        finally:
            local_payload.unlink(missing_ok=True)

        quoted_uri = shlex.quote(payload_uri)
        quoted_payload = shlex.quote(remote_payload)
        quoted_prefix = shlex.quote(self.settings.prefix)
        quoted_bucket = shlex.quote(self.settings.bucket)
        quoted_entity = shlex.quote(self.settings.wandb_entity)
        remote_command = (
            "set -e; "
            f"gcloud storage cp {quoted_uri} {quoted_payload}; "
            "cd ~/marin-asura-dclm/repo; "
            f"export MARIN_CLUSTER=asura MARIN_PREFIX={quoted_prefix} MARIN_TEMP_PREFIX={quoted_bucket} "
            f"FSSPEC_GS_TOKEN=cloud WANDB_MODE=online WANDB_ENTITY={quoted_entity} "
            "WANDB_DIR=$HOME/marin-asura-dclm/wandb HF_HOME=/tmp/hf-cache "
            "LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304; "
            "unset JAX_PLATFORMS; "
        )
        if device.kind == "tpu":
            remote_command += 'export WANDB_API_KEY="$(cat "$HOME/.config/wandb/api-key")"; '
        remote_command += f".venv/bin/python -m experiments.asura.worker {quoted_payload}"

        vms = tpu_hosts(self.settings) if device.kind == "tpu" else [self.settings.cpu_vm]
        if device.kind == "tpu" and len(vms) != request.replicas:
            raise ValueError(f"{self.settings.tpu_vm} has {len(vms)} hosts; {device.variant} needs {request.replicas}")

        # Every host of a multi-host slice runs the same program; JAX joins them into one slice.
        def run_on(vm: str) -> tuple[int, Path]:
            log_path = Path(tempfile.gettempdir()) / f"marin-asura-{request.name}-{task_id}-{vm}.log"
            with log_path.open("w") as log:
                result = subprocess.run(
                    [*ssh_command(self.settings, vm), f"--command={remote_command}"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            print(f"Remote job {request.name} on {vm}: exit={result.returncode}; log {log_path}", flush=True)
            return result.returncode, log_path

        print(f"Dispatching {request.name} to {', '.join(vms)}", flush=True)
        with ThreadPoolExecutor(max_workers=len(vms)) as pool:
            results = list(pool.map(run_on, vms))
        returncode, log_path = max(results, key=lambda result: result[0] != 0)
        return SshJobHandle(request.name, returncode, log_path)


def ssh_command(settings: GcpWorkerSettings, vm: str) -> list[str]:
    return [
        "gcloud",
        "compute",
        "ssh",
        vm,
        f"--project={settings.project}",
        f"--zone={settings.zone}",
        "--ssh-flag=-o ServerAliveInterval=30",
        "--quiet",
    ]


def tpu_hosts(settings: GcpWorkerSettings) -> list[str]:
    """The running TPU VM itself, or every running host of the managed instance group named ``tpu_vm``."""
    listing = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={settings.project}",
            f"--zones={settings.zone}",
            f'--filter=name~"^{settings.tpu_vm}(-[a-z0-9]{{4}})?$" AND status=RUNNING',
            "--format=value(name)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(listing.stdout.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    training = settings.training
    num_steps = math.ceil(training.target_tokens / (training.batch_size * training.seq_len))
    actual_tokens = num_steps * training.batch_size * training.seq_len

    os.environ["MARIN_CLUSTER"] = "asura"
    os.environ["MARIN_PREFIX"] = settings.prefix
    os.environ["MARIN_TEMP_PREFIX"] = settings.bucket

    data = tokenized(
        name=settings.data.cache_name,
        version=settings.data.version,
        paths=[f"{settings.prefix}/input/*.jsonl.zst"],
        tokenizer=settings.data.tokenizer,
        sample_count=None,
        resources=ResourceConfig.with_cpu(cpu=16, ram="60g", disk="80g"),
    )
    worker_settings = GcpWorkerSettings(
        project=settings.project,
        zone=settings.zone,
        bucket=settings.bucket,
        prefix=settings.prefix,
        cpu_vm=settings.cpu_vm,
        tpu_vm=settings.tpu_vm,
        accelerator=training.accelerator,
        wandb_entity=training.wandb_entity,
    )
    with set_current_client(GcpSshClient(worker_settings)):
        built_data = resolve(data, max_concurrent=1)
        available_tokens = built_data.num_train_tokens
        print(f"Distinct DCLM tokens in cache: {available_tokens:,}", flush=True)
        if available_tokens < actual_tokens:
            raise ValueError(f"Dataset has {available_tokens:,} tokens but this run requires {actual_tokens:,}")

        checkpoint = train_lm(
            name=f"checkpoints/asura-dclm-1p4b/{args.run_id}",
            version=settings.data.version,
            model=llama_1_4b,
            optimizer=AdamConfig(
                learning_rate=training.learning_rate,
                weight_decay=training.weight_decay,
                warmup=training.warmup,
                min_lr_ratio=training.min_lr_ratio,
                lr_schedule="cosine",
            ),
            datasets={data: 1.0},
            batch_size=training.batch_size,
            seq_len=training.seq_len,
            num_train_steps=num_steps,
            z_loss_weight=training.z_loss_weight,
            evals=None,
            resources=ResourceConfig.with_tpu(training.accelerator, disk="100g"),
            tensor_parallel_size=training.tensor_parallel_size,
            run_id=args.run_id,
            wandb_project=training.wandb_project,
            tags=["dclm", "1.4b", training.accelerator, "asura"],
            env_vars={"WANDB_MODE": "online", "WANDB_ENTITY": training.wandb_entity},
        )
        print(
            f"Training {actual_tokens:,} tokens in {num_steps:,} steps on {training.accelerator}; "
            f"W&B: {training.wandb_entity}/{training.wandb_project}/{args.run_id}",
            flush=True,
        )
        resolve(checkpoint, max_concurrent=1)


if __name__ == "__main__":
    main()
