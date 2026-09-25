# DCLM 1.4B on asura-0

The [500M-token recipe](../../experiments/asura/dclm_1p4b_500m_bsz512.yaml) trains a 1.4B Llama model on the prepared DCLM cache at global batch 512. The Marin controller runs in this checkout. It dispatches training to a Spot `ct5p-hightpu-4t` VM in `europe-west4-b`; this `v5p-8` host has four TPU chips. Checkpoints and the data cache live in `gs://marin-asura-0-europe-west4`. W&B runs go to `neel/marin`.

The cache contains 3,472,026,691 tokens from 2,872,638 documents. Its source is `mlfoundations/dclm-baseline-1.0` at revision `a3b142c183aebe5af344955ae20836eb34dcf69b`. The staged input and `staging_manifest.json` remain under `gs://marin-asura-0-europe-west4/dclm-1p4b-2p8b/2026-09-25/input/`.

## Start a worker

Create a Spot TPU VM with the name and zone in the recipe:

```sh
gcloud compute instances create marin-dclm-v5p8-20260925 \
  --project=asura-0 --zone=europe-west4-b \
  --machine-type=ct5p-hightpu-4t \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --image-project=ubuntu-os-accelerator-images \
  --boot-disk-size=100GB --scopes=cloud-platform \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE --max-run-duration=43200s
```

The completed 2.8B-token run left worker source archives in the same regional bucket. Restore them on the new VM:

```sh
gcloud compute ssh marin-dclm-v5p8-20260925 \
  --project=asura-0 --zone=europe-west4-b \
  --command='mkdir -p ~/marin-asura-dclm/repo && \
    gcloud storage cp gs://marin-asura-0-europe-west4/dclm-1p4b-2p8b/2026-09-25/source/marin-source-0bc6781.tar.gz /tmp/marin-source.tar.gz && \
    gcloud storage cp gs://marin-asura-0-europe-west4/dclm-1p4b-2p8b/2026-09-25/source/asura-experiment-20260925.tar.gz /tmp/asura-experiment.tar.gz && \
    gcloud storage cp gs://marin-asura-0-europe-west4/dclm-1p4b-2p8b/2026-09-25/source/asura-transition-source-20260925.tar.gz /tmp/asura-transition-source.tar.gz && \
    tar -xzf /tmp/asura-experiment.tar.gz -C ~/marin-asura-dclm/repo && \
    bash ~/marin-asura-dclm/repo/experiments/asura/setup_worker.sh /tmp/marin-source.tar.gz && \
    tar -xzf /tmp/asura-transition-source.tar.gz -C ~/marin-asura-dclm/repo'
```

Keep the cluster config at `~/.config/marin/clusters/asura.yaml` on both the local controller and VM. Copy the W&B API key to `~/.config/wandb/api-key` on the VM with mode `0600`. Keep the key out of GCS and source archives. [`setup_worker.sh`](../../experiments/asura/setup_worker.sh) installs the pinned environment and prints the four JAX devices.

## Launch

The prepared cache must be present. Each new run needs a distinct ID; reuse that ID when resuming its checkpoints and W&B history. Launch from this local checkout:

```sh
uv run --no-sync python -m experiments.asura.run_dclm \
  --config experiments/asura/dclm_1p4b_500m_bsz512.yaml \
  --run-id <new-run-id>
```

The launcher runs 477 optimizer steps over 500,170,752 tokens. Warmup lasts 22 steps, or 23,068,672 tokens. Change the YAML and use another run ID for a different batch size or token budget. Keep the local controller and its SSH connection alive until training finishes.

The [completed 2.8B-token run](https://wandb.ai/neel/marin/runs/dclm-1p4b-2p8b-v5p8-spot-20260925) ended at 138,628 tokens/s across the four-chip host, with 59.07% median MFU during its batch-512 continuation. At that rate, 500,170,752 tokens take about 60 minutes. Allow roughly 65–75 minutes from worker start for compilation and checkpointing; VM setup adds time. The 500M-token recipe has not been timed directly.

## Recover and finish

Check W&B, the local controller log, and the remote worker log. Temporary recovery checkpoints are under `gs://marin-asura-0-europe-west4/tmp/ttl=14d/checkpoints-temp/`; a complete checkpoint has `metadata.json`. The final permanent checkpoint is under the run's `checkpoints/` output path.

If the local SSH connection drops, check the remote `experiments.asura.worker` process, W&B step growth, and the latest checkpoint before relaunching. A remote worker may continue after the local controller reports an SSH failure. If Spot deletes the VM, create a replacement with the same name and zone, restore the worker and credentials, then relaunch with the same run ID. Delete the VM after the permanent checkpoint is present.

The completed 2.8B-token run finished 4,388 steps and 2,801,008,640 tokens in 5 hours 54 minutes, with final loss 3.300. Its checkpoint, source archives, logs, and run manifest remain in the bucket under `dclm-1p4b-2p8b/2026-09-25/`.
