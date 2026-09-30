# Fast Track TPU ledger

Best Fast Track results on the asura-0 TPU setup. Compare a new variant against the reference row for its
rung, trained on the same data. The TPU reference rows do not use the Hero data mix, so do not compare them
with the H100 rows in the [Fast Track README](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/README.md).

## Leaderboard

Paloma loss is the comparison metric, as in the Fast Track
[agent guide](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/agent.md). Throughput is the mean over the last 100 steps.
Wall time is the W&B run duration. It excludes slice provisioning and worker setup.

| Rung | Run | Tokens | Paloma loss | Paloma bpb | Uncheatable bpb | Tokens/s | MFU | Wall time |
|------|-----|-------:|------------:|-----------:|----------------:|---------:|----:|----------:|
| d768 | reference ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-dense-d768-dclmbpe-v5p32-20260928)) | 1.07B | 3.2548 | 1.3275 | 1.1478 | 4.25M | 34.3% | 6.0 min |
| d1024 | reference ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-dense-d1024-dclmbpe-v5p32-20260928)) | 2.89B | 2.9897 | 1.2219 | 1.0193 | 2.11M | 38.9% | 25.2 min |

## Reference setup

Both reference rows use this configuration. A variant must change only the thing it tests.

- Model: dense rungs from `build_h100_ladder_run` with `accelerator="v5p-32"` and the default data match
  (20 tokens per active parameter at the rung's baseline batch). d768 is 2,040 steps at batch 128; d1024 is
  2,760 steps at batch 256. Sequence length is 4,096.
- Attention: local layers use the 2,048-token sliding window through `CausalSelfAttention._attend`. TPU runs
  from before 2026-09-27 ran full causal attention on local layers.
- TPU kernels: Splash attention uses 2,048-token blocks with 1,024-wide compute tiles and the fused backward
  kernel. Layer remat keeps the Splash residuals, and the loss uses the Pallas cross-entropy kernel. Against
  materialized reference attention with XLA cross-entropy, 100 d768 steps from the same data and
  initialization differed by at most 0.0052 in training loss, a relative 9.8e-4.
- Data: `mlfoundations/dclm-baseline-1.0` at revision `a3b142c183aebe5af344955ae20836eb34dcf69b`, files
  `global-shard_01_of_10/local-shard_1_of_10/shard_00000000` through `shard_00000055`. That is 5,130,610
  documents. The cache is `tokenized/fasttrack-dclm-bpe16k-shuf-20260928/train` (6,783,798,009 tokens),
  shuffled at document level with `experiments/grug/fast_track/shuffle_cache.py` into 128 buckets.
- Tokenizer: `dclm-bpe-v16384`, from the `hero-bpe-v16384` recipe in
  [#9188](https://github.com/marin-community/marin/issues/9188). It uses the Llama 3 split pattern with
  digits grouped in runs of up to three, byte-level BPE, and the eight Llama 3/Marin special tokens at
  ids 0-7. One 131,072-entry BPE was trained on an 8 GB sample of the 56 shards, then truncated to its
  first 16,120 merges. BOS and EOS are appended to each document.
- Evaluation: all 16 Paloma subsets from the `allenai/paloma` validation split at revision `65cd6fc`, and
  the 7 Uncheatable Eval subsets from `ziqing-huang/uncheatable_eval` at commit `185c463` (the 2025-09-01 to
  2025-09-14 dumps). Evaluation runs at the midpoint and the end of training.
- Hardware: a Spot v5p-32 slice, four `ct5p-hightpu-4t` VMs in one managed instance group in
  `europe-west4-b`. A local controller dispatches the job to each host over SSH.

The setup is [`fast_track_dense_ladder.yaml`](https://github.com/marin-community/marin/blob/main/experiments/asura/fast_track_dense_ladder.yaml);
the Fast Track README has the launch commands. Data paths are under
`gs://marin-asura-0-europe-west4/fasttrack-public-20260926`. The scripts that restored the shards, trained
the tokenizer, and built the caches are under `tpu-ladder/scripts/` there. The tokenizer mirror is
`tokenizers/dclm-bpe-v16384/hf-hub-1.21.0/`. Final checkpoints are under
`grug/fasttrack-dense-{d768,d1024}-dclmbpe-v5p32-20260928/2026.09.28/checkpoints/`.

## Relation to the H100 baseline

The H100 dense baselines trained on the Hero step-108k mixture with `hero-bpe-v16384`. That mixture draws
about 39% of its sampling weight from gated NVIDIA Nemotron datasets, so this setup substitutes public DCLM.

| Rung | H100 Paloma loss | TPU Paloma loss | H100 Paloma bpb | TPU Paloma bpb | H100 uncheatable bpb | TPU uncheatable bpb |
|------|-----------------:|----------------:|----------------:|---------------:|---------------------:|--------------------:|
| d768 | 3.283 | 3.255 | 1.361 | 1.328 | 1.063 | 1.148 |
| d1024 | 3.006 | 2.990 | 1.248 | 1.222 | 0.945 | 1.019 |

The H100 dense scaling fit is `L = 1.2 + 66.70 * C^-0.0844`, with `C` in training FLOPs. On that fit the
H100 recipe needs 1.28x (d768) and 1.06x (d1024) the rung's FLOPs to reach the TPU losses. The gap depends on the subset. At
d1024 the TPU run is 0.06 to 0.13 bpb better on C4, Falcon RefinedWeb, Dolma subreddits, AO3, and BBC
News. It is worse on code: GitHub Python by 0.39 bpb, GitHub C++ by 0.34, and Paloma programming
languages by 0.21. DCLM baseline is filtered web text with little code.

## Throughput

The d1280 runs below trained 5,230,297,088 tokens at batch 512 with the TPU kernel patch. They predate the
sliding-window fix, so their losses are not comparable with the leaderboard.

| Hardware | Tokens/s | Tokens/s per chip | MFU | Wall time | W&B |
|----------|---------:|------------------:|----:|----------:|-----|
| v5p-32 | 1.440M | 90.0k | 44.2% | 62.6 min | [run](https://wandb.ai/neel/marin/runs/fasttrack-dense-d1280-dclm-v5p32-b512-opt-20260927) |
| v6e-16 | 1.168M | 73.0k | 18.0% | 77.0 min | [run](https://wandb.ai/neel/marin/runs/fasttrack-dense-d1280-dclm-v6e16-b512-opt-20260927) |

## Add an entry

1. Launch the variant with `experiments.asura.run_fast_track` and the ladder config, which fixes the
   reference cache, tokenizer, evaluation sets, batch, and step count for the rung. Change only the variant.
2. Add a row with the W&B link and the same columns. Mark it in the Run column; keep the reference rows.
3. For an effective-speedup estimate, use `effective_speedup` from the
   [agent guide](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/agent.md). Use `L_inf = 1.2` and `alpha = 0.0844` from
   the H100 dense fit until a four-rung TPU fit exists.
