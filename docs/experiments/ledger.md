# Fast Track TPU ledger

Best Fast Track results on the asura-0 TPU setup. Compare a new variant against the reference row for its
rung, trained on the same data. The TPU reference rows do not use the Hero data mix, so do not compare them
with the H100 rows in the [Fast Track README](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/README.md).

## Leaderboard

Paloma loss is the comparison metric, as in the Fast Track
[agent guide](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/agent.md). Tokens/s and MFU are means over the last 100 steps.
Wall time is the W&B run duration. It excludes slice provisioning and worker setup.

| Rung | Run | Tokens | Paloma loss | Paloma bpb | Uncheatable bpb | Tokens/s | MFU | Wall time |
|------|-----|-------:|------------:|-----------:|----------------:|---------:|----:|----------:|
| d768 | reference ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-dense-d768-dclmbpe-v5p32-20260928)) | 1.07B | 3.2548 | 1.3275 | 1.1478 | 4.25M | 33.9% | 6.0 min |
| d768 | reference, tuned optimizer‡ ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-tune20261001-t022-d768-dclmbpe-v5p32)) | 1.07B | 3.2431 | 1.3230 | 1.1444 | 4.28M | 34.1% | 5.9 min |
| d768 | looped, `mid2-grt-grow`§ ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-mid2-grt-grow-t26-d768-dclmbpe-v5p32-20261003r4)) | 1.07B | 3.2287 | 1.3169 | 1.1396 | 3.94M | 33.7% | 6.9 min |
| d768 | best, gated middle block, `mid4-grt-grow`§ ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-mid4-grt-grow-t26-d768-dclmbpe-v5p32-20261003r13)) | 1.07B | 3.2150 | 1.3111 | 1.1296 | 3.96M | 34.0% | 6.8 min |
| d1024 | reference ([W&B](https://wandb.ai/neel/marin/runs/fasttrack-dense-d1024-dclmbpe-v5p32-20260928)) | 2.89B | 2.9897 | 1.2219 | 1.0193 | 2.11M | 38.6% | 25.2 min |

‡The tuned-optimizer row is the best of a 28-trial [Vizier](https://github.com/google/vizier) sweep (GP-UCB-PE,
seeded with the reference) over the reference's optimizer, run with `experiments.asura.tune_fast_track`. It
searched multipliers on the heuristic's MuonH and Adam learning rates, Muon momentum, beta1, beta2, warmup, decay
fraction, and the minimum learning-rate ratio. The winner uses MuonH LR 0.0169 (0.81x the heuristic), Adam LR
0.00568 (1.18x), beta1 0.918, beta2 0.958 (the heuristic's is 0.996), momentum 0.950, 2.9% warmup, decay over the
final 77% of steps, and a minimum learning-rate ratio of 0.073.

The row shows seed 0, the run the sweep selected. Retrained with four more seeds, which change the initialization
and the data order, the tuned optimizer gives:

| Seed | 0 | 1 | 2 | 3 | 4 | Mean | Standard deviation |
|------|--:|--:|--:|--:|--:|-----:|-------------------:|
| Paloma loss | 3.2431 | 3.2492 | 3.2459 | 3.2541 | 3.2517 | 3.2488 | 0.0044 |

Seed 0 was the luckiest of the five, so its 0.0117 lead over the reference overstates the gain. The mean beats the
reference by 0.0060, which the reference recipe would need 1.04x the compute to match. That is about 1.3 standard
deviations of a single run, and the reference has only run with seed 0, so compare variants against the reference
row until the reference's own seeds confirm the gain. Seed runs are
[`fasttrack-tune20261001-t022-s{1..4}-d768-dclmbpe-v5p32`](https://wandb.ai/neel/marin/runs/fasttrack-tune20261001-t022-s1-d768-dclmbpe-v5p32).

§These rows come from
[`experiments/grug/looped/model.py`](https://github.com/marin-community/marin/blob/main/experiments/grug/looped/model.py)
and are launched with `--variant`. All of them use the optimizer of a 28-trial Vizier sweep of `mid2-grt`
(`tune_fast_track --variant mid2-grt`, the dense sweep's search space): MuonH LR 1.01x the heuristic, Adam LR 3.98x,
Muon momentum 0.98, beta1 0.80, beta2 0.993, 0.1% warmup, decay over the final 99% of steps, and a minimum
learning-rate ratio of 0.0007. FLOPs per token include every added projection and grown pass, so the MFU column is
comparable with the references'. Tokens/s and MFU on these rows are whole-run means, since the grown passes run only
in the last steps.

- `mid2-grt` keeps the first and last two layers unique and runs one shared two-block core twice, with the gated
  update of the [Gated Recurrent Transformer](https://arxiv.org/abs/2608.15062): each pass runs the core on a
  projection of the state and the prelude output, and a gate that starts near 0.98 blends the core's output into the
  state. It stores 6 of the reference's 8 blocks.
- `mid4-grt` keeps all 8 blocks distinct and wraps the middle four in one gated pass of the same update.
- `-pool` adds each pass's change to a pool the coda reads, in place of the gated update; see below.
- `-grow` adds a pass over the same core and gate for the last 10% of steps, and evaluates with it. The FLOPs below
  are training FLOPs; evaluation, like any later use of the model, runs every pass, at 1.27x the reference's forward
  FLOPs for `mid2-grt-grow` and 1.48x for `mid4-grt-grow`.

With one optimizer (the sweep's above) for every model on seed 0, against the reference recipe with that optimizer
(FLOPs relative to the reference):

| Model | Paloma loss | FLOPs |
|-------|------------:|------:|
| Dense | 3.2352 | 1.000x |
| Looped, `mid2-grt` | 3.2368 | 1.048x |
| Looped and grown, `mid2-grt-grow` | 3.2287 | 1.070x |
| Gated middle block, `mid4-grt` | 3.2241 | 1.030x |
| Gated middle block, grown, `mid4-grt-grow` | 3.2150 | 1.076x |
| Pooled middle block, grown, `mid4-pool-grow` | **3.2127** | 1.070x |

- **The gated update is the main gain, not the sharing.** Around a distinct middle block it is worth 0.011. Sharing
  the core across two passes gives up 0.013 of it.
- **Growth adds 0.007 to 0.014** on every model and seed tried. One extra pass for the last 10% is enough: a third
  for the last 20%, or a fourth, gains nothing more.
- **The gate needs unique layers around it.** With a key-value head per query head, gating the middle six blocks or
  all eight scored 0.012 and 0.032 worse than gating the middle four.
- **The tuned dense optimizer is not the gap.** The sweep's optimizer above improves the reference by 0.008.
- A key-value head per query head in every layer improves the dense and looped models alike by 0.023 to 0.038, at
  about 1.08x the FLOPs, and the gated middle block keeps its lead with it (3.1790 against 3.2126). It changes the
  reference recipe, so its runs (`dense-mha` and the `-allmha` variants) stay out of this ledger.

Over seeds, which change the initialization and the data order (the dense row uses the tuned dense optimizer):

| Seed | 0 | 1 | 2 | Mean |
|------|--:|--:|--:|-----:|
| Dense | 3.2431 | 3.2492 | 3.2459 | 3.2461 |
| Looped and grown (`mid2-grt-grow`) | 3.2287 | 3.2364 | 3.2327 | 3.2326 |
| Gated middle block (`mid4-grt`) | 3.2241 | 3.2229 | | 3.2235 |
| Gated middle block, grown (`mid4-grt-grow`) | 3.2150 | 3.2178 | | 3.2164 |

On both seeds it ran, the gated middle block beats the tuned dense run of the same seed. Grown, it gains 0.028 and
0.031, which dense would need 1.18x and 1.20x the compute to match, for 1.076x the FLOPs. Without growth it gains
0.019 and 0.026, worth 1.12x and 1.17x, for 1.030x the FLOPs and at 95% of the reference's throughput.

The string ensembles of [neel04/loop](https://github.com/neel04/loop) split the core into narrow private copies,
strings, each with its own key-value head, that all refine the shared state each pass. At d768 they add nothing that
the extra key-value heads do not: two gated strings, grown, score 3.2230, and one gated core with two key-value heads,
grown, 3.2233 at the same FLOPs. Reading six ungated strings out as a mixture scores 3.2358 at half the reference's
throughput. A port of the loop repo's refiner, narrowed to keep its compute under 10% of the reference's, changes no
gated rung by more than 0.004; its full-state restarts through the narrow latent cost 0.028, and re-anchoring the
strings on the refined state gains nothing. Two strings in the gated middle block, grown, score 3.2186 against
3.2150 for one core, at 65% of its throughput.

Training the strings to explore different hypotheses, after [Explorative Modeling](https://arxiv.org/abs/2607.27372),
hurt or did nothing. Every string's output after every pass was a hypothesis, decoded at every 32nd position by the
output head after a learned norm, or after a tuned lens in the later runs. Only the hypothesis that predicted a token
best got credit.
- **Crediting the winning strings made them hedge.** At weight 0.25 the best of the hypotheses reached 2.4 nats,
  while the strings' average, which the model reads, lost 0.17 (3.3897). At weight 0.03 it still lost 0.015.
- **Training only the refiner on that credit changed nothing.** Its proposals were detached from the next-token
  loss. It tied the jointly trained refiner: 3.2258 against 3.2225 and, in the gated middle block, 3.2195 against
  3.2186.
- **The model turned the proposals off.** The next-token loss scaled them to a gain of 0.01 to 0.02, against 0.15
  for the jointly trained refiner.
- **The exploration loss cost up to 19% of throughput.**
- **Choosing among the strings does not rescue the credit.** In place of the strings' average, a learned selector
  weighed each pass's strings from a query on the state and a key on each string. With the exploration loss it still
  lost 0.16 to 0.17 (3.3788 and 3.3912), whether or not it was also trained to pick each pass's winner. It picked
  the winner 62% of the time against 50% by chance, because the winner is the string that bet on the token that came
  next, which the context does not reveal. On its own, without the exploration loss, the selector stayed near an
  even average (3.2298).
- **Mixing the strings' distributions does not rescue it either.** The loop repo's readout mixes the strings'
  next-token distributions with a contextual prior, and here each string also had its own refiner. Credited by the
  winner, each string's own loss reached 7.2 to 9.2 nats, while the prior stayed near even: 3.3446 with two strings
  and 3.4199 with three. With the loop repo's proportional credit, the uniform mixture's likelihood, the prior also
  stayed even and the mixture about matched the fused readout's 3.2225: 3.2297 with one shared refiner, 3.2194 with
  each of two strings proposed by its own refiner (`mid2-grt-str2-uref-mix-grow`, 1.14x the reference's FLOPs), and
  3.2346 with three, at 59% to 70% of the fused readout's throughput.

What else was tried on the readout and the strings, each from one seed:
- **The track-record mixture** (two strings, each proposed by its own refiner, mixed by a prior that also weighs each
  string's record on the document's earlier tokens, trained on the deployed mixture's likelihood): 3.2105 at 1.15x
  the FLOPs, against 3.2149 for the same mixture without the record. Its prior picks the string that predicts the
  next token best 64% of the time, against 54% without the record, but one string takes 72% of the weight.
- **Children of those strings**, born at 90% of training from their parents' refiners and run in parallel with them:
  3.2197 at 1.25x the FLOPs, 0.009 worse than the mixture without children.
- **Candidate tokens from a proposal head**, each verified by one string: 3.2179.
- **Each string's surprise, fed back between rounds**: a lens read each string's next-token loss on the document's
  earlier tokens before the last round, and the last round's refiners and commit read it. Measured at every position,
  every fourth, or the document's rare tokens, it scored 3.2072, 3.2077 and 3.2087 at 1.32x and 1.21x the FLOPs,
  within noise of the mixture without it. The lens's own loss was 4.4 nats against the model's 3.2.
- **An in-context cache in the readout**, after the [neural cache](https://arxiv.org/abs/1612.04426), scored 3.1705,
  0.044 better than the gated middle block, but nothing in it is specific to looping, so it is out of scope here.

Training the rounds one at a time, so that depth would be a choice at inference, does not pay at this rung; its code
was removed. Each step draws a depth of one to four rounds of the gated middle block, runs all but the
last forward only, and trains the last, which starts from the prelude's output plus the earlier rounds' detached
refinement. At four rounds it scores 3.2237 (1.26x the reference's training FLOPs), the same as one gated pass trained
end to end (3.2241 at 1.03x) and 0.009 behind the grown one. Depth helps up to three rounds (3.2672 at one, 3.2240 at
three), then slowly hurts (3.2290 at eight): the core learns to refine for the depths it trained at, not toward a fixed
point. Training the last two rounds together scores 3.2396. Two narrow strings, each with its own coda, mixed by their
records on the document's earlier tokens, lose 0.024 to 0.058: pooled by those records 3.2478, with a refiner that
proposes once 3.2542, with a refiner per round 3.2820; the records never tell the strings apart (0.50 and 0.50).

Adding each pass's work to a pool, instead of carrying a gated state (`PooledTransformer`), ties the frontier with one
pass and loses with two shared ones. The pool starts as the prelude's output; each pass
runs the core on the prelude's output plus a projection of the pool, and adds the core's change of its input, where the
gate opens, to the pool, which the coda reads. Every pass's contribution then reaches the coda through additions alone:
a residual connection across passes in place of the gated update's highway. The gated middle block grown this way,
`mid4-pool-grow`, scores 3.2127, within noise of `mid4-grt-grow` (3.2150) at the same throughput. With two passes over
a shared core, `mid2-pool` loses 0.0085 to `mid2-grt` (3.2411 against 3.2326), as
[Hyperloop](https://arxiv.org/abs/2604.21254)'s ablation of an identity carry against a gated one predicts. Pooling
after the coda, which then runs every pass, scores 3.2383 at 1.29x the FLOPs; two strings on the shared core score
3.2345 sharing one pool and 3.2311 with a pool each, at 1.52x and 1.54x. Only the pool before the coda, with one string,
stays in the code.

Every readout that reads other positions sees only its own document, as the model's attention does: training and
evaluation sequences pack several documents. A first version of the cache read earlier documents too and scored
3.1590; earlier track-record runs read them as well, and none of those runs is reported.

What the earlier looped runs established, each from one seed (single-run differences under about 0.01 are noise):
- Distinct blocks matter most among shared cores. Two plain passes over all 4 shared blocks (`loop2`, 3.3242) trail
  plain 2 + 2×2 + 2 (`mid2`, 3.2852) by 0.039, and a 1 + 3×2 + 1 split with the gate trails `mid2-grt` by 0.014.
- A gate around the loop is worth about 0.02 over a plain loop with the heuristic optimizer; the hyper-connections of
  [Hyperloop](https://arxiv.org/abs/2604.21254) give about the same.
- Within noise or worse: adding the loop input back before each pass or before the coda, sharing only the attention
  and MLP matrices, [SMELT](https://arxiv.org/abs/2609.01343)'s 1/r scaling,
  [ASURA](https://neel04.github.io/my-website/projects/asura/)'s per-iteration projections and multi-pass loss, the
  loop repo's strands, refiner and per-string gates, late prior credit, strings that loop different numbers of times,
  growth on ungated strings, iso-FLOP or progressive growth schedules, and pass embeddings for the grown passes.
- Sharing the blocks only for the first 30% of training and then untying them beat the reference (3.2415), but the
  result is an unlooped model, so it is out of scope here.

Run ids follow `fasttrack-<variant>[-t22|-t26][-s<seed>]-d768-dclmbpe-v5p32-<date>` in W&B project `neel/marin`;
removed variants' settings are in their run configs.

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

1. Launch the variant with `experiments.asura.tpu_queue` and the ladder config, which fixes the
   reference cache, tokenizer, evaluation sets, batch, and step count for the rung. Change only the variant.
2. Add a row with the W&B link and the same columns. Mark it in the Run column; keep the reference rows.
3. For an effective-speedup estimate, use `effective_speedup` from the
   [agent guide](https://github.com/marin-community/marin/blob/main/experiments/grug/fast_track/agent.md). Use `L_inf = 1.2` and `alpha = 0.0844` from
   the H100 dense fit until a four-rung TPU fit exists.
