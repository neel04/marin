# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.looped.blocks import Scalars, fast_mlp
from experiments.grug.looped.model import (
    GatedConfig,
    GatedTransformer,
    LoopedModelConfig,
    LoopedTransformer,
    PoolConfig,
    PooledTransformer,
    ScheduledTransformer,
    looped,
)

SEQ_LEN = 16
# Six executed layers: one prelude layer, two passes over a two-block core, one coda layer. With
# global_every=4, layer 3 (the first block of the second pass) and the last layer are global.
BASE = GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=96,
    num_layers=6,
    num_heads=2,
    num_kv_heads=1,
    local_kv_heads=1,
    global_kv_heads=1,
    head_dim=16,
    max_seq_len=SEQ_LEN,
    sliding_window=4,
    global_every=4,
    dense_mlp=True,
)
MIDDLE_CYCLE = dict(prelude=1, coda=1, loops=2)


@pytest.fixture(autouse=True)
def _mesh():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        yield


def _tokens() -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(1), (len(jax.devices()), SEQ_LEN), 0, BASE.vocab_size)


def _forward(model: LoopedTransformer, tokens: jax.Array) -> jax.Array:
    return eqx.filter_jit(lambda m, ids: m(ids)[0])(model, tokens)


def _stored_block(model: LoopedTransformer, i: int):
    """The stored block that executed layer ``i`` runs: prelude blocks, then one core reused by every pass, then coda."""
    cfg: LoopedModelConfig = model.config
    coda_start = cfg.num_layers - cfg.coda

    if i < cfg.prelude:
        stored = i
    elif i >= coda_start:
        stored = cfg.prelude + cfg.core_blocks + (i - coda_start)
    else:
        stored = cfg.prelude + (i - cfg.prelude) % cfg.core_blocks

    return jax.tree.map(lambda leaf: leaf[stored], model.stacked_blocks.stacked)


def _run_layer(model: LoopedTransformer, hidden: jax.Array, i: int, block) -> jax.Array:
    cfg = model.config
    is_global = (i + 1) % cfg.global_every == 0 or i == cfg.num_layers - 1
    return block(hidden, AttentionMask.causal(), disable_rope=is_global, is_global=is_global)[0]


def _layer_by_layer(model: LoopedTransformer, tokens: jax.Array) -> jax.Array:
    hidden = model.embed_gated_norm(model.embed_norm(model.token_embed[tokens]))

    for i in range(model.config.num_layers):
        hidden = _run_layer(model, hidden, i, _stored_block(model, i))

    return model.final_gated_norm(model.final_norm(hidden))


def _prelude_and_coda(model: LoopedTransformer, tokens: jax.Array) -> jax.Array:
    """The model with its looped passes removed."""
    cfg = model.config
    hidden = model.embed_gated_norm(model.embed_norm(model.token_embed[tokens]))

    for i in [*range(cfg.prelude), *range(cfg.num_layers - cfg.coda, cfg.num_layers)]:
        hidden = _run_layer(model, hidden, i, _stored_block(model, i))

    return model.final_gated_norm(model.final_norm(hidden))


def test_looped_forward_runs_one_stored_core_every_pass():
    model = LoopedTransformer.init(looped(BASE, **MIDDLE_CYCLE), key=jax.random.PRNGKey(0))

    # One prelude block, one two-block core and one coda block.
    stored_blocks = jax.tree.leaves(model.stacked_blocks.stacked)[0].shape[0]
    assert stored_blocks == model.stacked_blocks.num_layers == 4

    tokens = _tokens()
    reference = eqx.filter_jit(_layer_by_layer)(model, tokens)
    np.testing.assert_allclose(_forward(model, tokens), reference, rtol=1e-5, atol=1e-5)


def _gated(**fields) -> GatedTransformer:
    cfg = looped(BASE, config_type=GatedConfig, **MIDDLE_CYCLE | fields)
    return GatedTransformer.init(cfg, key=jax.random.PRNGKey(0))


def _pooled(**fields) -> PooledTransformer:
    cfg = looped(BASE, config_type=PoolConfig, **MIDDLE_CYCLE | fields)
    return PooledTransformer.init(cfg, key=jax.random.PRNGKey(0))


def _fast(**fields) -> PooledTransformer:
    """A pooled rung whose fast weights act from the start: their gains are nonzero."""
    model = _pooled(fast_rank=4, **fields)
    assert model.fast is not None
    gain = Scalars(jnp.ones_like(model.fast.gain.weight))
    return dataclasses.replace(model, fast=dataclasses.replace(model.fast, gain=gain))


def _outer_block(model: ScheduledTransformer, position: int):
    """The stored prelude or coda block at ``position``: the prelude blocks come first, then the coda's."""
    return jax.tree.map(lambda leaf: leaf[position], model.stacked_blocks.stacked)


def _prefix(model: ScheduledTransformer, tokens: jax.Array) -> jax.Array:
    """The prelude output, one layer at a time."""
    hidden = model.embed_gated_norm(model.embed_norm(model.token_embed[tokens]))
    for i in range(model.config.prelude):
        hidden = _run_layer(model, hidden, i, _outer_block(model, i))
    return hidden


def _core_output(model: ScheduledTransformer, hidden: jax.Array, p: int) -> jax.Array:
    """The core run on ``hidden`` as pass ``p``, one layer at a time."""
    layers = model.config.pass_layout(p)
    for j, i in enumerate(range(layers.start, layers.stop)):
        hidden = _run_layer(model, hidden, i, jax.tree.map(lambda leaf, j=j: leaf[j], model.core))
    return hidden


def _decoded(model: ScheduledTransformer, hidden: jax.Array) -> jax.Array:
    """The coda and the final norms run on ``hidden``, one layer at a time."""
    cfg = model.config
    for k, i in enumerate(range(cfg.num_layers - cfg.coda, cfg.num_layers)):
        hidden = _run_layer(model, hidden, i, _outer_block(model, cfg.prelude + k))
    return model.final_gated_norm(model.final_norm(hidden))


def _replaced_each_pass(model: GatedTransformer, tokens: jax.Array) -> jax.Array:
    """A rung that replaces its state with the core's output each pass."""
    hidden = _prefix(model, tokens)
    for p in range(model.config.loops):
        hidden = _core_output(model, hidden, p)
    return _decoded(model, hidden)


def _without_passes(model: ScheduledTransformer, tokens: jax.Array) -> jax.Array:
    """The rung with its looped passes removed."""
    return _decoded(model, _prefix(model, tokens))


def _with_constant_gate(model: ScheduledTransformer, keep_logit: float) -> ScheduledTransformer:
    """``model`` with a gate that keeps ``sigmoid(keep_logit)`` of every state coordinate, writing the rest."""
    gate = model.gate
    assert gate is not None

    # A zero output weight makes the gate its bias alone.
    gate = dataclasses.replace(
        gate, gate_out=jnp.zeros_like(gate.gate_out), gate_out_bias=jnp.full_like(gate.gate_out_bias, keep_logit)
    )
    return dataclasses.replace(model, gate=gate)


def test_gated_update_that_keeps_the_state_skips_the_loop():
    model = _with_constant_gate(_gated(), keep_logit=30.0)
    tokens = _tokens()

    reference = eqx.filter_jit(_without_passes)(model, tokens)
    np.testing.assert_allclose(_forward(model, tokens), reference, rtol=1e-5, atol=1e-5)


def test_gated_update_that_replaces_the_state_runs_the_core():
    model = _with_constant_gate(_gated(), keep_logit=-30.0)

    # Project [state ; prelude] onto the state alone, so each pass runs the core on the state.
    assert model.core_input is not None
    d = BASE.hidden_dim
    project = jnp.concatenate([jnp.eye(d), jnp.zeros((d, d))])
    model = dataclasses.replace(model, core_input=dataclasses.replace(model.core_input, project=project))

    tokens = _tokens()
    reference = eqx.filter_jit(_replaced_each_pass)(model, tokens)
    np.testing.assert_allclose(_forward(model, tokens), reference, rtol=1e-5, atol=1e-5)


def test_pool_that_takes_no_writes_skips_the_loop():
    model = _with_constant_gate(_pooled(), keep_logit=30.0)
    tokens = _tokens()

    reference = eqx.filter_jit(_without_passes)(model, tokens)
    np.testing.assert_allclose(_forward(model, tokens), reference, rtol=1e-5, atol=1e-5)


def test_pooled_passes_add_their_changes_of_the_prelude_output():
    model = _with_constant_gate(_pooled(), keep_logit=-30.0)
    # Blind to the pool, every pass runs the core on the prelude output alone, so the pool sums the passes' changes.
    assert model.read is not None
    model = dataclasses.replace(model, read=dataclasses.replace(model.read, project=jnp.zeros_like(model.read.project)))
    tokens = _tokens()

    def expected(model: PooledTransformer, tokens: jax.Array) -> jax.Array:
        prefix = _prefix(model, tokens)
        changes = [_core_output(model, prefix, p) - prefix for p in range(model.config.loops)]
        return _decoded(model, prefix + sum(changes))

    np.testing.assert_allclose(_forward(model, tokens), eqx.filter_jit(expected)(model, tokens), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("build", [_gated, _pooled, _fast])
def test_scheduled_rung_reads_no_later_token(build):
    model = build(pass_starts=(0.0, 0.0, 0.5))
    tokens = _tokens()
    middle = SEQ_LEN // 2
    changed = tokens.at[:, middle].set((tokens[:, middle] + 1) % BASE.vocab_size)

    before, after = _forward(model, tokens), _forward(model, changed)
    np.testing.assert_allclose(before[:, :middle], after[:, :middle], rtol=1e-5, atol=1e-5)
    assert not np.allclose(before[:, middle:], after[:, middle:])


@pytest.mark.parametrize("build", [_gated, _pooled, _fast])
def test_growing_rung_trains_as_the_ungrown_rung_before_it_grows(build):
    grown = build(pass_starts=(0.0, 0.0, 0.9))
    plain = dataclasses.replace(grown, config=dataclasses.replace(grown.config, pass_starts=()))
    tokens = _tokens()
    weight = jnp.ones(tokens.shape)

    def loss(model: ScheduledTransformer, progress: float) -> jax.Array:
        return model.next_token_loss(tokens, weight, progress=jnp.asarray(progress))

    early = eqx.filter_jit(loss)(grown, 0.5)
    np.testing.assert_allclose(early, eqx.filter_jit(loss)(plain, 0.5), rtol=1e-6)

    # In the last tenth of training, the extra pass changes the loss, if only slightly: the gate starts writing 2%.
    assert eqx.filter_jit(loss)(grown, 0.95) != early


def _blind_full_writer() -> PooledTransformer:
    """A pooled rung blind to its pool that writes its whole change every pass, so the pool sums the core's changes of
    the prelude output."""
    model = _with_constant_gate(_pooled(), keep_logit=-30.0)
    assert model.read is not None
    return dataclasses.replace(model, read=dataclasses.replace(model.read, project=jnp.zeros_like(model.read.project)))


def test_rung_runs_further_passes_at_inference():
    model = _blind_full_writer()
    deeper = dataclasses.replace(model, config=model.config.with_inference_depth(3))
    tokens = _tokens()

    # The third pass reuses the core and the second pass's attention spans.
    def expected(model: PooledTransformer, tokens: jax.Array) -> jax.Array:
        prefix = _prefix(model, tokens)
        changes = [_core_output(model, prefix, p) - prefix for p in range(3)]
        return _decoded(model, prefix + sum(changes))

    np.testing.assert_allclose(_forward(deeper, tokens), eqx.filter_jit(expected)(model, tokens), rtol=1e-5, atol=1e-5)


def test_fast_weights_start_as_the_plain_pooled_rung():
    tokens = _tokens()
    fast = _with_constant_gate(_pooled(fast_rank=4), keep_logit=-30.0)
    plain = _with_constant_gate(_pooled(), keep_logit=-30.0)

    np.testing.assert_allclose(_forward(fast, tokens), _forward(plain, tokens), rtol=1e-6, atol=1e-6)


def test_fast_weights_change_each_tokens_mlp_weights():
    model = _fast()
    assert model.core is not None and model.fast is not None
    mlp = jax.tree.map(lambda leaf: leaf[0], model.core.mlp)
    fast = jax.tree.map(lambda leaf: leaf[0], model.fast)
    x, features = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 3, BASE.hidden_dim))

    out = fast_mlp(mlp, x, fast, features)

    # A rank-4 change of the gate and up projections, its own for each token.
    for t in range(3):
        scale = jnp.diag((features[0, t] @ fast.to_scale) * fast.gain.weight)
        changed = dataclasses.replace(
            mlp,
            w_gate=mlp.w_gate + fast.gate_in @ scale @ fast.gate_out,
            w_up=mlp.w_up + fast.up_in @ scale @ fast.up_out,
        )
        expected = changed(x[:, t : t + 1], moe_output_reshard=False)
        np.testing.assert_allclose(out[:, t : t + 1], expected, rtol=1e-5, atol=1e-6)


def test_fast_rung_logs_its_scales_and_their_change_each_pass():
    model = _fast()
    tokens = _tokens()

    _, metrics = eqx.filter_jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), return_router_metrics=True))(
        model
    )

    # Two passes, each with its scales' RMS and their change relative to it; the first changes from nothing.
    assert {k for k in metrics if k.startswith("train/fast")} == {
        f"train/fast/{name}_pass{p}" for name in ("scale_rms", "scale_change") for p in range(2)
    }
    assert float(metrics["train/fast/scale_rms_pass1"]) > 0
    np.testing.assert_allclose(metrics["train/fast/scale_change_pass0"], 1.0, rtol=1e-5)
    assert 0 < float(metrics["train/fast/scale_change_pass1"]) < 2
    # A pass grown later logs zeros until it starts.
    _, early = eqx.filter_jit(
        lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), return_router_metrics=True, progress=jnp.array(0.5))
    )(_fast(pass_starts=(0.0, 0.9)))
    assert float(early["train/fast/scale_rms_pass0"]) > 0
    assert float(early["train/fast/scale_rms_pass1"]) == float(early["train/fast/scale_change_pass1"]) == 0
    # Without fast weights, nothing is logged.
    _, plain_metrics = eqx.filter_jit(
        lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), return_router_metrics=True)
    )(_pooled())
    assert not any(k.startswith("train/fast") for k in plain_metrics)
