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
from experiments.grug.looped.model import (
    GatedRecurrentConfig,
    GatedRecurrentTransformer,
    LoopedModelConfig,
    LoopedTransformer,
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


def _gated_recurrent(keep_logit: float) -> GatedRecurrentTransformer:
    """A Gated Recurrent Transformer whose gate keeps ``sigmoid(keep_logit)`` of every state coordinate."""
    model = GatedRecurrentTransformer.init(
        looped(BASE, config_type=GatedRecurrentConfig, **MIDDLE_CYCLE), key=jax.random.PRNGKey(0)
    )
    cell = model.cell
    assert cell is not None

    # A zero output weight makes the gate its bias alone.
    cell = dataclasses.replace(
        cell, gate_out=jnp.zeros_like(cell.gate_out), gate_out_bias=jnp.full_like(cell.gate_out_bias, keep_logit)
    )
    return dataclasses.replace(model, cell=cell)


def test_gated_recurrent_cell_that_keeps_the_state_skips_the_loop():
    model = _gated_recurrent(keep_logit=30.0)
    tokens = _tokens()

    reference = eqx.filter_jit(_prelude_and_coda)(model, tokens)
    np.testing.assert_allclose(_forward(model, tokens), reference, rtol=1e-5, atol=1e-5)


def test_gated_recurrent_cell_that_replaces_the_state_with_the_core_output_is_the_looped_model():
    model = _gated_recurrent(keep_logit=-30.0)

    # Project [state ; prelude] onto the state alone, so each pass runs the core on the state.
    d = BASE.hidden_dim
    project = jnp.concatenate([jnp.eye(d), jnp.zeros((d, d))])
    model = dataclasses.replace(model, cell=dataclasses.replace(model.cell, project=project))

    fields = {field.name: getattr(model, field.name) for field in dataclasses.fields(LoopedTransformer)}
    plain = LoopedTransformer(**fields | {"config": looped(BASE, **MIDDLE_CYCLE)})

    tokens = _tokens()
    np.testing.assert_allclose(_forward(model, tokens), _forward(plain, tokens), rtol=1e-5, atol=1e-5)
