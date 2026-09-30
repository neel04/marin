# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.model import CausalSelfAttention, GrugModelConfig

SEQ_LEN = 16
WINDOW = 4
SCONV_KERNEL = 4
# Perturbing position 0 reaches keys 0..SCONV_KERNEL-1 through the K short conv; a windowed query at t
# only sees keys t-WINDOW+1..t, so every query from here on is outside the perturbation's reach.
FIRST_UNREACHABLE = (SCONV_KERNEL - 1) + WINDOW


def _max_output_change(is_global: bool) -> jax.Array:
    cfg = GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=SEQ_LEN,
        sliding_window=WINDOW,
        sconv_kernel=SCONV_KERNEL,
        dense_mlp=True,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        attn = CausalSelfAttention.init(cfg, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (len(jax.devices()), SEQ_LEN, cfg.hidden_dim))
        perturbed = x.at[:, 0].add(1.0)
        # A traced flag, as in the layer scan, exercises the per-layer lax.cond path.
        flag = jnp.asarray(is_global)
        call = eqx.filter_jit(lambda module, inputs: module(inputs, AttentionMask.causal(), is_global=flag))
        change = jnp.abs(call(attn, perturbed) - call(attn, x))
    return jnp.max(change[:, FIRST_UNREACHABLE:])


@pytest.mark.parametrize("is_global", [False, True])
def test_local_layers_apply_sliding_window_off_gpu(is_global: bool):
    change = _max_output_change(is_global)
    if is_global:
        assert change > 1e-3
    else:
        assert change == 0.0
