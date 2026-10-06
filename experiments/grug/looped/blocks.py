# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What looped Fast Track rungs are built from: the runner that scans stacked blocks as executed layers, the gate and
core input of each pass, the pool's read, the fast weights the pool programs, and the FLOP and layer helpers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, PRNGKeyArray
from levanter.grug.attention import AttentionMask
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.fast_track.model import (
    _REMAT_POLICY,
    Block,
    DenseMLP,
    GrugModelConfig,
    RMSNorm,
    _batch_reshard,
    _batch_spec,
    _init_weight,
)

Hidden = Float[Array, "B S D"]

# The Gated Recurrent Transformer's initial gate logit: sigmoid(4) = 0.98 of the state is kept at first.
_INITIAL_KEEP_LOGIT = 4.0


def _own_forward(block: Block, hidden: Hidden, mask: AttentionMask | Array, is_global: Array, extra: None) -> Hidden:
    hidden, _ = block(hidden, mask, disable_rope=is_global, is_global=is_global)
    return hidden


@dataclass(frozen=True)
class LayerRunner:
    """Scans stacked blocks as a contiguous range of executed layers, which set each block's attention span."""

    mask_for_layer: Callable[[Array], AttentionMask]
    is_global: Array

    def __call__(
        self,
        hidden: Hidden,
        blocks: Block,
        layers: slice,
        step: Callable[..., Hidden] = _own_forward,
        extras: Any = None,
    ) -> Hidden:
        """Runs ``step(block, hidden, mask, is_global, extra)`` for each block, by default the block's own forward, with
        ``extras`` the blocks' ``extra`` inputs, stacked."""
        step = eqx.filter_checkpoint(step, policy=_REMAT_POLICY)

        def run_layer(hidden: Hidden, layer: tuple[Block, Array, Any]) -> tuple[Hidden, None]:
            block, layer_is_global, extra = layer
            return step(block, hidden, self.mask_for_layer(layer_is_global), layer_is_global, extra), None

        hidden, _ = jax.lax.scan(run_layer, hidden, (blocks, self.is_global[layers], extras))
        return hidden


class PassGate(eqx.Module):
    """The Gated Recurrent Transformer's update (arXiv 2608.15062): ``state <- g * state + (1 - g) * output``.

    The gate is a two-layer MLP of the normalized state and prelude output. Its output bias starts at
    ``_INITIAL_KEEP_LOGIT``, so each pass starts out keeping nearly all of the state. The paper also adds noise to the
    state and the gate during training; the trainer gives the forward pass no randomness, so this gate does not.
    """

    state_norm: RMSNorm
    prelude_norm: RMSNorm
    gate_in: Float[Array, "2D D"]
    gate_in_bias: Float[Array, " D"]
    gate_out: Float[Array, "D D"]
    gate_out_bias: Float[Array, " D"]

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "PassGate":
        d = cfg.hidden_dim
        in_key, out_key = random.split(key)
        return PassGate(
            state_norm=RMSNorm.init(d, cfg.layer_norm_eps),
            prelude_norm=RMSNorm.init(d, cfg.layer_norm_eps),
            gate_in=_init_weight(in_key, (2 * d, d), cfg.initializer_std),
            gate_in_bias=jnp.zeros(d),
            gate_out=_init_weight(out_key, (d, d), cfg.initializer_std),
            gate_out_bias=jnp.full(d, _INITIAL_KEEP_LOGIT),
        )

    def prelude_term(self, prelude: Hidden) -> Hidden:
        """The prelude output's share of the gate's first layer, which every pass reuses."""
        d = prelude.shape[-1]
        return self.prelude_norm(prelude) @ self.gate_in[d:] + self.gate_in_bias

    def keep(self, state: Hidden, prelude_term: Hidden) -> Hidden:
        """The fraction of each state coordinate the pass keeps."""
        hidden = jax.nn.silu(self.state_norm(state) @ self.gate_in[: state.shape[-1]] + prelude_term)
        return jax.nn.sigmoid(hidden @ self.gate_out + self.gate_out_bias)


class CoreInput(eqx.Module):
    """The Gated Recurrent Transformer's core input, ``W [state ; prelude]``."""

    project: Float[Array, "2D D"]

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CoreInput":
        d = cfg.hidden_dim
        return CoreInput(project=_init_weight(key, (2 * d, d), cfg.initializer_std))

    def prelude_term(self, prelude: Hidden) -> Hidden:
        """The prelude output's share, which every pass reuses: ``W [state ; prelude] = W_s state + W_p prelude``."""
        return prelude @ self.project[prelude.shape[-1] :]

    def __call__(self, state: Hidden, prelude_term: Hidden) -> Hidden:
        return state @ self.project[: state.shape[-1]] + prelude_term


class PoolRead(eqx.Module):
    """What a pooled pass reads of the pool: a projection of the normalized pool, which it adds to the prelude output."""

    norm: RMSNorm
    project: Float[Array, "D D"]

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "PoolRead":
        d = cfg.hidden_dim
        return PoolRead(norm=RMSNorm.init(d, cfg.layer_norm_eps), project=_init_weight(key, (d, d), cfg.initializer_std))

    def __call__(self, pool: Hidden) -> Hidden:
        return self.norm(pool) @ self.project


class Scalars(eqx.Module):
    """Learned scalars, one per channel. The name ``weight`` routes them to Adam, as norm gains are, where MuonH's
    norm-preserving step would hold a zero start at zero."""

    weight: Array


class FastWeights(eqx.Module):
    """The pool's per-token change of each core block's MLP gate and up projections, ``W + U diag(s_t) V`` with rank-r
    scales ``s_t`` read from the pool. A weight that a token writes and only that token reads, as fast-weight
    programmers' are, reduces to such a modulation of the token's own path; the pool writes it every pass. Its gains
    start at zero, so the rung starts as the plain pooled one."""

    to_scale: Float[Array, "L D R"]
    gate_in: Float[Array, "L D R"]
    gate_out: Float[Array, "L R M"]
    up_in: Float[Array, "L D R"]
    up_out: Float[Array, "L R M"]
    gain: Scalars

    @staticmethod
    def init(cfg: GrugModelConfig, *, blocks: int, rank: int, key: PRNGKeyArray) -> "FastWeights":
        """Fast weights of ``rank`` for each of ``blocks`` stacked blocks."""
        scale_key, gate_in_key, gate_out_key, up_in_key, up_out_key = random.split(key, 5)
        d, m = cfg.hidden_dim, cfg.intermediate_dim
        # Unit-variance scales and low-rank inputs, and outputs as the MLP's own projections: MuonH keeps each matrix's
        # norm, so the gains alone set the change's scale.
        out_spec = P(None, None, "model")
        return FastWeights(
            to_scale=_init_weight(scale_key, (blocks, d, rank), d**-0.5),
            gate_in=_init_weight(gate_in_key, (blocks, d, rank), d**-0.5),
            gate_out=reshard(_init_weight(gate_out_key, (blocks, rank, m), cfg.initializer_std), out_spec),
            up_in=_init_weight(up_in_key, (blocks, d, rank), d**-0.5),
            up_out=reshard(_init_weight(up_out_key, (blocks, rank, m), cfg.initializer_std), out_spec),
            gain=Scalars(jnp.zeros((blocks, rank))),
        )

    def scales(self, features: Hidden) -> Float[Array, "B S L R"]:
        """Every block's scales of its changes, in float32, as ``fast_mlp`` computes one block's."""
        blocks = self.to_scale.shape[0]
        to_scale = rearrange(self.to_scale.astype(jnp.float32), "l d r -> d (l r)")
        scales = rearrange(features.astype(jnp.float32) @ to_scale, "b s (l r) -> b s l r", l=blocks)
        return scales * self.gain.weight.astype(jnp.float32)


def fast_layer(
    block: Block, x: Hidden, mask: AttentionMask | Array, is_global: Array, fast: FastWeights, *, features: Hidden
) -> Hidden:
    """A core block's forward as ``Block.__call__`` runs a dense block, with its MLP's input projections changed per
    token by ``fast``, from ``features``, the normalized pool."""
    segment_ids = mask.segment_ids[0] if isinstance(mask, AttentionMask) and mask.segment_ids is not None else None

    attn_out = block.attn(block.attn_gated_norm(block.rms_attn(x)), mask, disable_rope=is_global, is_global=is_global)
    if block.sconv_attn is not None:
        attn_out = block.sconv_attn(attn_out, segment_ids)
    x = x + attn_out

    assert isinstance(block.mlp, DenseMLP), "The looped model has dense blocks only"
    mlp_out = fast_mlp(block.mlp, block.mlp_gated_norm(block.rms_mlp(x)), fast, features)
    if block.sconv_mlp is not None:
        mlp_out = block.sconv_mlp(mlp_out, segment_ids)
    return x + mlp_out


def fast_mlp(mlp: DenseMLP, x: Hidden, fast: FastWeights, features: Hidden) -> Hidden:
    """``DenseMLP``'s forward with each token's gate and up projections changed by its scales from the pool."""
    batch_size, seq_len, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    scale = rearrange(features @ fast.to_scale, "b s r -> (b s) r") * fast.gain.weight

    def changed(weight: Array, into: Array, out_of: Array) -> Array:
        return x_flat @ weight + ((x_flat @ into) * scale) @ out_of

    gate = changed(mlp.w_gate, fast.gate_in, fast.gate_out)
    up = changed(mlp.w_up, fast.up_in, fast.up_out)
    out_flat = jnp.matmul(jax.nn.silu(gate) * up, mlp.w_down, out_sharding=_batch_spec())
    return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=batch_size, s=seq_len))


def scale_metrics(scales: Array, previous: Array) -> tuple[Array, Array]:
    """The RMS of a pass's fast-weight scales, and the RMS of their change since the previous pass's relative to it: a
    change of a few percent means the weights are the same in every pass."""
    rms = jnp.sqrt(jnp.mean(jnp.square(scales)))
    change = jnp.sqrt(jnp.mean(jnp.square(scales - previous)))
    return rms, change / (rms + 1e-8)


def layer_flops(cfg: GrugModelConfig, index: int) -> float:
    """Forward FLOPs per token of executed layer ``index``, priced as ``lm_flops_per_token`` prices layers."""
    is_global = (index + 1) % cfg.global_every == 0 or index == cfg.num_layers - 1
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=1,
        num_kv_heads=cfg.global_kv_heads if is_global else cfg.local_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=cfg.max_seq_len,
        vocab_size=0,
        glu=True,
        # A single layer under a sliding window and a global stride above one is a local layer.
        sliding_window=None if is_global else cfg.sliding_window,
        global_every=None if is_global else 2,
    )


def stacked_layers(blocks: Block, layers: slice) -> Block:
    """The ``layers`` of stacked ``blocks``."""
    return jax.tree.map(lambda leaf: leaf[layers], blocks)
