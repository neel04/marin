# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Looped Fast Track dense models.

A looped rung runs ``prelude`` unique layers, then ``loops`` passes over one shared core of blocks, then ``coda``
unique layers. The model stores only those blocks, but the trainer prices ``num_layers`` executed layers, so a
looped rung keeps the baseline's FLOPs per token. ``GatedRecurrentTransformer`` adds a learned gate around each pass,
the best looped recipe in docs/experiments/ledger.md.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from typing import Protocol, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import (
    _REMAT_POLICY,
    Block,
    GrugModelConfig,
    RMSNorm,
    Transformer,
    _embedding_gather,
    _init_weight,
    _layer_masks,
    _long_layer_schedule,
)

Hidden = Float[Array, "B S D"]

# The Gated Recurrent Transformer's initial gate logit: sigmoid(4) = 0.98 of the state is kept at first.
_INITIAL_KEEP_LOGIT = 4.0


class RunLayers(Protocol):
    def __call__(self, hidden: Hidden, blocks: Block, layers: slice) -> Hidden:
        """Run stacked ``blocks`` as the executed ``layers``, which set each block's attention span."""
        ...


@dataclass(frozen=True)
class LoopedModelConfig(GrugModelConfig):
    """A Fast Track dense rung whose executed layers are a prelude, ``loops`` passes over a core, and a coda."""

    prelude: int = 0
    coda: int = 0
    loops: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.dense_mlp:
            raise ValueError("The looped model supports dense blocks only")

        if self.looped_layers <= 0 or self.looped_layers % self.loops != 0:
            raise ValueError(f"{self.looped_layers} looped layers do not split into {self.loops} passes")

    @property
    def model_type(self) -> type["LoopedTransformer"]:
        return LoopedTransformer

    @property
    def looped_layers(self) -> int:
        """Executed layers between the prelude and the coda."""
        return self.num_layers - self.prelude - self.coda

    @property
    def core_blocks(self) -> int:
        return self.looped_layers // self.loops

    # Executed layers, which set each block's attention span.

    @property
    def prelude_layers(self) -> slice:
        return slice(0, self.prelude)

    def pass_layers(self, p: int) -> slice:
        start = self.prelude + p * self.core_blocks
        return slice(start, start + self.core_blocks)

    @property
    def coda_layers(self) -> slice:
        return slice(self.num_layers - self.coda, self.num_layers)

    # Stored blocks: the prelude, one core and the coda. The prelude and the core sit where the prelude and the
    # first pass execute, so only the coda moves.

    @property
    def stored_coda(self) -> slice:
        """The coda's place among the stored blocks, after the prelude and the core."""
        start = self.prelude + self.core_blocks
        return slice(start, start + self.coda)


@dataclass(frozen=True)
class GatedRecurrentConfig(LoopedModelConfig):
    """A looped rung with the Gated Recurrent Transformer's update (arXiv 2608.15062)."""

    @property
    def model_type(self) -> type["GatedRecurrentTransformer"]:
        return GatedRecurrentTransformer


ConfigT = TypeVar("ConfigT", bound=LoopedModelConfig)


def looped(base: GrugModelConfig, config_type: type[ConfigT] = LoopedModelConfig, **loop_fields) -> ConfigT:
    """``base`` as a looped rung of ``config_type`` with the given loop fields."""
    return config_type(**{field.name: getattr(base, field.name) for field in fields(base)}, **loop_fields)


# Named variants for ``run_fast_track --variant``, so each ledger row maps to one configuration. Both keep the first
# and last two layers unique and run the middle two blocks twice, the best layout tried: Hyperloop's (arXiv
# 2604.21254) 25/50/25 split, looped twice as SMELT (arXiv 2609.01343) recommends.
_HALF_LOOPED = dict(prelude=2, coda=2, loops=2)

VARIANTS: dict[str, Callable[[GrugModelConfig], LoopedModelConfig]] = {
    "mid2": partial(looped, **_HALF_LOOPED),
    "mid2-grt": partial(looped, config_type=GatedRecurrentConfig, **_HALF_LOOPED),
}


class LoopedTransformer(Transformer):
    """Stores the prelude, one core and the coda as ``stacked_blocks``, in that order."""

    config: LoopedModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: LoopedModelConfig, *, key: PRNGKeyArray) -> "LoopedTransformer":
        # Transformer.init builds a block for every executed layer; keep only the ones the model stores.
        model = Transformer.init(cfg, key=key)

        def stored(leaf: Array) -> Array:
            prelude_and_core = leaf[: cfg.stored_coda.start]
            coda = leaf[cfg.coda_layers]
            return jnp.concatenate([prelude_and_core, coda])

        blocks = jax.tree.map(stored, model.stacked_blocks.stacked)
        stacked_blocks = dataclasses.replace(model.stacked_blocks, stacked=blocks, num_layers=cfg.stored_coda.stop)
        return dataclasses.replace(model, stacked_blocks=stacked_blocks)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | Array | None = None,
    ) -> tuple[Hidden, dict[str, Array]]:
        cfg = self.config
        blocks = self.stacked_blocks.stacked
        prelude = _layers(blocks, cfg.prelude_layers)
        core = _layers(blocks, cfg.pass_layers(0))
        coda = _layers(blocks, cfg.stored_coda)

        embedded = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(embedded))

        batch_size, seq_len = hidden.shape[:2]
        run = self._layer_runner(AttentionMask.causal() if mask is None else mask, batch_size, seq_len)
        hidden = run(hidden, prelude, cfg.prelude_layers)
        hidden = self._loop(hidden, core, run)
        hidden = run(hidden, coda, cfg.coda_layers)

        return self.final_gated_norm(self.final_norm(hidden)), {}

    def _loop(self, hidden: Hidden, core: Block, run: RunLayers) -> Hidden:
        """Run the shared ``core`` once per pass."""
        for p in range(self.config.loops):
            hidden = run(hidden, core, self.config.pass_layers(p))

        return hidden

    def _layer_runner(self, mask: AttentionMask | Array, batch_size: int, seq_len: int) -> RunLayers:
        """A function that scans stacked blocks as a contiguous range of executed layers."""
        cfg = self.config
        mask_for_layer = _layer_masks(mask, cfg, batch_size, seq_len)
        is_global = _long_layer_schedule(cfg.num_layers, cfg.global_every)

        def run_layer(hidden: Hidden, layer: tuple[Block, Array]) -> tuple[Hidden, None]:
            block, layer_is_global = layer
            block = eqx.filter_checkpoint(block, policy=_REMAT_POLICY)
            layer_mask = mask_for_layer(layer_is_global)

            hidden, _ = block(hidden, layer_mask, disable_rope=layer_is_global, is_global=layer_is_global)
            return hidden, None

        def run(hidden: Hidden, blocks: Block, layers: slice) -> Hidden:
            hidden, _ = jax.lax.scan(run_layer, hidden, (blocks, is_global[layers]))
            return hidden

        return run


class RecurrentCell(eqx.Module):
    """The Gated Recurrent Transformer's step around the shared core.

    Each pass runs the core on a projection of the state and the prelude output, then blends the core's output into
    the state with an elementwise gate: ``state <- g * state + (1 - g) * core(W [state ; prelude])``. The gate is a
    two-layer MLP of the normalized state and prelude output. Its output bias starts at ``_INITIAL_KEEP_LOGIT``, so
    each pass starts out keeping nearly all of the state. The paper also adds noise to the state and the gate during
    training; the trainer gives the forward pass no randomness, so this cell does not.
    """

    project: Float[Array, "2D D"]
    state_norm: RMSNorm
    prelude_norm: RMSNorm
    gate_in: Float[Array, "2D D"]
    gate_in_bias: Float[Array, " D"]
    gate_out: Float[Array, "D D"]
    gate_out_bias: Float[Array, " D"]

    @staticmethod
    def init(cfg: GatedRecurrentConfig, *, key: PRNGKeyArray) -> "RecurrentCell":
        d = cfg.hidden_dim
        project_key, gate_in_key, gate_out_key = random.split(key, 3)
        return RecurrentCell(
            project=_init_weight(project_key, (2 * d, d), cfg.initializer_std),
            state_norm=RMSNorm.init(d, cfg.layer_norm_eps),
            prelude_norm=RMSNorm.init(d, cfg.layer_norm_eps),
            gate_in=_init_weight(gate_in_key, (2 * d, d), cfg.initializer_std),
            gate_in_bias=jnp.zeros(d),
            gate_out=_init_weight(gate_out_key, (d, d), cfg.initializer_std),
            gate_out_bias=jnp.full(d, _INITIAL_KEEP_LOGIT),
        )

    def core_input(self, state: Hidden, prelude: Hidden) -> Hidden:
        return jnp.concatenate([state, prelude], axis=-1) @ self.project

    def keep(self, state: Hidden, prelude: Hidden) -> Hidden:
        """The fraction of each state coordinate the pass keeps."""
        normalized = jnp.concatenate([self.state_norm(state), self.prelude_norm(prelude)], axis=-1)
        hidden = jax.nn.silu(normalized @ self.gate_in + self.gate_in_bias)
        return jax.nn.sigmoid(hidden @ self.gate_out + self.gate_out_bias)


class GatedRecurrentTransformer(LoopedTransformer):
    config: GatedRecurrentConfig = eqx.field(static=True)
    # Empty until ``init`` adds it; ``Transformer.init`` builds the model without it. One cell serves every pass.
    cell: RecurrentCell | None = None

    @staticmethod
    def init(cfg: GatedRecurrentConfig, *, key: PRNGKeyArray) -> "GatedRecurrentTransformer":
        model_key, cell_key = random.split(key)
        model = LoopedTransformer.init(cfg, key=model_key)
        return dataclasses.replace(model, cell=RecurrentCell.init(cfg, key=cell_key))

    def _loop(self, hidden: Hidden, core: Block, run: RunLayers) -> Hidden:
        """``state <- keep * state + (1 - keep) * core(W [state ; prelude])`` per pass, from the prelude output."""
        assert self.cell is not None, "GatedRecurrentTransformer.init adds the cell"
        prelude = hidden
        state = hidden

        for p in range(self.config.loops):
            core_input = self.cell.core_input(state, prelude)
            output = run(core_input, core, self.config.pass_layers(p))

            keep = self.cell.keep(state, prelude)
            state = keep * state + (1 - keep) * output

        return state


def _layers(blocks: Block, layers: slice) -> Block:
    return jax.tree.map(lambda leaf: leaf[layers], blocks)
