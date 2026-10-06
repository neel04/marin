# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Looped Fast Track dense models.

A looped rung runs ``prelude`` unique layers, then ``loops`` passes over one shared core of blocks, then ``coda``
unique layers. The model stores only those blocks, but the trainer prices ``num_layers`` executed layers, so a
looped rung keeps the baseline's FLOPs per token.

Two rungs gate every pass and can add passes late in training. ``GatedTransformer`` carries one state through its
passes with the Gated Recurrent Transformer's update, which replaces the share of the state its gate lets go.
``PooledTransformer`` instead adds each pass's change to a pool the coda reads, so no later pass scales an earlier one's
contribution; it is the best recipe in docs/experiments/ledger.md. The pool can also change its core's MLP weights for
each token and pass (``FastWeights``).
"""

import abc
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import random
from jaxtyping import Array, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import (
    Block,
    GrugModelConfig,
    Transformer,
    _embedding_gather,
    _layer_masks,
    _long_layer_schedule,
)
from experiments.grug.looped.blocks import (
    CoreInput,
    FastWeights,
    Hidden,
    LayerRunner,
    PassGate,
    PoolRead,
    fast_layer,
    layer_flops,
    scale_metrics,
    stacked_layers,
)


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
class GatedConfig(LoopedModelConfig):
    """A looped rung whose passes are each gated and follow a schedule that can add passes late in training. Its own
    passes are the Gated Recurrent Transformer's update of one shared state."""

    pass_starts: tuple[float, ...] = ()
    """When each pass starts, as a fraction of training; evaluation runs them all. Passes past ``loops`` reuse the core,
    the gate and the last pass's attention spans, so ``(0, 0, 0.9)`` grows two passes to three for the last tenth of
    training. Empty runs the ``loops`` passes throughout."""
    depth_curve: tuple[int, ...] = ()
    """Further numbers of passes the final evaluation scores the model at."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.pass_starts and (
            self.pass_starts[0] != 0 or list(self.pass_starts) != sorted(self.pass_starts) or self.pass_starts[-1] >= 1
        ):
            raise ValueError(f"Pass starts {self.pass_starts} must rise from 0 and stay below 1")

        if any(depth < 1 for depth in self.depth_curve):
            raise ValueError(f"Depth curve {self.depth_curve} must hold positive numbers of passes")

    @property
    def model_type(self) -> type["GatedTransformer"]:
        return GatedTransformer

    @property
    def pass_schedule(self) -> tuple[float, ...]:
        """When each pass starts, as a fraction of training."""
        return self.pass_starts or (0.0,) * self.loops

    def pass_layout(self, p: int) -> slice:
        """The executed layers whose attention spans pass ``p`` takes."""
        return self.pass_layers(min(p, self.loops - 1))

    def inference_depths(self) -> tuple[int, ...]:
        return tuple(depth for depth in self.depth_curve if depth != len(self.pass_schedule))

    def with_inference_depth(self, depth: int) -> "GatedConfig":
        return dataclasses.replace(self, pass_starts=(0.0,) * depth, depth_curve=())

    def extra_flops_per_token(self) -> float:
        """The gate and its core input every pass, and the passes the schedule adds or skips."""
        d = self.hidden_dim
        # Every pass: the gate's state half and output layer, and the core input's state half. Once per forward: the
        # prelude halves of the gate and the core input.
        return self.scheduled_flops_per_token(each_pass=3 * 2 * d * d) + 2 * 2 * d * d

    def scheduled_flops_per_token(self, each_pass: float) -> float:
        """What the pass schedule adds to the ``loops`` core passes the trainer already prices: ``each_pass`` FLOPs for
        every pass, and the core of each pass the schedule adds or skips, each pass weighted by its share of training."""

        def core(p: int) -> float:
            return sum(layer_flops(self, i) for i in range(*self.pass_layout(p).indices(self.num_layers)))

        flops = sum((1 - start) * (core(p) + each_pass) for p, start in enumerate(self.pass_schedule))
        return flops - sum(core(p) for p in range(self.loops))


@dataclass(frozen=True)
class PoolConfig(GatedConfig):
    """A gated rung whose passes add their changes to a pool the coda reads, instead of replacing a carried state."""

    fast_rank: int = 0
    """The rank of the pool's per-token change of the core blocks' MLP input projections (``FastWeights``), or 0."""

    @property
    def model_type(self) -> type["PooledTransformer"]:
        return PooledTransformer

    def extra_flops_per_token(self) -> float:
        """The gate, the pool's read and the fast weights every pass, and the passes the schedule adds or skips."""
        d, r = self.hidden_dim, self.fast_rank
        # Each core block's scales from the pool, and the low-rank changes of its MLP's gate and up projections.
        fast = self.core_blocks * (2 * d * r + 2 * (2 * d * r + 2 * r * self.intermediate_dim))
        # Every pass: the gate's state half and output layer, the read, and the fast weights. Once per forward: the
        # gate's prelude half.
        return self.scheduled_flops_per_token(each_pass=3 * 2 * d * d + fast) + 2 * d * d


ConfigT = TypeVar("ConfigT", bound=LoopedModelConfig)


def looped(base: GrugModelConfig, config_type: type[ConfigT] = LoopedModelConfig, **loop_fields) -> ConfigT:
    """``base`` as a looped rung of ``config_type`` with the given loop fields."""
    return config_type(**{field.name: getattr(base, field.name) for field in fields(base)}, **loop_fields)


# Named variants for ``run_fast_track --variant``, so each ledger row maps to one configuration. The mid2 variants keep
# the first and last two layers unique and run the middle two blocks twice: Hyperloop's (arXiv 2604.21254) 25/50/25
# split, looped twice as SMELT (arXiv 2609.01343) recommends.
_HALF_LOOPED = dict(prelude=2, coda=2, loops=2)
_GATED = dict(config_type=GatedConfig, **_HALF_LOOPED)
_POOLED = dict(config_type=PoolConfig, **_HALF_LOOPED)
# A third pass over the core for the last tenth of training.
_GROW_TWO_PASSES = dict(pass_starts=(0.0, 0.0, 0.9))
# The dense rung's 8 distinct blocks, its middle four in one gated pass, which grows a second pass for the last tenth.
_ONE_PASS = dict(loops=1)
_GROW_ONE_PASS = dict(pass_starts=(0.0, 0.9))
# Deeper growth, a pass more every tenth of training from 80%, 70% or 60% of the way through, each scored at further
# numbers of passes at the end.
_GROW_TWICE = dict(pass_starts=(0.0, 0.8, 0.9), depth_curve=(1, 2, 4, 6))
_GROW_THRICE = dict(pass_starts=(0.0, 0.7, 0.8, 0.9), depth_curve=(1, 2, 3, 6))
_GROW_FOUR_TIMES = dict(pass_starts=(0.0, 0.6, 0.7, 0.8, 0.9), depth_curve=(1, 3, 6))
# grow2's rung with each core block's MLP programmed per token by the pool, at rank 32: the best rung within about 1.2x
# the reference's training compute.
_FAST = dict(fast_rank=32)


VARIANTS: dict[str, Callable[[GrugModelConfig], LoopedModelConfig]] = {
    "mid2": partial(looped, **_HALF_LOOPED),
    "mid2-grt": partial(looped, **_GATED),
    "mid2-grt-grow": partial(looped, **_GATED | _GROW_TWO_PASSES),
    "mid4-grt": partial(looped, **_GATED | _ONE_PASS),
    "mid4-grt-grow": partial(looped, **_GATED | _ONE_PASS | _GROW_ONE_PASS),
    "mid2-pool": partial(looped, **_POOLED),
    "mid2-pool-grow": partial(looped, **_POOLED | _GROW_TWO_PASSES),
    "mid4-pool-grow": partial(looped, **_POOLED | _ONE_PASS | _GROW_ONE_PASS),
    "mid4-pool-grow2": partial(looped, **_POOLED | _ONE_PASS | _GROW_TWICE),
    "mid4-pool-grow3": partial(looped, **_POOLED | _ONE_PASS | _GROW_THRICE),
    "mid4-pool-grow4": partial(looped, **_POOLED | _ONE_PASS | _GROW_FOUR_TIMES),
    "mid4-pool-grow2-fast": partial(looped, **_POOLED | _ONE_PASS | _GROW_TWICE | _FAST),
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
        progress: Array | None = None,
    ) -> tuple[Hidden, dict[str, Array]]:
        del progress
        cfg = self.config
        blocks = self.stacked_blocks.stacked
        prelude = stacked_layers(blocks, cfg.prelude_layers)
        core = stacked_layers(blocks, cfg.pass_layers(0))
        coda = stacked_layers(blocks, cfg.stored_coda)

        embedded = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(embedded))

        batch_size, seq_len = hidden.shape[:2]
        run = self._layer_runner(AttentionMask.causal() if mask is None else mask, batch_size, seq_len)
        hidden = run(hidden, prelude, cfg.prelude_layers)
        hidden = self._loop(hidden, core, run)
        hidden = run(hidden, coda, cfg.coda_layers)

        return self.final_gated_norm(self.final_norm(hidden)), {}

    def _loop(self, hidden: Hidden, core: Block, run: LayerRunner) -> Hidden:
        """Run the shared ``core`` once per pass."""
        for p in range(self.config.loops):
            hidden = run(hidden, core, self.config.pass_layers(p))

        return hidden

    def _layer_runner(self, mask: AttentionMask | Array, batch_size: int, seq_len: int) -> LayerRunner:
        cfg = self.config
        mask_for_layer = _layer_masks(mask, cfg, batch_size, seq_len)
        return LayerRunner(mask_for_layer, _long_layer_schedule(cfg.num_layers, cfg.global_every))


class ScheduledTransformer(LoopedTransformer):
    """A looped rung whose passes over the core follow ``GatedConfig``'s schedule, each under a ``PassGate``; a subclass
    says what a pass does.

    Stores the prelude and the coda as ``stacked_blocks`` and the core as ``core``: its own tree, as the ledger's runs
    did. The fields stay empty until a subclass's ``init`` fills them; ``Transformer.init`` builds the model without
    them.
    """

    config: GatedConfig = eqx.field(static=True)
    core: Block | None = None
    gate: PassGate | None = None

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | Array | None = None,
        progress: Array | None = None,
    ) -> tuple[Hidden, dict[str, Array]]:
        """The final-normalized hidden states and the passes' metrics. ``progress`` decides which grown passes run in
        training; without it, as in evaluation, every scheduled pass runs."""
        cfg = self.config
        blocks = self.stacked_blocks.stacked
        prelude = stacked_layers(blocks, slice(0, cfg.prelude))
        coda = stacked_layers(blocks, slice(cfg.prelude, cfg.prelude + cfg.coda))

        embedded = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(embedded))
        batch_size, seq_len = hidden.shape[:2]
        run = self._layer_runner(AttentionMask.causal() if mask is None else mask, batch_size, seq_len)
        prefix = run(hidden, prelude, cfg.prelude_layers)

        one_pass = self._pass(prefix, run)
        hidden = prefix
        pass_inputs = []
        for p, start in enumerate(cfg.pass_schedule):
            pass_inputs.append(hidden)
            if progress is None or start == 0:
                hidden = one_pass(hidden, p)
            else:
                hidden = jax.lax.cond(progress >= start, lambda h, p=p: one_pass(h, p), lambda h: h, hidden)

        metrics = self._pass_metrics(pass_inputs, progress)
        hidden = run(hidden, coda, cfg.coda_layers)
        return self.final_gated_norm(self.final_norm(hidden)), metrics

    @abc.abstractmethod
    def _pass(self, prefix: Hidden, run: LayerRunner) -> Callable[[Hidden, int], Hidden]:
        """The update of each pass ``p``, from the hidden states before it to the ones after, given the prelude output
        ``prefix``."""

    def _pass_metrics(self, pass_inputs: list[Hidden], progress: Array | None) -> dict[str, Array]:
        """Scalars to log about the passes, given what each one started from."""
        return {}


class GatedTransformer(ScheduledTransformer):
    """Each pass is the Gated Recurrent Transformer's update: ``state <- keep * state + (1 - keep) * core(W [state ;
    prelude])``."""

    core_input: CoreInput | None = None

    @staticmethod
    def init(cfg: GatedConfig, *, key: PRNGKeyArray) -> "GatedTransformer":
        # Split as the ledger's runs were, so the rung's initialization is unchanged.
        keys = random.split(key, 9)
        model = _with_core_apart(LoopedTransformer.init(cfg, key=keys[0]), cfg, gate_key=keys[2])
        return dataclasses.replace(model, core_input=CoreInput.init(cfg, key=keys[3]))

    def _pass(self, prefix: Hidden, run: LayerRunner) -> Callable[[Hidden, int], Hidden]:
        gate, core_input, core = self.gate, self.core_input, self.core
        assert gate is not None and core_input is not None and core is not None, "GatedTransformer.init adds them"
        gate_term = gate.prelude_term(prefix)
        input_term = core_input.prelude_term(prefix)

        def one_pass(state: Hidden, p: int) -> Hidden:
            output = run(core_input(state, input_term), core, self.config.pass_layout(p))
            keep = gate.keep(state, gate_term)
            return keep * state + (1 - keep) * output

        return one_pass


class PooledTransformer(ScheduledTransformer):
    """Each pass adds its change to a pool that starts as the prelude output ``P`` and that the coda reads:

        u = P + read(A)               the core's input: the prelude output, plus what the pass reads of the pool
        A <- A + write * (core(u) - u)

    with ``write = 1 - keep`` from the gate. Every pass's contribution reaches the coda through additions alone, a
    residual connection across passes where the gated update's is a highway: no later pass scales it. The core's input
    stays anchored on ``P``; were the core to read the pool itself, this would be the gated update with ``keep = 1 -
    write``. A pass writes its core's change of its input, so it adds its transformation, not another copy of ``P`` or
    of its read.
    """

    config: PoolConfig = eqx.field(static=True)
    read: PoolRead | None = None
    fast: FastWeights | None = None

    @staticmethod
    def init(cfg: PoolConfig, *, key: PRNGKeyArray) -> "PooledTransformer":
        # Split as the gated rung does, so the pooled rung starts from the same blocks and gate.
        keys = random.split(key, 9)
        model = _with_core_apart(LoopedTransformer.init(cfg, key=keys[0]), cfg, gate_key=keys[2])
        model = dataclasses.replace(model, read=PoolRead.init(cfg, key=keys[3]))
        if cfg.fast_rank:
            fast = FastWeights.init(cfg, blocks=cfg.core_blocks, rank=cfg.fast_rank, key=keys[6])
            model = dataclasses.replace(model, fast=fast)
        return model

    def _pass(self, prefix: Hidden, run: LayerRunner) -> Callable[[Hidden, int], Hidden]:
        gate, read, core, fast = self.gate, self.read, self.core, self.fast
        assert gate is not None and read is not None and core is not None, "PooledTransformer.init adds them"
        gate_term = gate.prelude_term(prefix)

        def one_pass(pool: Hidden, p: int) -> Hidden:
            inputs = prefix + read(pool)
            layers = self.config.pass_layout(p)
            if fast is None:
                output = run(inputs, core, layers)
            else:
                output = run(inputs, core, layers, partial(fast_layer, features=read.norm(pool)), fast)
            write = 1 - gate.keep(pool, gate_term)
            return pool + write * (output - inputs)

        return one_pass

    def _pass_metrics(self, pools: list[Hidden], progress: Array | None) -> dict[str, Array]:
        """With fast weights, each pass's RMS of their scales, and the RMS of their change since the pass before relative
        to it: a change of a few percent means the weights are the same in every pass. A pass yet to start logs zeros."""
        if self.fast is None:
            return {}
        assert self.read is not None, "PooledTransformer.init adds it"

        metrics = {}
        previous = jnp.zeros(())
        for p, (pool, start) in enumerate(zip(pools, self.config.pass_schedule, strict=True)):
            scales = self.fast.scales(self.read.norm(pool))
            rms, change = scale_metrics(scales, previous)
            started = 1.0 if progress is None else (progress >= start).astype(jnp.float32)
            metrics[f"train/fast/scale_rms_pass{p}"] = started * rms
            metrics[f"train/fast/scale_change_pass{p}"] = started * change
            previous = scales
        return metrics


def _with_core_apart(model: LoopedTransformer, cfg: GatedConfig, gate_key: PRNGKeyArray) -> LoopedTransformer:
    """``model``, a ``ScheduledTransformer`` as ``LoopedTransformer.init`` builds it, with its prelude and coda as
    ``stacked_blocks``, its core apart, and its gate."""

    def prelude_and_coda(leaf: Array) -> Array:
        return jnp.concatenate([leaf[cfg.prelude_layers], leaf[cfg.stored_coda]])

    stored = model.stacked_blocks.stacked
    blocks = jax.tree.map(prelude_and_coda, stored)
    return dataclasses.replace(
        model,
        stacked_blocks=dataclasses.replace(model.stacked_blocks, stacked=blocks, num_layers=cfg.prelude + cfg.coda),
        core=stacked_layers(stored, cfg.pass_layers(0)),
        gate=PassGate.init(cfg, key=gate_key),
    )
