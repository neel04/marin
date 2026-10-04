# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Looped Fast Track dense models.

A looped rung runs ``prelude`` unique layers, then ``loops`` passes over one shared core of blocks, then ``coda``
unique layers. The model stores only those blocks, but the trainer prices ``num_layers`` executed layers, so a
looped rung keeps the baseline's FLOPs per token.

Two rungs gate every pass and can add passes late in training. ``GatedTransformer`` carries one state through its
passes with the Gated Recurrent Transformer's update, which replaces the share of the state its gate lets go.
``PooledTransformer`` instead adds each pass's change to a pool the coda reads, so no later pass scales an earlier one's
contribution; it is the best recipe in docs/experiments/ledger.md.
"""

import abc
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
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.fast_track.model import (
    _CE_BLOCK_SIZES,
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
class GatedConfig(LoopedModelConfig):
    """A looped rung whose passes are each gated and follow a schedule that can add passes late in training. Its own
    passes are the Gated Recurrent Transformer's update of one shared state."""

    pass_starts: tuple[float, ...] = ()
    """When each pass starts, as a fraction of training; evaluation runs them all. Passes past ``loops`` reuse the core,
    the gate and the last pass's attention spans, so ``(0, 0, 0.9)`` grows two passes to three for the last tenth of
    training. Empty runs the ``loops`` passes throughout."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.pass_starts and (
            self.pass_starts[0] != 0 or list(self.pass_starts) != sorted(self.pass_starts) or self.pass_starts[-1] >= 1
        ):
            raise ValueError(f"Pass starts {self.pass_starts} must rise from 0 and stay below 1")

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
            return sum(_layer_flops(self, i) for i in range(*self.pass_layout(p).indices(self.num_layers)))

        flops = sum((1 - start) * (core(p) + each_pass) for p, start in enumerate(self.pass_schedule))
        return flops - sum(core(p) for p in range(self.loops))


@dataclass(frozen=True)
class PoolConfig(GatedConfig):
    """A gated rung whose passes add their changes to a pool the coda reads, instead of replacing a carried state."""

    @property
    def model_type(self) -> type["PooledTransformer"]:
        return PooledTransformer

    def extra_flops_per_token(self) -> float:
        """The gate and the pool's read every pass, and the passes the schedule adds or skips."""
        d = self.hidden_dim
        # Every pass: the gate's state half and output layer, and the read. Once per forward: the gate's prelude half.
        return self.scheduled_flops_per_token(each_pass=3 * 2 * d * d) + 2 * d * d


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


VARIANTS: dict[str, Callable[[GrugModelConfig], LoopedModelConfig]] = {
    "mid2": partial(looped, **_HALF_LOOPED),
    "mid2-grt": partial(looped, **_GATED),
    "mid2-grt-grow": partial(looped, **_GATED | _GROW_TWO_PASSES),
    "mid4-grt": partial(looped, **_GATED | _ONE_PASS),
    "mid4-grt-grow": partial(looped, **_GATED | _ONE_PASS | _GROW_ONE_PASS),
    "mid2-pool": partial(looped, **_POOLED),
    "mid2-pool-grow": partial(looped, **_POOLED | _GROW_TWO_PASSES),
    "mid4-pool-grow": partial(looped, **_POOLED | _ONE_PASS | _GROW_ONE_PASS),
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
    def init(cfg: LoopedModelConfig, *, key: PRNGKeyArray) -> "PassGate":
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
    def init(cfg: LoopedModelConfig, *, key: PRNGKeyArray) -> "CoreInput":
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
    def init(cfg: LoopedModelConfig, *, key: PRNGKeyArray) -> "PoolRead":
        d = cfg.hidden_dim
        return PoolRead(norm=RMSNorm.init(d, cfg.layer_norm_eps), project=_init_weight(key, (d, d), cfg.initializer_std))

    def __call__(self, pool: Hidden) -> Hidden:
        return self.norm(pool) @ self.project


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
    ) -> tuple[Hidden, dict[str, Array]]:
        return self._forward(token_ids, mask, progress=None), {}

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
        progress: Array | None = None,
    ) -> Array | tuple[Array, dict[str, Array]]:
        """The next-token loss; ``progress`` decides which grown passes run in training."""
        hidden = self._forward(token_ids, mask, progress)
        weight = loss_weight.astype(loss_dtype)
        loss = _cross_entropy(
            hidden, self.output_proj, _next_tokens(token_ids), weight, reduction, logsumexp_weight, loss_dtype
        )
        if reduction == "none" or not return_router_metrics:
            return loss
        return loss, {"train/cross_entropy_loss": loss}

    def _forward(
        self, token_ids: Int[Array, "B S"], mask: AttentionMask | Array | None, progress: Array | None
    ) -> Hidden:
        """The final-normalized hidden states. Without ``progress``, as in evaluation, every scheduled pass runs."""
        cfg = self.config
        blocks = self.stacked_blocks.stacked
        prelude = _layers(blocks, slice(0, cfg.prelude))
        coda = _layers(blocks, slice(cfg.prelude, cfg.prelude + cfg.coda))

        embedded = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(embedded))
        batch_size, seq_len = hidden.shape[:2]
        run = self._layer_runner(AttentionMask.causal() if mask is None else mask, batch_size, seq_len)
        prefix = run(hidden, prelude, cfg.prelude_layers)

        one_pass = self._pass(prefix, run)
        hidden = prefix
        for p, start in enumerate(cfg.pass_schedule):
            if progress is None or start == 0:
                hidden = one_pass(hidden, p)
            else:
                hidden = jax.lax.cond(progress >= start, lambda h, p=p: one_pass(h, p), lambda h: h, hidden)

        hidden = run(hidden, coda, cfg.coda_layers)
        return self.final_gated_norm(self.final_norm(hidden))

    @abc.abstractmethod
    def _pass(self, prefix: Hidden, run: RunLayers) -> Callable[[Hidden, int], Hidden]:
        """The update of each pass ``p``, from the hidden states before it to the ones after, given the prelude output
        ``prefix``."""


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

    def _pass(self, prefix: Hidden, run: RunLayers) -> Callable[[Hidden, int], Hidden]:
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

    @staticmethod
    def init(cfg: PoolConfig, *, key: PRNGKeyArray) -> "PooledTransformer":
        # Split as the gated rung does, so the pooled rung starts from the same blocks and gate.
        keys = random.split(key, 9)
        model = _with_core_apart(LoopedTransformer.init(cfg, key=keys[0]), cfg, gate_key=keys[2])
        return dataclasses.replace(model, read=PoolRead.init(cfg, key=keys[3]))

    def _pass(self, prefix: Hidden, run: RunLayers) -> Callable[[Hidden, int], Hidden]:
        gate, read, core = self.gate, self.read, self.core
        assert gate is not None and read is not None and core is not None, "PooledTransformer.init adds them"
        gate_term = gate.prelude_term(prefix)

        def one_pass(pool: Hidden, p: int) -> Hidden:
            inputs = prefix + read(pool)
            output = run(inputs, core, self.config.pass_layout(p))
            write = 1 - gate.keep(pool, gate_term)
            return pool + write * (output - inputs)

        return one_pass


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
        core=_layers(stored, cfg.pass_layers(0)),
        gate=PassGate.init(cfg, key=gate_key),
    )


def _cross_entropy(
    hidden: Float[Array, "... D"],
    lm_head: Float[Array, "D V"],
    labels: Int[Array, "..."],
    weight: Float[Array, "..."] | None,
    reduction: str,
    logsumexp_weight: float | None,
    dtype: jnp.dtype,
) -> Array:
    """The base model's next-token loss on final-normalized ``hidden``."""
    on_tpu = jax.default_backend() == "tpu"
    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
        dtype=dtype,
        implementation="pallas_tpu" if on_tpu else "xla_fast_bwd",
        block_sizes=None if on_tpu else _CE_BLOCK_SIZES,
    )


def _next_tokens(token_ids: Int[Array, "B S"]) -> Int[Array, "B S"]:
    """The token after each position; the last position's, which no weight counts, is 0."""
    return jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)


def _layer_flops(cfg: GrugModelConfig, index: int) -> float:
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


def _layers(blocks: Block, layers: slice) -> Block:
    return jax.tree.map(lambda leaf: leaf[layers], blocks)
