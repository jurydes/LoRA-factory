"""
2026.10.7
2026.10.7
5.17.0
1.15.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

# The stance is process wide, so whether Unsloth owns it lives in unsloth_zoo, shared by
# every generated module: inference in one model's module, training in another's.
from unsloth_zoo.temporary_patches.utils import UNSLOTH_EAGER_STANCE_OWNED, unsloth_claim_eager_stance, unsloth_owns_stance, unsloth_release_stance

def unsloth_training_stance():
    # eager_on_recompile is global: left on by generate() or eval forwards, every recompile
    # in a later training step (new sequence length, model.train() without for_training)
    # would silently run the eager body. Training forwards restore the default stance,
    # unless the user chose eager_on_recompile themselves. Every set_stance call installs a
    # fresh stance object, so ours is still active only if the current one `is` it; a user
    # set_stance since then (even to eager_on_recompile again) is theirs and stays.
    global INFERENCE_RUNS
    INFERENCE_RUNS = 0
    if len(UNSLOTH_EAGER_STANCE_OWNED) != 0 and torch_dynamo_eval_frame is not None:
        current = torch_dynamo_eval_frame._stance
        # Ownership is only dropped once ours is reset: a temporary stance scope's exit restores
        # one of our exact objects, which a later training forward must still recognise.
        if current.stance == "eager_on_recompile" and unsloth_owns_stance(current):
            # Only the reset object: an outer scope may still restore another one of ours.
            # Put back what the caller had before Unsloth switched (force_eager, a backend, ...).
            previous = unsloth_release_stance(current)
            if previous is not None and hasattr(torch_dynamo_eval_frame, "_set_stance"):
                torch_dynamo_eval_frame._set_stance(previous)
            else:
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE


from unsloth_zoo.loss_utils import (
    HAS_CUT_CROSS_ENTROPY,
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)
from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy, unsloth_loss_count_kwargs, unsloth_fsdp_unshard
from unsloth_zoo.fused_losses.aux_loss import unsloth_ga_scale_aux_loss

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 1, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr):
        if attr == "to": return return_none
        # A catch-all __getattr__ makes hasattr() true for every name, dunders included, so the
        # sentinel answers yes to protocol probes it cannot honour. torch.distributed's output cast
        # tests `hasattr(x, "__dataclass_fields__")` and then calls `dataclasses.replace(x)` on
        # whatever said yes, so with FSDP2 mixed precision every step died in
        # `TypeError: replace() should be called on dataclass instances` (unsloth#409, reached
        # through `_fsdp_state._cast_output_dtype`). Protocol probes get an honest AttributeError;
        # ordinary attribute access still gets the callable that explains UNSLOTH_RETURN_LOGITS.
        if len(attr) > 4 and attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")
        return raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
    # Stateless pickling so accelerate gather_object works on the sentinel
    def __reduce__(self): return (type(self), ())
    # Gathered copies must compare equal in accelerate debug mode
    def __eq__(self, other): return type(other).__name__ == "EmptyLogits"
    __hash__ = object.__hash__
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass
# The loop above stomps pickle hooks with stubs returning None; restore them.
for function in ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"):
    try: delattr(EMPTY_LOGITS, function)
    except Exception: pass
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
from unsloth_zoo.temporary_patches.utils import torch_compiler_disable_unless_decode
from unsloth_zoo.temporary_patches.common import torch_compile
import transformers
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.gemma3.modeling_gemma3 import (Callable, torch, nn, init, ACT2FN, Cache, PreTrainedConfig, GenerationMixin, use_kernel_forward_from_hub, _preprocess_mask_arguments, blockwise_overlay, create_causal_mask, maybe_pad_block_sequence_ids, sliding_window_overlay, BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, PreTrainedModel, Unpack, TransformersKwargs, can_return_tuple, deprecate_kwarg, maybe_autocast, Gemma3Config, Gemma3TextConfig, Gemma3CausalLMOutputWithPast, Gemma3PreTrainedModel, Gemma3TextModel, Gemma3ForCausalLM, Gemma3Model, Gemma3ForConditionalGeneration, create_masks_for_generate, _gemma3_rms_norm_generic, flatten_for_elementwise_norm, unwrap_norm_weight, create_causal_mask, create_masks_for_generate)

@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma3MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return Gemma3MLP_forward(self, x=x)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3RMSNorm_forward(self, x):
    x_2d, shape = flatten_for_elementwise_norm(x)
    out = _gemma3_rms_norm_generic(x_2d, unwrap_norm_weight(self.weight), self.eps)
    return out.reshape(shape)

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma3RotaryEmbedding_forward(self, x, position_ids, layer_type):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

    inv_freq_expanded = (
        inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(dtype=torch.float, device=x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    # Disable any outside autocast context if any, to really force fp32
    with maybe_autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma3RotaryEmbedding(nn.Module):
    @deprecate_kwarg("device", version="5.18")
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    @deprecate_kwarg("device", version="5.18")
    def compute_default_rope_parameters(
        config: Gemma3TextConfig, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            layer_type (`str`):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor


    def forward(self, x, position_ids, layer_type):
        return Gemma3RotaryEmbedding_forward(self, x=x, position_ids=position_ids, layer_type=layer_type)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
@use_kernel_forward_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:

        if isinstance(attention_mask, dict):

            attention_mask = attention_mask.get(getattr(module, 'layer_type', None), None)

        if attention_mask is not None:

            attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(attn_weights.dtype).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights



def forward_past_key_values(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
    return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs.pop("cache_position", None), **kwargs)

class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not self.config.use_bidirectional_attention

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.layer_type == "sliding_attention"

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        return forward_past_key_values(self, hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)


def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


@torch_compiler_disable_unless_decode
@can_return_tuple
def Gemma3ForCausalLM_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

    >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    if not torch.compiler.is_compiling() and torch.is_grad_enabled() and getattr(self, "training", False):
        unsloth_training_stance()
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    # Stock ForCausalLMLoss uses caller pre-shifted targets (context parallel, padding-free) as is.
    explicit_shift_labels = None
    if (kwargs) != () and type(kwargs) is dict:
        explicit_shift_labels = (kwargs).get("shift_labels", None)
        if not torch.is_tensor(explicit_shift_labels): explicit_shift_labels = None
    
    if labels is not None: unsloth_fsdp_unshard(self.lm_head)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        # Skipped while tracing (set_stance raises there, and the counter would guard every step)
        # and around a compiled decode step, which eager_on_recompile would otherwise freeze.
        if not torch.compiler.is_compiling() and not UNSLOTH_DECODE_COMPILE[0]:
            if torch_dynamo_eval_frame is not None:
                old_stance = torch_dynamo_eval_frame._stance.stance
            else:
                old_stance = None
            if old_stance is not None and INFERENCE_RUNS == 1:
                # Skip guards and return to eager -> we still need guards!
                # Already eager (another model's inference, or the user's choice): leave that stance
                # object, and so its ownership, as it is. Otherwise switch and claim the new object.
                if old_stance != "eager_on_recompile":
                    previous = getattr(torch_dynamo_eval_frame, "_stance", None)
                    torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
                    # The stance object itself, or True where torch keeps no such object.
                    unsloth_claim_eager_stance(getattr(torch_dynamo_eval_frame, "_stance", True), previous)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Removing compiler guards after 1 inference run. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
            elif old_stance == "eager_on_recompile":
                pass
            elif old_stance == "default" and INFERENCE_RUNS > 1:
                # Reset compiler stance
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Reseting guards. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
                INFERENCE_RUNS = 0
            INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Inference returns these logits to the caller, so they must carry the same
        # scale and softcap transforms the training branches below apply. Leaving
        # them off does not change greedy decoding, since tanh is monotonic, but it
        # does change the distribution: sampling, returned logprobs and any scoring
        # that reads the logits all see uncapped values.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY) and NOT_RETURN_LOGITS and _can_fuse_loss(self.loss_function) and labels is not None and not requires_grad_ and explicit_shift_labels is None and getattr(self.lm_head, "bias", None) is None:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and NOT_RETURN_LOGITS:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        _labels = labels if explicit_shift_labels is None else explicit_shift_labels
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(_labels, _labels.dim() - 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = _labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) != () else 0,
            shift_labels         = explicit_shift_labels is None,
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and not NOT_RETURN_LOGITS:
        # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        # already materialised the full lm_head matmul; apply the captured logit
        # scale/softcap transforms and route loss through self.loss_function on
        # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        return Gemma3ForCausalLM_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)

Gemma3ForCausalLM.__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3MultiModalProjector_forward(self, vision_outputs: torch.Tensor):
    batch_size, _, hidden_size = vision_outputs.shape

    reshaped_vision_outputs = vision_outputs.transpose(1, 2)
    reshaped_vision_outputs = reshaped_vision_outputs.reshape(
        batch_size, hidden_size, self.patches_per_image, self.patches_per_image
    )
    reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

    pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
    pooled_vision_outputs = pooled_vision_outputs.flatten(2)
    pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

    normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

    projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
    return projected_vision_outputs.type_as(vision_outputs)

class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        return Gemma3MultiModalProjector_forward(self, vision_outputs=vision_outputs)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def get_block_sequence_ids_for_mask(token_type_ids: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    # First find where a new image block starts: 1 if image and previous not image
    # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
    is_image = (token_type_ids == 1).to(device=device)
    is_previous_image = nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~is_previous_image
    group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    block_sequence_ids = torch.where(is_image, group_ids, -1)
    return block_sequence_ids


def create_masks_for_vision_model(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    block_sequence_ids: torch.Tensor,
) -> dict:
    """Create full_attention and sliding_attention masks with correct composition.

    For global (full attention) layers:  OR(causal, blockwise)
    For local (sliding window) layers:  AND(sliding_window, OR(causal, blockwise))
    """
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    # Full attention: OR(causal, blockwise) — use block_sequence_ids directly
    full_mask = create_causal_mask(**mask_kwargs, block_sequence_ids=block_sequence_ids)

    # The sliding mask must be sized against a `sliding_attention` layer, but layer 0 may not be one
    # and `create_causal_mask` defaults to the first `full_attention` layer, which has a different kv_length.
    if getattr(past_key_values, "is_sliding", None) and True in past_key_values.is_sliding:
        sliding_layer_idx = past_key_values.is_sliding.index(True)
    else:
        sliding_layer_idx = 0

    # We need to manually pad the sequence IDs for the sliding mask
    # as it's passed as an `or_mask_function` which bypasses internal padding.
    early_exit, _, _, _, kv_length, _, kv_offset = _preprocess_mask_arguments(
        **mask_kwargs,
        layer_idx=sliding_layer_idx,
    )
    if early_exit:
        padded_block_sequence_ids = block_sequence_ids
    else:
        padded_block_sequence_ids = maybe_pad_block_sequence_ids(
            block_sequence_ids, attention_mask, kv_length, kv_offset
        )

    # Sliding attention: AND(sliding_window, OR(causal, blockwise))
    # Pass blockwise as or_mask_function (applied as step 2 in create_causal_mask)
    # Pass sliding_window as and_mask_function (applied as step 3, after OR)
    sliding_mask = create_causal_mask(
        **mask_kwargs,
        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
        and_mask_function=sliding_window_overlay(config.sliding_window),
        layer_idx=sliding_layer_idx,
    )

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


@torch_compiler_disable_unless_decode
@can_return_tuple
def Gemma3ForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **lm_kwargs: Unpack[TransformersKwargs],
) -> tuple | Gemma3CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import httpx
    >>> from io import BytesIO
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    >>> messages = [
    ...     {
    ...         "role": "system",
    ...         "content": [
    ...             {"type": "text", "text": "You are a helpful assistant."}
    ...         ]
    ...     },
    ...     {
    ...         "role": "user", "content": [
    ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    ...             {"type": "text", "text": "Where is the cat standing?"},
    ...         ]
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenize=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
    ```
    """
    if not torch.compiler.is_compiling() and torch.is_grad_enabled() and getattr(self, "training", False):
        unsloth_training_stance()
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=mask_attention_mask_out(labels = labels, attention_mask = attention_mask),
        return_dict=True,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (lm_kwargs) != () and type(lm_kwargs) is dict:
        n_items = (lm_kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (lm_kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    # Stock ForCausalLMLoss uses caller pre-shifted targets (context parallel, padding-free) as is.
    explicit_shift_labels = None
    if (lm_kwargs) != () and type(lm_kwargs) is dict:
        explicit_shift_labels = (lm_kwargs).get("shift_labels", None)
        if not torch.is_tensor(explicit_shift_labels): explicit_shift_labels = None
    
    if labels is not None: unsloth_fsdp_unshard(self.lm_head)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        # Skipped while tracing (set_stance raises there, and the counter would guard every step)
        # and around a compiled decode step, which eager_on_recompile would otherwise freeze.
        if not torch.compiler.is_compiling() and not UNSLOTH_DECODE_COMPILE[0]:
            if torch_dynamo_eval_frame is not None:
                old_stance = torch_dynamo_eval_frame._stance.stance
            else:
                old_stance = None
            if old_stance is not None and INFERENCE_RUNS == 1:
                # Skip guards and return to eager -> we still need guards!
                # Already eager (another model's inference, or the user's choice): leave that stance
                # object, and so its ownership, as it is. Otherwise switch and claim the new object.
                if old_stance != "eager_on_recompile":
                    previous = getattr(torch_dynamo_eval_frame, "_stance", None)
                    torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
                    # The stance object itself, or True where torch keeps no such object.
                    unsloth_claim_eager_stance(getattr(torch_dynamo_eval_frame, "_stance", True), previous)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Removing compiler guards after 1 inference run. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
            elif old_stance == "eager_on_recompile":
                pass
            elif old_stance == "default" and INFERENCE_RUNS > 1:
                # Reset compiler stance
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Reseting guards. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
                INFERENCE_RUNS = 0
            INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Inference returns these logits to the caller, so they must carry the same
        # scale and softcap transforms the training branches below apply. Leaving
        # them off does not change greedy decoding, since tanh is monotonic, but it
        # does change the distribution: sampling, returned logprobs and any scoring
        # that reads the logits all see uncapped values.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY) and NOT_RETURN_LOGITS and _can_fuse_loss(self.loss_function) and labels is not None and not requires_grad_ and explicit_shift_labels is None and getattr(self.lm_head, "bias", None) is None:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and NOT_RETURN_LOGITS:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        _labels = labels if explicit_shift_labels is None else explicit_shift_labels
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(_labels, _labels.dim() - 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = _labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
            shift_labels         = explicit_shift_labels is None,
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and not NOT_RETURN_LOGITS:
        # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        # already materialised the full lm_head matmul; apply the captured logit
        # scale/softcap transforms and route loss through self.loss_function on
        # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.config.text_config.vocab_size, **lm_kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits=logits, labels=labels.to(self.lm_head.weight.device), vocab_size=self.config.text_config.vocab_size, **lm_kwargs)


    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )

class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]):
        return self.model.get_image_features(pixel_values, **kwargs)


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Gemma3CausalLMOutputWithPast:
        return Gemma3ForConditionalGeneration_forward(self, input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **lm_kwargs)

    def prepare_inputs_for_generation(self, input_ids, use_cache=True, is_first_iteration=False, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, use_cache=use_cache, is_first_iteration=is_first_iteration, **kwargs
        )
        if not (is_first_iteration or not use_cache):
            # Don't pass to not apply bidirectional mask on top
            model_inputs["token_type_ids"] = None

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        if token_type_ids is not None:
            block_sequence_ids = get_block_sequence_ids_for_mask(token_type_ids, device=inputs_embeds.device)
            return create_masks_for_vision_model(
                block_sequence_ids=block_sequence_ids,
                **mask_kwargs,
            )

        return create_masks_for_generate(**mask_kwargs)

Gemma3ForConditionalGeneration.__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True
