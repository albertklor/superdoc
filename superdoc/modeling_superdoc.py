import collections
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Check for flex_attention availability (PyTorch 2.5+ with CUDA)
_FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    flex_attention = None
    create_block_mask = None

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import logging
from .configuration_superdoc import SuperDocConfig


logger = logging.get_logger(__name__)


class SuperDocPatchEmbeddings(nn.Module):
    """SuperDoc image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes."""

    def __init__(self, config):
        super().__init__()

        image_size = (
            config.input_size
            if isinstance(config.input_size, collections.abc.Iterable)
            else (config.input_size, config.input_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, position_embedding=None):
        embeddings = self.proj(pixel_values)

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(patch_height, patch_width), mode="bicubic")
            embeddings = embeddings + position_embedding

        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class SuperDocTextEmbeddings(nn.Module):
    """
    SuperDoc text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SuperDocPreTrainedModel(PreTrainedModel):
    config: SuperDocConfig
    base_model_prefix = "layoutlmv3"
    input_modalities = ("image", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, SuperDocModel):
            if self.config.visual_embed:
                nn.init.zeros_(module.cls_token)
                nn.init.zeros_(module.pos_embed)
            if hasattr(module, "visual_bbox"):
                module.visual_bbox.copy_(module.create_visual_bbox(image_size=(module.size, module.size)))
        elif isinstance(module, SuperDocTextEmbeddings):
            module.position_ids.copy_(torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


class SuperDocSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def cogview_attention(self, attention_scores, alpha=32):
        """
        https://huggingface.co/papers/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        block_mask=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Add relative position biases to create combined attention bias
        attention_bias = None
        if self.has_relative_attention_bias or self.has_spatial_attention_bias:
            attention_bias = torch.zeros(batch_size, self.num_attention_heads, seq_length, seq_length,
                                         device=hidden_states.device, dtype=hidden_states.dtype)
            if self.has_relative_attention_bias and rel_pos is not None:
                attention_bias = attention_bias + rel_pos
            if self.has_spatial_attention_bias and rel_2d_pos is not None:
                attention_bias = attention_bias + rel_2d_pos

        # Try to use flex_attention for TRUE sparse computation (CUDA only)
        # This achieves O(N*W) complexity instead of O(N²)
        use_flex_attention = (
            _FLEX_ATTENTION_AVAILABLE and
            block_mask is not None and
            hidden_states.device.type == 'cuda' and
            not output_attentions  # flex_attention doesn't return attention weights
        )

        if use_flex_attention:
            # Create score_mod function to add relative position biases
            if attention_bias is not None:
                # Capture the bias tensor in a closure for score_mod
                bias_tensor = attention_bias

                def score_mod(score, b, h, q_idx, kv_idx):
                    return score + bias_tensor[b, h, q_idx, kv_idx]
            else:
                score_mod = None

            try:
                # Use flex_attention with block-sparse mask for TRUE O(N*W) computation
                context_layer = flex_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    scale=1.0 / math.sqrt(self.attention_head_size),
                )
                attention_probs = None
            except Exception as e:
                # Fall back to dense attention if flex_attention fails
                logger.warning(f"flex_attention failed: {e}. Falling back to dense attention.")
                use_flex_attention = False

        if not use_flex_attention:
            # Combine attention mask with bias for dense attention
            if attention_mask is not None:
                if attention_bias is not None:
                    attention_bias = attention_bias + attention_mask
                else:
                    attention_bias = attention_mask

            # Use scaled_dot_product_attention for memory efficiency when not outputting attentions
            # This uses Flash Attention or Memory-Efficient Attention when available
            if not output_attentions and attention_bias is not None:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_bias,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    scale=1.0 / math.sqrt(self.attention_head_size),
                )
                attention_probs = None
            elif not output_attentions and attention_bias is None:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    scale=1.0 / math.sqrt(self.attention_head_size),
                )
                attention_probs = None
            else:
                # Fallback to manual computation when we need attention weights
                attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

                if attention_bias is not None:
                    attention_scores = attention_scores + attention_bias

                attention_probs = self.cogview_attention(attention_scores)
                attention_probs = self.dropout(attention_probs)
                context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class SuperDocSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Attention with LayoutLMv2->SuperDoc
class SuperDocAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SuperDocSelfAttention(config)
        self.output = SuperDocSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        block_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            block_mask=block_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Layer with LayoutLMv2->SuperDoc
class SuperDocLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SuperDocAttention(config)
        self.intermediate = SuperDocIntermediate(config)
        self.output = SuperDocOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        block_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            block_mask=block_mask,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SuperDocEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SuperDocLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # Sliding window attention parameters
        self.global_attn_every_n_layers = config.global_attn_every_n_layers
        self.sliding_window_size = config.sliding_window_size

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

    def _compute_spatial_sort_order(self, bbox, text_seq_len=None):
        """
        Compute sort order based on bbox centers for sliding window attention.
        Sorts tokens in reading order (top-to-bottom, left-to-right).

        IMPORTANT: CLS tokens are kept at fixed positions:
        - Text CLS stays at position 0
        - Visual CLS stays at position text_seq_len
        Only content tokens within each modality are sorted.

        Returns:
            sort_indices: (batch, seq) indices to sort tokens by spatial position
            unsort_indices: (batch, seq) indices to restore original order
        """
        batch_size, seq_len = bbox.shape[:2]
        device = bbox.device

        with torch.no_grad():
            # Start with identity mapping
            sort_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).clone()

            # Compute bbox centers
            center_x = (bbox[:, :, 0] + bbox[:, :, 2]) / 2.0  # (batch, seq)
            center_y = (bbox[:, :, 1] + bbox[:, :, 3]) / 2.0  # (batch, seq)

            # Identify padding tokens (bbox=[0,0,0,0])
            is_pad = (bbox == 0).all(dim=-1)  # (batch, seq)

            # Create sort key: (y_center * 10000 + x_center) for reading order
            sort_key = center_y * 10000 + center_x  # (batch, seq)
            # Padding tokens get max value to stay at the end of their section
            sort_key = sort_key.masked_fill(is_pad, float('inf'))

            if text_seq_len is None or text_seq_len == 0:
                # Visual-only mode: CLS at position 0, sort content tokens (positions 1+)
                if seq_len > 1:
                    visual_content_keys = sort_key[:, 1:]  # (batch, seq_len-1)
                    visual_content_order = torch.argsort(visual_content_keys, dim=-1)
                    sort_indices[:, 1:] = visual_content_order + 1
            else:
                if text_seq_len > 1:
                    # Sort text content tokens (positions 1 to text_seq_len-1), keep CLS at 0
                    text_content_keys = sort_key[:, 1:text_seq_len]  # (batch, text_seq_len-1)
                    text_content_order = torch.argsort(text_content_keys, dim=-1)  # relative indices
                    # Convert to absolute indices (add 1 since we skipped position 0)
                    sort_indices[:, 1:text_seq_len] = text_content_order + 1

                if text_seq_len < seq_len - 1:
                    # Sort visual content tokens (positions text_seq_len+1 to end), keep visual CLS at text_seq_len
                    visual_content_keys = sort_key[:, text_seq_len + 1:]  # (batch, num_visual_content)
                    visual_content_order = torch.argsort(visual_content_keys, dim=-1)  # relative indices
                    # Convert to absolute indices
                    sort_indices[:, text_seq_len + 1:] = visual_content_order + text_seq_len + 1

            # Compute inverse permutation (unsort indices)
            unsort_indices = torch.argsort(sort_indices, dim=-1)  # (batch, seq)

        return sort_indices.detach(), unsort_indices.detach()

    def _create_sliding_window_mask(self, seq_len, window_size, text_seq_len, device, dtype, padding_mask=None):
        """
        Create sliding window attention mask with global attention for CLS tokens.

        Args:
            seq_len: total sequence length
            window_size: number of tokens to attend to on each side
            text_seq_len: length of text sequence (visual starts at text_seq_len), None for visual-only
            device: torch device
            dtype: torch dtype for mask values
            padding_mask: (batch, seq_len) tensor where True = padding token

        Returns:
            attention_mask: (batch, 1, seq_len, seq_len) mask with 0 for attend, -inf for masked
        """
        # Start with all masked
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)

        # Create sliding window for all content tokens (allows cross-modality at boundary)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 0.0

        if text_seq_len is None or text_seq_len == 0:
            # Visual-only mode: CLS is at position 0, all tokens are visual
            mask[0, :] = 0.0  # Visual CLS attends to all tokens
            mask[:, 0] = 0.0  # All tokens attend to visual CLS
        elif text_seq_len > 0:
            # Text CLS token (position 0): global attention to ALL text, NOT visual
            # First, ensure CLS row only attends to text (override sliding window)
            mask[0, :] = float('-inf')  # Reset row
            mask[0, :text_seq_len] = 0.0  # Text CLS attends to all text only
            # All text tokens attend to text CLS (column 0 for text rows)
            mask[:text_seq_len, 0] = 0.0

            # Visual CLS token (position text_seq_len): global attention to ALL visual, NOT text
            if text_seq_len < seq_len:
                visual_start = text_seq_len
                # Reset visual CLS row and set only visual attention
                mask[visual_start, :] = float('-inf')
                mask[visual_start, visual_start:] = 0.0  # Visual CLS attends to all visual only
                # All visual tokens attend to visual CLS
                mask[visual_start:, visual_start] = 0.0

        # Expand for batch and head dimensions: (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Incorporate padding mask if provided
        if padding_mask is not None:
            # padding_mask: (batch, seq_len) where True = padding
            # Expand to (batch, 1, 1, seq_len) for broadcasting
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            # Set attention to padding tokens to -inf
            mask = mask.expand(padding_mask.shape[0], -1, -1, -1).clone()
            mask.masked_fill_(padding_mask_expanded, float('-inf'))

        return mask

    def _create_flex_block_mask(self, seq_len, window_size, text_seq_len, batch_size, device, padding_mask=None):
        """
        Create a BlockMask for flex_attention with sliding window pattern and padding support.

        This enables TRUE sparse attention computation (O(N*W) instead of O(N²))
        when running on CUDA with flex_attention.

        Args:
            seq_len: total sequence length
            window_size: number of tokens to attend to on each side
            text_seq_len: length of text sequence (visual starts at text_seq_len), None for visual-only
            batch_size: batch size (needed for per-batch padding masks)
            device: torch device (must be CUDA for flex_attention)
            padding_mask: (batch, seq_len) tensor where True = padding token

        Returns:
            BlockMask for flex_attention, or None if flex_attention unavailable
        """
        if not _FLEX_ATTENTION_AVAILABLE or device.type != 'cuda':
            return None

        # Capture padding_mask for use in mask_mod
        _padding_mask = padding_mask

        # Define the mask_mod function for sliding window + CLS global attention + padding
        def sliding_window_with_cls_and_padding(b, h, q_idx, kv_idx):
            # Sliding window: attend if within window_size
            in_window = torch.abs(q_idx - kv_idx) <= window_size

            if text_seq_len is None or text_seq_len == 0:
                # Visual-only mode: CLS at position 0 has global attention
                is_cls = (q_idx == 0) | (kv_idx == 0)
                attend = in_window | is_cls
            else:
                # Text CLS (pos 0): global attention to text only
                text_cls_query = (q_idx == 0) & (kv_idx < text_seq_len)
                text_cls_key = (kv_idx == 0) & (q_idx < text_seq_len)

                # Visual CLS (pos text_seq_len): global attention to visual only
                visual_cls_query = (q_idx == text_seq_len) & (kv_idx >= text_seq_len)
                visual_cls_key = (kv_idx == text_seq_len) & (q_idx >= text_seq_len)

                attend = in_window | text_cls_query | text_cls_key | visual_cls_query | visual_cls_key

            # Mask out padding tokens (don't attend to padding)
            if _padding_mask is not None:
                is_kv_padding = _padding_mask[b, kv_idx]
                attend = attend & ~is_kv_padding

            return attend

        try:
            # Create block mask with batch dimension for per-batch padding support
            block_mask = create_block_mask(
                sliding_window_with_cls_and_padding,
                B=batch_size,
                H=None,  # Broadcast across heads
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
            return block_mask
        except Exception as e:
            logger.warning(f"Failed to create flex_attention block_mask: {e}. Falling back to dense mask.")
            return None

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # Since this is a simple indexing operation that is independent of the input,
        # no need to track gradients for this operation
        #
        # Without this no_grad context, training speed slows down significantly
        with torch.no_grad():
            rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # Since this is a simple indexing operation that is independent of the input,
        # no need to track gradients for this operation
        #
        # Without this no_grad context, training speed slows down significantly
        with torch.no_grad():
            rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
            rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        patch_height=None,
        patch_width=None,
        text_seq_len=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Determine if we should use sliding window attention
        use_sliding_window = self.global_attn_every_n_layers > 1 and bbox is not None

        # Reorder tokens ONCE at the beginning if using sliding window
        unsort_indices = None
        if use_sliding_window:
            batch_size = hidden_states.shape[0]
            batch_idx = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)

            # Compute spatial sort order based on bbox centers
            sort_indices, unsort_indices = self._compute_spatial_sort_order(
                bbox=bbox,
                text_seq_len=text_seq_len,
            )

            # Reorder hidden states
            hidden_states = hidden_states[batch_idx, sort_indices]

            # Reorder bbox for relative position computation
            bbox = bbox[batch_idx, sort_indices]

            # Reorder position_ids
            position_ids = position_ids[batch_idx, sort_indices]

            # Reorder attention mask if present
            if attention_mask is not None:
                # attention_mask is extended: (batch, 1, 1, seq) or (batch, 1, seq, seq)
                if attention_mask.dim() == 4 and attention_mask.shape[2] == 1:
                    # (batch, 1, 1, seq) - reorder last dim using gather
                    # sort_indices: (batch, seq)
                    # We need to gather along the last dimension
                    attention_mask = torch.gather(
                        attention_mask,
                        dim=3,
                        index=sort_indices.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq)
                    )
                elif attention_mask.dim() == 4:
                    # (batch, 1, seq, seq) - this is rare, but handle it
                    # For now, we'll skip reordering this complex case and just
                    # rely on the sliding_mask for attention pattern
                    pass

            # Create sliding window mask (with CLS global attention and padding)
            seq_len = hidden_states.shape[1]
            # Identify padding tokens from reordered bbox
            padding_mask = (bbox == 0).all(dim=-1)  # (batch, seq)

            # Try to create flex_attention BlockMask for TRUE sparse attention (CUDA only)
            flex_block_mask = self._create_flex_block_mask(
                seq_len=seq_len,
                window_size=self.sliding_window_size,
                text_seq_len=text_seq_len,
                batch_size=batch_size,
                device=hidden_states.device,
                padding_mask=padding_mask,
            )

            # Create dense fallback mask (used on CPU or when flex_attention fails)
            sliding_mask = self._create_sliding_window_mask(
                seq_len=seq_len,
                window_size=self.sliding_window_size,
                text_seq_len=text_seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                padding_mask=padding_mask,
            )
        else:
            flex_block_mask = None

        # Compute relative position embeddings on the (possibly reordered) sequence
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # Store hidden states in original (unsorted) order for consistency
                if unsort_indices is not None:
                    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device).unsqueeze(1)
                    all_hidden_states = all_hidden_states + (hidden_states[batch_idx, unsort_indices],)
                else:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            # Determine whether to use sliding window attention for this layer
            # Use global attention if: (1) config says all layers are global, (2) this is a global layer,
            # or (3) sliding window wasn't set up (e.g., bbox is None)
            use_global_attn = (
                not use_sliding_window or
                self.global_attn_every_n_layers == 1 or
                (i + 1) % self.global_attn_every_n_layers == 0
            )

            # Choose mask: sliding window or full attention
            layer_mask = attention_mask if use_global_attn else sliding_mask
            # Use flex_attention block_mask for sliding window layers (enables TRUE sparse attention)
            layer_block_mask = None if use_global_attn else flex_block_mask

            layer_outputs = layer_module(
                hidden_states,
                layer_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
                block_mask=layer_block_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Reorder back to original positions at the end
        if unsort_indices is not None:
            batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device).unsqueeze(1)
            hidden_states = hidden_states[batch_idx, unsort_indices]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate
class SuperDocIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput
class SuperDocOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperDocModel(SuperDocPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.text_embed:
            self.embeddings = SuperDocTextEmbeddings(config)

        if config.visual_embed:
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embeddings in forward
            self.patch_embed = SuperDocPatchEmbeddings(config)

            self.size = int(config.input_size / config.patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.size * self.size + 1, config.hidden_size))
            self.pos_drop = nn.Dropout(p=0.0)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.register_buffer(
                    "visual_bbox", self.create_visual_bbox(image_size=(self.size, self.size)), persistent=False
                )

            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.encoder = SuperDocEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def create_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        return torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        bbox (`torch.LongTensor` of shape `(batch_size, token_sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.
        token_type_ids (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, token_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        text_seq_len = None
        if pixel_values is not None:
            patch_height, patch_width = (
                int(pixel_values.shape[2] / self.config.patch_size),
                int(pixel_values.shape[3] / self.config.patch_size),
            )
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                text_seq_len = embedding_output.shape[1]
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
            text_seq_len=text_seq_len,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SuperDocClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SuperDocForTokenClassification(SuperDocPreTrainedModel):
    """
    SuperDoc Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
    for sequence labeling (information extraction) tasks such as FUNSD, SROIE, CORD and Kleister-NDA.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = SuperDocModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = SuperDocClassificationHead(config, pool_feature=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.layoutlmv3.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SuperDocForQuestionAnswering(SuperDocPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = SuperDocModel(config)
        self.qa_outputs = SuperDocClassificationHead(config, pool_feature=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.layoutlmv3.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, QuestionAnsweringModelOutput]:
        r"""
        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForQuestionAnswering
        >>> from datasets import load_dataset
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> question = "what's his name?"
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SuperDocForSequenceClassification(SuperDocPreTrainedModel):
    """
    SuperDoc Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the RVL-CDIP dataset.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.layoutlmv3 = SuperDocModel(config)
        self.classifier = SuperDocClassificationHead(config, pool_feature=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.layoutlmv3.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForSequenceClassification
        >>> from datasets import load_dataset
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")
        >>> sequence_label = torch.tensor([1])

        >>> outputs = model(**encoding, labels=sequence_label)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
        )

        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SuperDocForQuestionAnswering",
    "SuperDocForSequenceClassification",
    "SuperDocForTokenClassification",
    "SuperDocModel",
    "SuperDocPreTrainedModel",
]