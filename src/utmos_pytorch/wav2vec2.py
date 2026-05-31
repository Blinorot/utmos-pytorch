import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


"""
This module is adapted from https://github.com/facebookresearch/fairseq/.
It re-implements the required Wav2Vec2 inference components in PyTorch
without depending on fairseq.
"""


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class Wav2Vec2Config:
    # Feature extractor
    extractor_mode: str = "default"  # "default" or "layer_norm"
    conv_feature_layers: str = (
        "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]"
    )
    conv_bias: bool = False

    # Transformer encoder
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    layer_type: str = "transformer"
    layer_norm_first: bool = False

    # Dropouts
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0

    # Projection / pretraining options kept only for config compatibility
    final_dim: int = 0
    quantize_targets: bool = False
    quantize_input: bool = False
    same_quantizer: bool = False
    target_glu: bool = False
    feature_grad_mult: float = 1.0

    # Positional convolution
    conv_pos: int = 128
    conv_pos_groups: int = 16
    pos_conv_depth: int = 1
    conv_pos_batch_norm: bool = False

    # Length handling
    required_seq_len_multiple: int = 2
    crop_seq_to_multiple: int = 1

    # Kept for compatibility, not used here
    mask_prob: float = 0.65
    mask_selection: str = "static"
    mask_other: float = 0.0
    mask_length: int = 10
    no_mask_overlap: bool = False
    mask_min_space: int = 1
    require_same_masks: bool = True
    mask_dropout: float = 0.0

    mask_channel_prob: float = 0.0
    mask_channel_before: bool = False
    mask_channel_selection: str = "static"
    mask_channel_other: float = 0.0
    mask_channel_length: int = 10
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1

    num_negatives: int = 100
    negatives_from_everywhere: bool = False
    cross_sample_negatives: int = 0
    codebook_negatives: int = 0

    latent_vars: int = 320
    latent_groups: int = 2
    latent_dim: int = 0
    latent_temp: Tuple[float, float, float] = (2.0, 0.5, 0.999995)
    quantizer_depth: int = 1
    quantizer_factor: int = 3
    logit_temp: float = 0.1

    max_positions: int = 100000
    checkpoint_activations: bool = False

    # Conformer/adapters are not supported in this inference-only version
    depthwise_conv_kernel_size: int = 31
    attn_type: str = ""
    pos_enc_type: str = "abs"
    fp16: bool = False
    adp_num: int = -1
    adp_dim: int = 64
    adp_act_fn: str = "relu"
    adp_trf_idx: str = "all"


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------


def parse_conv_feature_layers(s: str) -> List[Tuple[int, int, int]]:
    # Same config style as Fairseq wav2vec2, but no global builtins.
    return eval(s, {"__builtins__": {}}, {})


def get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name in {"gelu_accurate", "gelu_fast"}:
        # Fairseq has variants, but UTMOS/wav2vec_small normally uses "gelu".
        return F.gelu
    if name in {"silu", "swish"}:
        return F.silu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation_fn: {name}")


def pad_tensor_to_multiple(
    x: Tensor,
    multiple: int,
    dim: int = -1,
    value: float = 0.0,
) -> Tuple[Tensor, int]:
    if multiple <= 1:
        return x, 0

    dim = dim if dim >= 0 else x.dim() + dim
    size = x.size(dim)
    remainder = size % multiple

    if remainder == 0:
        return x, 0

    pad_length = multiple - remainder

    pad = [0, 0] * x.dim()
    pad_index = 2 * (x.dim() - dim - 1) + 1
    pad[pad_index] = pad_length

    return F.pad(x, pad, value=value), pad_length


def pad_optional_tensor_to_multiple(
    x: Optional[Tensor],
    multiple: int,
    dim: int = -1,
    value: float = 0.0,
) -> Tuple[Optional[Tensor], int]:
    if x is None:
        return None, 0

    y, pad_length = pad_tensor_to_multiple(
        x,
        multiple=multiple,
        dim=dim,
        value=value,
    )
    return y, pad_length


def index_put(x: Tensor, indices: Tensor, value: float) -> Tensor:
    # Fairseq utility replacement for the simple use case.
    x = x.clone()
    x[indices] = value
    return x


# ---------------------------------------------------------------------
# Fairseq-free helper modules
# ---------------------------------------------------------------------


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim: int = -2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal: bool = False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: Tensor) -> Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def LayerNorm(normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


# ---------------------------------------------------------------------
# Fairseq-compatible but Fairseq-free MultiheadAttention
# ---------------------------------------------------------------------


class MultiheadAttention(nn.Module):
    """
    Fairseq-key-compatible self-attention, but no Fairseq dependency.

    Parameter names intentionally match Fairseq:
      self_attn.q_proj.weight
      self_attn.k_proj.weight
      self_attn.v_proj.weight
      self_attn.out_proj.weight

    Forward intentionally relies on torch.nn.functional.multi_head_attention_forward.
    No xFormers, no incremental state, no pruning, no sparse attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)

        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if before_softmax:
            raise NotImplementedError("before_softmax=True is not supported here.")

        if need_head_weights:
            need_weights = True

        if self.self_attention:
            key = query
            value = query
        elif self.encoder_decoder_attention:
            value = key

        assert key is not None
        assert value is not None

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0], device=query.device, dtype=query.dtype),
            in_proj_bias=torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
            )
            if self.q_proj.bias is not None
            else None,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask.to(torch.bool)
            if key_padding_mask is not None
            else None,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

    def upgrade_state_dict_named(self, state_dict: Dict[str, Tensor], name: str):
        """
        Optional compatibility for very old checkpoints that used in_proj_weight.
        """
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []

        for k in list(state_dict.keys()):
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)

                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict:
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
                    keys_to_remove.append(k_bias)

        for k in keys_to_remove:
            del state_dict[k]

        state_dict.update(items_to_add)


# ---------------------------------------------------------------------
# Conv feature extractor
# ---------------------------------------------------------------------


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            is_layer_norm: bool = False,
            is_group_norm: bool = False,
            conv_bias: bool = False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (
                is_layer_norm and is_group_norm
            ), "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )

            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )

            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()

        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            dim, k, stride = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x: Tensor) -> Tensor:
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


# ---------------------------------------------------------------------
# Positional convolution
# ---------------------------------------------------------------------


def make_conv_pos(e: int, k: int, g: int, is_batch_norm: bool = False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )

    dropout = 0.0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0.0, std=std)
    nn.init.constant_(pos_conv.bias, 0.0)

    if not is_batch_norm:
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(k), nn.GELU())

    return pos_conv


# ---------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Same structure as Fairseq TransformerSentenceEncoderLayer for wav2vec2.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm_first = layer_norm_first

        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        att_args: Any = None,
    ) -> Tuple[Tensor, Tuple[Optional[Tensor], Tensor]]:
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)

            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x

        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


# ---------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        args: Wav2Vec2Config,
        skip_pos_conv: bool = False,
        override_encoder_layer: Optional[int] = None,
    ):
        super().__init__()

        if args.layer_type != "transformer":
            raise ValueError(
                "This Fairseq-free inference module supports only "
                "layer_type='transformer'."
            )

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple
        self.layer_norm_first = args.layer_norm_first
        self.layerdrop = args.encoder_layerdrop

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)

        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e: int, k: int, g: int, l: int):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim,
                k,
                args.conv_pos_groups,
                num_layers,
            )

        elif skip_pos_conv:
            self.pos_conv = None

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
                is_batch_norm=getattr(args, "conv_pos_batch_norm", False),
            )

        if override_encoder_layer is None:
            encoder_layers = args.encoder_layers
        else:
            encoder_layers = override_encoder_layer

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.extract_features(
            x,
            padding_mask=padding_mask,
        )

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x, pad_length = pad_tensor_to_multiple(
            x,
            self.required_seq_len_multiple,
            dim=-2,
            value=0.0,
        )

        if pad_length > 0 and padding_mask is None:
            padding_mask = torch.zeros(
                (x.size(0), x.size(1)),
                dtype=torch.bool,
                device=x.device,
            )
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_optional_tensor_to_multiple(
                padding_mask,
                self.required_seq_len_multiple,
                dim=-1,
                value=1.0,
            )

        x = F.dropout(x, p=self.dropout, training=self.training)

        # [B, T, C] -> [T, B, C]
        x = x.transpose(0, 1)

        r: Optional[Tensor] = None

        for layer in self.layers:
            # In eval mode this is identical to original: no layerdrop.
            # In train mode, the original has stochastic numpy layerdrop.
            # This module is intended for inference only.
            x, (z, lr) = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
            )

        if r is not None:
            x = r

        # [T, B, C] -> [B, T, C]
        x = x.transpose(0, 1)

        if pad_length > 0:
            x = x[:, :-pad_length]

        return x


# ---------------------------------------------------------------------
# Wav2Vec2 feature-only model
# ---------------------------------------------------------------------


class Wav2Vec2Model(nn.Module):
    """
    Fairseq-free Wav2Vec2 inference module.

    Intended to replace Fairseq Wav2Vec2Model for:

        model(source, mask=False, features_only=True)["x"]

    It does NOT implement:
      - masking
      - quantization
      - contrastive pretraining logits
      - Conformer
      - adapters
      - xFormers
      - incremental decoding
      - Fairseq registry/dataclass logic
    """

    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = parse_conv_feature_layers(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.conv_kernel_sizes = [layer[1] for layer in feature_enc_layers]
        self.conv_strides = [layer[2] for layer in feature_enc_layers]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        # Kept as attributes for compatibility with code that calls
        # remove_pretraining_modules().
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None
        self.input_quantizer = None

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Tensor,
    ) -> Tensor:
        for i in range(len(self.conv_kernel_sizes)):
            kernel_size = self.conv_kernel_sizes[i]
            stride = self.conv_strides[i]
            input_lengths = torch.floor((input_lengths - kernel_size) / stride + 1)

        return input_lengths.to(torch.long)

    def forward(
        self,
        source: Tensor,
        padding_mask: Optional[Tensor] = None,
        mask: bool = False,
        features_only: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        if mask is not False:
            raise ValueError("This inference-only module supports only mask=False.")

        if features_only is not True:
            raise ValueError(
                "This inference-only module supports only features_only=True."
            )

        # Original behavior:
        # if feature_grad_mult > 0:
        #     features = self.feature_extractor(source)
        #     if feature_grad_mult != 1.0: GradMultiply.apply(...)
        # else:
        #     with torch.no_grad(): features = self.feature_extractor(source)
        #
        # For inference/eval, gradients are irrelevant. This keeps the same
        # forward values.
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        # [B, C, T'] -> [B, T', C]
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2],
                dtype=features.dtype,
                device=features.device,
            )

            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1

            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).to(torch.bool)
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        x = features

        x = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        return x, padding_mask, unmasked_features

    def extract_features(
        self,
        source: Tensor,
        padding_mask: Optional[Tensor] = None,
        mask: bool = False,
        layer: Optional[int] = None,
        corpus_key: Any = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        return self.forward(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
            corpus_key=corpus_key,
        )

    def remove_pretraining_modules(self, last_layer: Optional[int] = None):
        # Kept for compatibility with loading code.
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                layer for i, layer in enumerate(self.encoder.layers) if i <= last_layer
            )
