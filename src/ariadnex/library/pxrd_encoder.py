# Copyright (c) 2026 The AriadneX authors.
#
# _init_bert_params, SelfMultiheadAttention, TransformerEncoderLayer and
# TransformerEncoder are adapted from Uni-Core
# (https://github.com/dptech-corp/Uni-Core):
#   Copyright (c) DP Technology.
#   Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""Inference implementation of our PXRD-patch transformer encoder.

This module retains the absolute-position, GELU path used by
``models/last.ckpt`` and its ``pxrd_encoder.`` parameter names.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _base_architecture(args: Any) -> None:
    """Fill only the constructor defaults used by the bundled checkpoint."""

    defaults = {
        "encoder_layers": 8,
        "encoder_embed_dim": 512,
        "encoder_ffn_embed_dim": 2048,
        "encoder_attention_heads": 64,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "max_seq_len": 2048,
        "post_ln": False,
    }
    for name, default in defaults.items():
        setattr(args, name, getattr(args, name, default))


def _init_bert_params(module: nn.Module) -> None:
    """Match the source BERT initialization for freshly constructed models."""

    if not getattr(module, "can_global_init", True):
        return

    def normal_(data: Tensor) -> None:
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class SelfMultiheadAttention(nn.Module):
    """Unmasked self-attention with the checkpoint's ``in_proj`` names."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5
        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, target_length, embed_dim = x.size()
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"attention input dim mismatch: got {embed_dim}, expected {self.embed_dim}"
            )
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = (
            q.view(batch_size, target_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        k = (
            k.view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.head_dim)
        )
        v = (
            v.view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.head_dim)
        )
        scores = torch.bmm(q, k.transpose(1, 2))
        attention = F.dropout(
            F.softmax(scores, dim=-1),
            p=self.dropout,
            training=self.training,
        )
        output = torch.bmm(attention, v)
        output = (
            output.view(batch_size, self.num_heads, target_length, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, target_length, embed_dim)
        )
        return self.out_proj(output)


class TransformerEncoderLayer(nn.Module):
    """Pre-normalized GELU Transformer layer used in ``last.ckpt``."""

    def __init__(
        self,
        *,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.dropout = float(dropout)
        self.activation_dropout = float(activation_dropout)
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            dropout=self.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, int(ffn_embed_dim))
        self.fc2 = nn.Linear(int(ffn_embed_dim), self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return residual + x


class TransformerEncoder(nn.Module):
    """Absolute-position Transformer stack with no mask or bias branches."""

    def __init__(
        self,
        *,
        encoder_layers: int,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        emb_dropout: float,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
    ) -> None:
        super().__init__()
        self.emb_dropout = float(emb_dropout)
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=int(ffn_embed_dim),
                    attention_heads=int(attention_heads),
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                    activation_dropout=float(activation_dropout),
                )
                for _ in range(int(encoder_layers))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer_norm(x)


class PXRDPatchTransformer(nn.Module):
    """Dense PXRD-patch transformer encoder with ``cls_token`` output."""

    input_type = "dense"
    is_pxrd_patch_transformer = True

    def __init__(
        self,
        args: Any = None,
        max_seq_len: int = 2048,
        patch_size: int = 16,
        signal_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if args is None:
            args = SimpleNamespace()
        for key, value in kwargs.items():
            setattr(args, key, value)
        args.max_seq_len = getattr(args, "max_seq_len", max_seq_len)
        _base_architecture(args)
        self.args = args

        self.patch_size = int(getattr(args, "patch_size", patch_size))
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        configured_signal_length = getattr(args, "signal_length", signal_length)
        if configured_signal_length in (None, "", "null", "None"):
            self.signal_length = None
        else:
            self.signal_length = int(configured_signal_length)
            if self.signal_length <= 0:
                raise ValueError("signal_length must be > 0 when provided")

        activation_fn = str(getattr(args, "activation_fn", "gelu")).strip().lower()
        if activation_fn != "gelu":
            raise ValueError("The bundled encoder supports activation_fn='gelu' only")
        position_encoding_type = (
            str(getattr(args, "position_encoding_type", "absolute")).strip().lower()
        )
        if position_encoding_type != "absolute":
            raise ValueError("The bundled encoder supports position_encoding_type='absolute' only")
        if bool(args.post_ln):
            raise ValueError("The bundled checkpoint uses pre-layer normalization only")

        embed_dim = int(args.encoder_embed_dim)
        self.embed_dim = embed_dim
        self.embed_patches = nn.Sequential(
            nn.Linear(self.patch_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.vnode_encoder = nn.Embedding(1, embed_dim)
        self.embed_positions = nn.Embedding(int(args.max_seq_len), embed_dim)
        self.sentence_encoder = TransformerEncoder(
            encoder_layers=int(args.encoder_layers),
            embed_dim=embed_dim,
            ffn_embed_dim=int(args.encoder_ffn_embed_dim),
            attention_heads=int(args.encoder_attention_heads),
            emb_dropout=float(args.emb_dropout),
            dropout=float(args.dropout),
            attention_dropout=float(args.attention_dropout),
            activation_dropout=float(args.activation_dropout),
        )
        self.apply(_init_bert_params)

    @staticmethod
    def _ensure_dense(values: Tensor) -> Tensor:
        if values.ndim == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)
        if values.ndim != 2:
            raise ValueError("dense PXRD tensors must be [B,L] or [B,L,1]")
        return values

    def forward(
        self,
        pxrd_y_dense: Tensor,
        length: Optional[int] = None,
        pxrd_x_dense: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        del pxrd_x_dense  # absolute positions do not consume theta coordinates
        pxrd_y_dense = self._ensure_dense(pxrd_y_dense)
        effective_length = int(pxrd_y_dense.size(1))
        if self.signal_length is not None and effective_length != int(self.signal_length):
            raise ValueError(
                f"pxrd_y_dense length={effective_length} does not match "
                f"configured signal_length={self.signal_length}"
            )
        if length is not None and int(length) != effective_length:
            raise ValueError(
                f"length={length} does not match effective pxrd_y_dense length={effective_length}"
            )
        if effective_length % self.patch_size != 0:
            raise ValueError(
                f"pxrd_y_dense length {effective_length} must be divisible by "
                f"patch_size={self.patch_size}"
            )

        batch_size = int(pxrd_y_dense.size(0))
        token_count = effective_length // self.patch_size
        if token_count > int(self.args.max_seq_len):
            raise ValueError(
                f"num_tokens={token_count} exceeds max_seq_len={self.args.max_seq_len}"
            )
        patches = pxrd_y_dense.reshape(batch_size, token_count, self.patch_size)
        patch_embeddings = self.embed_patches(patches)
        positions = torch.arange(token_count, device=patch_embeddings.device)
        patch_embeddings = patch_embeddings + self.embed_positions(positions).unsqueeze(0)
        cls_token = self.vnode_encoder.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, patch_embeddings], dim=1)
        tokens = tokens.to(self.sentence_encoder.emb_layer_norm.weight.dtype)
        encoded = self.sentence_encoder(tokens)
        return {"cls_token": encoded[:, 0, :], "features": encoded[:, 1:, :]}


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    raise TypeError(f"Unsupported checkpoint mapping object: {type(value)!r}")


def load_pxrd_patch_transformer_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PXRDPatchTransformer, dict[str, Any]]:
    """Load strict ``pxrd_encoder.*`` weights and return an eval model/config."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch versions before the ``weights_only`` argument
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("PXRD-patch transformer checkpoint must contain a mapping")

    hyper_parameters = _plain_mapping(checkpoint.get("hyper_parameters", {}))
    if "pxrd_encoder" not in hyper_parameters:
        raise ValueError("Checkpoint hyper_parameters is missing pxrd_encoder")
    config = _plain_mapping(hyper_parameters["pxrd_encoder"])
    config.pop("_target_", None)
    model = PXRDPatchTransformer(args=SimpleNamespace(), **config)

    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, Mapping):
        raise ValueError("PXRD-patch transformer checkpoint state_dict must be a mapping")
    encoder_state = {
        str(key)[len("pxrd_encoder.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("pxrd_encoder.")
    }
    if not encoder_state:
        raise ValueError("Checkpoint does not contain pxrd_encoder.* weights")
    model.load_state_dict(encoder_state, strict=True)
    model.eval()
    return model, config


__all__ = ["PXRDPatchTransformer", "load_pxrd_patch_transformer_checkpoint"]
