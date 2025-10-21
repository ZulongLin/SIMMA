import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=True):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        chunk_size = 128

        if L <= chunk_size:
            scale = self.scale or 1. / sqrt(E)
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)

            attn = scale * scores
            attn = self.dropout(torch.softmax(attn, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", attn, values)
        else:
            scale = self.scale or 1. / sqrt(E)

            if self.mask_flag and attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            V_chunks = []
            attn_chunks = []

            for i in range(0, L, chunk_size):
                queries_chunk = queries[:, i:i + chunk_size, :, :]
                scores_chunk = torch.einsum("blhe,bshe->bhls", queries_chunk, keys)

                if self.mask_flag and attn_mask is not None:
                    mask_chunk = attn_mask.mask[:, :, i:i + chunk_size, :]
                    scores_chunk.masked_fill_(mask_chunk, -np.inf)

                attn_chunk = scale * scores_chunk
                attn_chunk = self.dropout(torch.softmax(attn_chunk, dim=-1))

                V_chunk = torch.einsum("bhls,bshd->blhd", attn_chunk, values)

                V_chunks.append(V_chunk)
                if self.output_attention:
                    attn_chunks.append(attn_chunk)

            V = torch.cat(V_chunks, dim=1)
            attn = torch.cat(attn_chunks, dim=2) if self.output_attention else None

        if self.output_attention:
            return (V.contiguous(), attn)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, atten = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), atten
