import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt

from models.Transformer.Transformer import Transformer
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CMTA.SpatialCompressionBlock import SpatialCompressionBlock, split_modals_features
import numpy as np


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # queries: (B, L, H, E)
        _, S, _, D = values.shape  # values: (B, S, H, D)
        scale = self.scale or 1. / math.sqrt(E)

        queries_flat = queries.view(B, L, H * E)  # queries_flat: (B, L, H * E)
        keys_flat = keys.view(B, S, H * E).transpose(1, 2)  # keys_flat: (B, H * E, S)
        scores = torch.matmul(queries_flat, keys_flat)  # scores: (B, L, S)

        # Apply mask to scores
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # attn_mask: (B, 1, 1, S)
            scores = scores.masked_fill(attn_mask, float('-inf'))  # scores: (B, L, S)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # A: (B, L, S)

        values_flat = values.view(B, S, H * D)  # values_flat: (B, S, H * D)
        V = torch.matmul(A, values_flat).view(B, L, H, D)  # V: (B, L, H, D)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            q, k, v,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = v + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(q, k, v, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,D)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)  # (B,D,d_model)
        # x: [Batch Variate d_model]
        return self.dropout(x)


class MultiModalFormer(nn.Module):
    def __init__(self, args):
        super(MultiModalFormer, self).__init__()
        self.args = args
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(args.in_len, args.d_model, args.dropout)
        self.split_modals_features = self.initialize_split_modals_features(args.video_dims, args.audio_dims,
                                                                           args.au_dims)
        self.timeAtten = Transformer(args)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=args.dropout), args.d_model, args.n_heads),
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    dropout=args.dropout,
                ) for l in range(args.e_layers)
            ],
            # conv_layers=[ConvLayer(self.args.d_model) for l in range(args.e_layers)],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )

        self.projector = nn.Linear(args.d_model, args.in_len, bias=True)
        self.rppg_mapping = nn.Linear(args.rppg_dims, args.single_modal_dims, bias=True)
        self.au_mapping = nn.Linear(args.au_dims, args.single_modal_dims, bias=True)
        self.video_mapping = nn.Linear(args.video_dims, args.single_modal_dims, bias=True)
        self.audio_mapping = nn.Linear(args.audio_dims, args.single_modal_dims, bias=True)

        self.logit_linear = nn.Linear(self.args.in_len, 1)

    def initialize_split_modals_features(self, x, y, z):
        # Initialize the split_modals_features class
        if x != 0 and y != 0 and z != 0:
            split_block = split_modals_features(a=x, b=y, c=z)
        elif x != 0 and y != 0:
            split_block = split_modals_features(a=x, b=y)
        elif x != 0 and z != 0:
            split_block = split_modals_features(a=x, b=z)
        elif y != 0 and z != 0:
            split_block = split_modals_features(a=y, b=z)
        elif x != 0:
            split_block = split_modals_features(a=x)
        elif y != 0:
            split_block = split_modals_features(b=y)
        elif z != 0:
            split_block = split_modals_features(c=z)
        else:
            raise ValueError("At least one non-zero value is required.")
        return split_block

    def modals_norm(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()  # (B,1,single_modal_dims)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # (B,1,single_modal_dims)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def modals_attention(self, q, k, v, v_means, v_stdev):
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        embedding_q = self.enc_embedding(q)
        embedding_k = self.enc_embedding(k)
        embedding_v = self.enc_embedding(v)
        enc_out, attns = self.encoder(embedding_q, embedding_k, embedding_v)
        enc_out = self.projector(enc_out).permute(0, 2, 1)
        enc_out = enc_out * (v_stdev[:, 0, :].unsqueeze(1)).repeat(1, self.args.in_len, 1)
        enc_out = enc_out + (v_means[:, 0, :].unsqueeze(1)).repeat(1, self.args.in_len, 1)
        return enc_out, attns

    def modals_fusion(self, x=None, y=None, z=None):
        if (self.args.num_modals == 3):
            v_features_norm, v_features_means, v_features_stev = self.modals_norm(x)
            a_features_norm, a_features_means, a_features_stev = self.modals_norm(y)
            r_features_norm, r_features_means, r_features_stev = self.modals_norm(z)
            v_attn_features = self.modals_attention(q=a_features_norm, k=r_features_norm, v=v_features_norm,
                                                    v_means=v_features_means, v_stdev=v_features_stev)
            a_attn_features = self.modals_attention(q=v_features_norm, k=r_features_norm, v=a_features_norm,
                                                    v_means=a_features_means, v_stdev=a_features_stev)
            r_attn_features = self.modals_attention(q=v_features_norm, k=a_features_norm, v=r_features_norm,
                                                    v_means=r_features_means, v_stdev=r_features_stev)
            result = torch.cat([v_attn_features[0], a_attn_features[0], r_attn_features[0]], dim=2)
        elif (self.args.num_modals == 2):
            if (x is None):
                x = y
                y = z
            elif (y is None):
                y = z
            x_features_norm, x_features_means, x_features_stev = self.modals_norm(x)
            y_features_norm, y_features_means, y_features_stev = self.modals_norm(y)
            x_attn_features = self.modals_attention(q=y_features_norm, k=y_features_norm, v=x_features_norm,
                                                    v_means=x_features_means, v_stdev=x_features_stev)
            # x_attn_features += self.timeAtten(x)
            y_attn_features = self.modals_attention(q=x_features_norm, k=x_features_norm, v=y_features_norm,
                                                    v_means=y_features_means, v_stdev=y_features_stev)
            # y_attn_features += self.timeAtten(y)
            result = torch.cat([x_attn_features[0], y_attn_features[0]], dim=2)

        else:
            list = [x, y, z]
            x = [elem for elem in list if elem is not None][0]

            result = x
        return result

    def modals_forecast(self, v_features=None, a_features=None, r_features=None):
        features_attn = self.modals_fusion(x=v_features, y=a_features, z=r_features)

        return features_attn

    def forward(self, x):
        x_features, y_features, z_features = self.split_modals_features(x)
        if (x_features is not None and self.args.audio and self.args.num_modals > 1):
            x_features = self.audio_mapping(x_features)
        if (y_features is not None and self.args.au and self.args.num_modals > 1):
            y_features = self.au_mapping(y_features)
        # x = x.unsqueeze(0).permute(0, 2, 1)
        if (self.args.use_align):
            x = self.modals_forecast(x_features, y_features, z_features)
        if (self.args.use_transformer):
            x, attn = self.timeAtten(x)
        logit = self.logit_linear(x.mean(dim=2))
        return logit
