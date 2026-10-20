import os

import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import math
from typing import Optional


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # y = x + self.proj_out(y, emb)
        y = x + y
        return y


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class TemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        # self.norm = nn.LayerNorm(latent_dim)
        self.norm = DyT(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, mask=None, emb=None):
        """
        Args:
            x: input tensor of shape [B, T, D]
            emb: time embedding
            mask: optional mask of shape [B, T] (for padding mask) or [B, T, T] (for causal/pairwise mask)
                  False/0 means mask out (no attention), True/1 means keep
        """
        B, T, D = x.shape
        H = self.num_head

        # Normalize and project to query, key, value
        query = self.query(self.norm(x)).view(B, T, H, -1)
        key = self.key(self.norm(x)).view(B, T, H, -1)
        value = self.value(self.norm(x)).view(B, T, H, -1)

        # Compute attention scores B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # Padding mask [B, T]
                mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1, 1]
            elif mask.dim() == 3:  # Pairwise mask [B, T, T]
                mask = mask.unsqueeze(-1)  # [B, 1, T, T, 1]

            # Use large negative value to mask out positions
            attention = attention.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        weight = self.dropout(F.softmax(attention, dim=2))

        # Apply attention to values
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, encoder_out, mask=None, emb=None):
        """
        Args:
            x: decoder input [B, T, D]
            encoder_out: encoder output [B, S, D]
            tgt_mask: [B, T, T]
            memory_mask: optional mask [B, T, S] for encoder-decoder attention
        """
        B, T, D = x.shape
        _, S, _ = encoder_out.shape
        H = self.num_head

        # Project queries (from decoder)
        query = self.query(self.norm(x)).view(B, T, H, -1)

        # Project keys/values (from encoder)
        key = self.key(self.norm(encoder_out)).view(B, S, H, -1)
        value = self.value(self.norm(encoder_out)).view(B, S, H, -1)

        # Attention scores [B, T, S, H]
        attention = torch.einsum('bthd,bshd->bhts', query, key) / math.sqrt(D // H)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        weight = self.dropout(F.softmax(attention, dim=2))

        # Apply attention to values
        y = torch.einsum('bhts,bshd->bthd', weight, value).reshape(B, T, D)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, mask=None, emb=None):
        x = self.sa_block(x, mask, emb)
        x = self.ffn(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = TemporalCrossAttention(
            latent_dim, num_head, dropout, time_embed_dim
        )
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, encoder_out, tgt_mask=None, memory_mask=None, emb=None):
        """
        Args:
            x: decoder input [B, T, D]
            encoder_out: encoder output [B, S, D]
            tgt_mask: target mask for self-attention [B, T] or [B, T, T]
            memory_mask: mask for encoder-decoder attention [B, T, S]
        """
        x = self.sa_block(x, tgt_mask, emb)

        x = self.ca_block(x, encoder_out, memory_mask, emb)

        x = self.ffn(x)
        return x
