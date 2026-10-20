import torch
from torch import nn
import torch.nn.functional as F

from .transformer_model import TransformerEncoderLayer


def generate_causal_mask(num_patches, device):
    return torch.tril(torch.ones(num_patches, num_patches, device=device)).unsqueeze(0).bool()


class TrajectoryHead(nn.Module):
    def __init__(self, latent_dim, pred_len, num_sample):
        super().__init__()
        self.pred_len = pred_len
        self.num_sample = num_sample
        self.proj = nn.Linear(latent_dim, num_sample * pred_len * 2)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.proj(x).view(x.shape[0], self.num_sample, self.pred_len, 2).contiguous()


class PatchTSTTrajectory(nn.Module):
    def __init__(
            self,
            input_feats,
            obs_len=8,
            pred_len=12,
            patch_size=2,
            stride=2,
            latent_dim=256,
            ff_size=1024,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            num_sample=20,
            **kwargs):
        super().__init__()
        self.input_feats = input_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (obs_len - patch_size) // stride + 1
        if self.num_patches <= 0:
            raise ValueError(f'patch_size={patch_size} is larger than obs_len={obs_len}.')

        self.patch_proj = nn.Linear(input_feats * patch_size, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(self.num_patches, latent_dim))
        self.time_cond_embed = nn.Linear(input_feats * obs_len, latent_dim)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                latent_dim=latent_dim,
                time_embed_dim=latent_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.head = TrajectoryHead(latent_dim, pred_len, num_sample)

    def forward(self, time_traj, freq_traj=None, exp='train', mod=None, predict_dct=True):
        bsz = time_traj.shape[0]
        patches = time_traj.unfold(1, self.patch_size, self.stride).permute(0, 1, 3, 2)
        x = self.patch_proj(patches.reshape(bsz, self.num_patches, -1))
        x = x + self.pos_embed.unsqueeze(0)
        emb = self.time_cond_embed(time_traj.reshape(bsz, -1))
        mask = generate_causal_mask(x.shape[1], x.device)
        for block in self.encoder:
            x = block(x, mask, emb)
        return self.head(x), {}


class InceptionBlock1D(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel, padding=kernel // 2)
            for kernel in (1, 3, 5)
        ])
        self.out = nn.Conv1d(hidden_channels * len(self.branches), channels, kernel_size=1)

    def forward(self, x):
        y = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.out(F.gelu(y))


class TimesBlock(nn.Module):
    def __init__(self, seq_len, channels, hidden_channels, top_k=2):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.period_convs = nn.ModuleList([
            InceptionBlock1D(channels, hidden_channels)
            for _ in range(top_k)
        ])

    def forward(self, x):
        # Explicit DFT period scoring avoids cuFFT instability on short
        # trajectory sequences in some CUDA/PyTorch combinations.
        seq_len = x.shape[1]
        freq_count = seq_len // 2 + 1
        time_idx = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freq_idx = torch.arange(freq_count, device=x.device, dtype=x.dtype)
        angles = 2 * torch.pi * time_idx[:, None] * freq_idx[None, :] / seq_len
        detached = x.detach()
        real = torch.einsum('bld,lf->bfd', detached, torch.cos(angles))
        imag = -torch.einsum('bld,lf->bfd', detached, torch.sin(angles))
        amplitude = torch.sqrt(real.square() + imag.square()).mean(dim=(0, 2))
        amplitude[0] = 0
        top_k = min(self.top_k, max(1, amplitude.shape[0] - 1))
        _, indices = torch.topk(amplitude, top_k)

        outputs = []
        x_c = x.transpose(1, 2)
        for conv in self.period_convs[:top_k]:
            outputs.append(conv(x_c).transpose(1, 2))

        weights = amplitude[indices].softmax(dim=0).view(1, top_k, 1, 1)
        stacked = torch.stack(outputs, dim=1)
        return x + (stacked * weights).sum(dim=1)


class TimesNetTrajectory(nn.Module):
    def __init__(
            self,
            input_feats,
            obs_len=8,
            pred_len=12,
            latent_dim=256,
            ff_size=1024,
            num_layers=2,
            dropout=0.1,
            num_sample=20,
            top_k=2,
            **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_feats, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(obs_len, latent_dim))
        self.blocks = nn.ModuleList([
            TimesBlock(obs_len, latent_dim, ff_size, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.head = TrajectoryHead(latent_dim, pred_len, num_sample)

    def forward(self, time_traj, freq_traj=None, exp='train', mod=None, predict_dct=True):
        x = self.input_proj(time_traj) + self.pos_embed.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x), {}


class FourierBlock(nn.Module):
    def __init__(self, latent_dim, modes=4):
        super().__init__()
        self.modes = modes
        self.proj_real = nn.Linear(latent_dim, latent_dim)
        self.proj_imag = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # Explicit low-frequency DFT avoids cuFFT failures observed on short
        # trajectory sequences while keeping gradients through the projections.
        seq_len = x.shape[1]
        modes = min(self.modes, seq_len // 2 + 1)
        time_idx = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freq_idx = torch.arange(modes, device=x.device, dtype=x.dtype)
        angles = 2 * torch.pi * time_idx[:, None] * freq_idx[None, :] / seq_len
        cos_basis = torch.cos(angles)
        sin_basis = torch.sin(angles)

        real = torch.einsum('bld,lm->bmd', x, cos_basis)
        imag = -torch.einsum('bld,lm->bmd', x, sin_basis)
        real = self.proj_real(real)
        imag = self.proj_imag(imag)

        weights = torch.ones(modes, device=x.device, dtype=x.dtype)
        if modes > 1:
            weights[1:] = 2
        if seq_len % 2 == 0 and modes == seq_len // 2 + 1:
            weights[-1] = 1
        y = (
            torch.einsum('bmd,lm,m->bld', real, cos_basis, weights)
            - torch.einsum('bmd,lm,m->bld', imag, sin_basis, weights)
        ) / seq_len
        return self.norm(x + y)


class FEDformerTrajectory(nn.Module):
    def __init__(
            self,
            input_feats,
            obs_len=8,
            pred_len=12,
            latent_dim=256,
            ff_size=1024,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            num_sample=20,
            modes=4,
            **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_feats, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(obs_len, latent_dim))
        self.time_cond_embed = nn.Linear(input_feats * obs_len, latent_dim)
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(latent_dim, modes=modes)
            for _ in range(num_layers)
        ])
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                latent_dim=latent_dim,
                time_embed_dim=latent_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.head = TrajectoryHead(latent_dim, pred_len, num_sample)

    def forward(self, time_traj, freq_traj=None, exp='train', mod=None, predict_dct=True):
        bsz = time_traj.shape[0]
        x = self.input_proj(time_traj) + self.pos_embed.unsqueeze(0)
        emb = self.time_cond_embed(time_traj.reshape(bsz, -1))
        mask = generate_causal_mask(x.shape[1], x.device)
        for fourier, encoder in zip(self.fourier_blocks, self.encoder):
            x = fourier(x)
            x = encoder(x, mask, emb)
        return self.head(x), {}
