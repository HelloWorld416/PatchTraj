import torch
import math
from torch import layer_norm, nn

from .transformer_model import *
from .layers import MLP, MoELayer, PositionalEncoding, ConcatSquashLinear, social_transformer
from .patch_embed import CausalPatchEmbed, DynamicPatching, DynamicPatchingMoE, DynamicPatchingMoEWithFPN


def generate_causal_mask(num_patches):
    """生成下三角掩码矩阵"""
    return torch.tril(torch.ones(num_patches, num_patches)).unsqueeze(0).bool()


class PatchFormer(nn.Module):
    def __init__(self,
                 input_feats,
                 obs_len=8,
                 pred_len=12,
                 patch_size=4,
                 patch_list=None,
                 stride=2,
                 num_frames=20,
                 latent_dim=512,
                 patch_embed=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 num_experts=8,
                 dropout=0.2,
                 activation="gelu",
                 num_sample=20,
                 output_dim=2,
                 dynamic_patch=False,
                 fusion_mode="bidirectional",
                 patch_stride_mode="nonoverlap",
                 **kargs):
        super().__init__()
        self.input_feats = input_feats
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.time_embed_dim = latent_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.fusion_mode = fusion_mode
        self.patch_stride_mode = patch_stride_mode
        self.supported_fusion_modes = {
            "concat",
            "freq_to_time",
            "time_to_freq",
            "bidirectional",
        }
        if self.fusion_mode not in self.supported_fusion_modes:
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode}. "
                f"Choose from {sorted(self.supported_fusion_modes)}."
            )

        self.patch_size = patch_size
        self.patch_list = patch_list
        self.stride = stride
        self.patch_embed = patch_embed
        self.num_patches = (num_frames - patch_size) // stride + 1

        self.dynamic_patch = dynamic_patch

        # Input Embedding
        self.time_cond_embed = nn.Linear(self.input_feats * self.obs_len, self.time_embed_dim)
        if self.dynamic_patch == True:
            # self.patch_embed = DynamicPatching(self.input_feats, self.obs_len, self.patch_embed, self.patch_list)
            # self.patch_embed = DynamicPatchingMoE(self.input_feats, self.obs_len, self.patch_embed, self.patch_list, self.num_experts)
            self.time_patch_embed = DynamicPatchingMoEWithFPN(
                self.input_feats,
                self.obs_len,
                self.patch_embed,
                self.patch_list,
                self.num_experts,
                patch_stride_mode=self.patch_stride_mode,
            )
            self.freq_patch_embed = DynamicPatchingMoEWithFPN(
                self.input_feats,
                self.num_frames,
                self.patch_embed,
                self.patch_list,
                self.num_experts,
                patch_stride_mode=self.patch_stride_mode,
            )
        else:
            self.patch_embed = CausalPatchEmbed(self.obs_len, self.patch_size, self.stride, self.input_feats, self.patch_embed)

        # positional embedding
        # whether to use dynamic_patch mechanism
        if self.dynamic_patch == True:
            min_patch = min(patch_list)
            self.pred_num_patches = math.ceil(self.pred_len / min_patch)
            self.time_obs_pos_embed = nn.Parameter(torch.randn(self.time_patch_embed.num_patches[min_patch], latent_dim))
            self.freq_obs_pos_embed = nn.Parameter(torch.randn(self.freq_patch_embed.num_patches[min_patch], latent_dim))
            self.pred_pos_embed = nn.Parameter(torch.randn(self.pred_num_patches, latent_dim))
            self.decoder_tokens = nn.Parameter(
                torch.randn(self.pred_num_patches, latent_dim))
            self.output_dim = 2 * min_patch
        else:
            self.pred_num_patches = math.ceil(self.pred_len / self.patch_size)
            self.obs_pos_embed = nn.Parameter(torch.randn(self.obs_len // self.patch_size, latent_dim))
            self.pred_pos_embed = nn.Parameter(torch.randn(self.pred_num_patches, latent_dim))
            self.decoder_tokens = nn.Parameter(torch.randn(self.pred_num_patches, latent_dim))
            self.output_dim = 2 * self.patch_size

        # cross modal fusion
        self.cross_attn_time2freq = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.cross_attn_freq2time = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

        # prepare encoder
        self.temporal_encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_encoder_blocks.append(
                TransformerEncoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )
        # prepare decoder
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.num_sample * self.output_dim))
        # self.dctout = zero_module(nn.Linear(6 * self.latent_dim, self.num_frames * self.input_feats))

        # # define mamba denosing module
        # self.mamba_block1 = Block(dim=2 * context_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
        # self.mamba_block2 = Block(dim=2 * context_dim, mixer_cls=Mamba, norm_cls=RMSNorm, fused_add_norm=True)
        # self.mamba_act = nn.SiLU()

    def forward(self, time_traj, freq_traj, exp='train', mod=None, predict_dct=True):
        """
        time_traj: [N, T, D]
        agent_num_list: [[start1, end1], [start2, end2], ...]
        """
        B, T = time_traj.shape[:2]

        if mod is not None:
            time_mod_emb = self.time_cond_embed(time_traj.reshape(B, -1))

        # time branch
        time_patch_emb, t_moe_outputs = self.time_patch_embed(time_traj)
        time_embed = time_patch_emb + self.time_obs_pos_embed.unsqueeze(0)

        # frequency branch
        freq_patch_emb, f_moe_outputs = self.freq_patch_embed(freq_traj)
        freq_embed = freq_patch_emb + self.freq_obs_pos_embed.unsqueeze(0)

        # time-frequency cross modal fusion
        fused_time = time_embed
        fused_freq = freq_embed
        if self.fusion_mode in ("freq_to_time", "bidirectional"):
            time_enhanced, _ = self.cross_attn_time2freq(
                query=time_embed,
                key=freq_embed,
                value=freq_embed,
                need_weights=False,
            )
            fused_time = time_embed + time_enhanced  # [B, num_time_patches, D]
        if self.fusion_mode in ("time_to_freq", "bidirectional"):
            freq_enhanced, _ = self.cross_attn_freq2time(
                query=freq_embed,
                key=time_embed,
                value=time_embed,
                need_weights=False,
            )
            fused_freq = freq_embed + freq_enhanced  # [B, num_freq_patches, D]

        # concatenate time&freq features
        encoder_input = torch.cat([fused_time, fused_freq], dim=1)  # [B, num_time+num_freq, D]
        mask = generate_causal_mask(encoder_input.shape[1]).to(encoder_input.device)

        # transformer encoder layer
        enc_list = []
        encoder_out = encoder_input
        for i, module in enumerate(self.temporal_encoder_blocks):
            if i < (self.num_layers // 2):
                enc_list.append(encoder_out)
                encoder_out = module(encoder_out, mask, time_mod_emb)
            elif i >= (self.num_layers // 2):
                encoder_out = module(encoder_out, mask, time_mod_emb)
                encoder_out += enc_list[-1]
                enc_list.pop()

        # transformer decoder layer
        dec_list = []
        decoder_input = self.decoder_tokens.expand(B, -1, -1) + self.pred_pos_embed.unsqueeze(0)
        decoder_out = decoder_input
        for j, module in enumerate(self.temporal_decoder_blocks):
            if j < (self.num_layers // 2):
                dec_list.append(decoder_out)
                decoder_out = module(decoder_out, encoder_out, emb=time_mod_emb)
            elif j >= (self.num_layers // 2):
                decoder_out = module(decoder_out, encoder_out, emb=time_mod_emb)
                decoder_out += dec_list[-1]
                dec_list.pop()

        # output predicted trajectory
        pred = self.out(decoder_out).view(B, self.num_sample, -1, 2)[:, :, :self.pred_len].contiguous()
        # if predict_dct:
        #     dct_pred = self.dctout(decoder_out.reshape(B, -1)).view(B, self.num_frames, -1).contiguous()

        moe_outputs = {'t': t_moe_outputs,
                         'f': f_moe_outputs}

        # return pred, dct_pred, moe_outputs
        return pred, moe_outputs
