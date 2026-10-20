import torch
from torch import layer_norm, nn
import torch.nn.functional as F


class CausalPatchEmbed(nn.Module):
    def __init__(self, length=20, patch_size=4, stride=2, in_dim=2, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_dim = in_dim
        self.proj = nn.Linear(patch_size * in_dim, embed_dim)

        # 因果位置编码（每个Patch只能感知自身位置）
        self.pos_embed = nn.Parameter(torch.randn(1, (length - patch_size) // stride + 1, embed_dim))

    def forward(self, x):
        B, T, _ = x.shape
        # 因果分块（必须用unfold实现滑动窗口）
        patches = x.unfold(1, self.patch_size, self.stride)  # patch number: N = [(L - P) / S] + 1
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.reshape(B, -1, self.patch_size * self.in_dim)

        # 投影并添加位置编码
        patch_emb = self.proj(patches)
        patch_emb += self.pos_embed  # 位置编码不跨越未来

        return patch_emb


class DynamicPatching(nn.Module):
    def __init__(self, in_dim: int=2, obs_len: int=8, latent_dim: int=64, patch_list: list=[2, 4, 8]):
        super().__init__()
        self.in_dim = in_dim
        self.obs_len = obs_len
        self.patch_list = patch_list
        # 动态创建不同尺度的线性投影层
        self.patch_ops = nn.ModuleDict({
            f'size{ps}': nn.Linear(in_dim * ps, latent_dim)
            for ps in patch_list
        })

        # 为每种尺度创建独立的位置编码
        self.pos_embeds = nn.ParameterDict({
            f'size{ps}': nn.Parameter(torch.randn(obs_len // ps, latent_dim))
            for ps in patch_list
        })

        # 动态权重生成器（根据输入决定各尺度重要性）
        self.weight_net = nn.Sequential(
            nn.Linear(in_dim * obs_len, latent_dim),  # 输入为展平的观测序列
            nn.ReLU(),
            nn.Linear(latent_dim, len(patch_list)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: input trajs [B, 8, 2] (obs part)
        Returns:
            fused_embed: 融合的多尺度Patch嵌入 [B, num_patches, latent_dim]
        """
        B = x.shape[0]
        patch_embeds = []

        # 处理每种Patch尺寸
        for ps in self.patch_list:
            # 分割序列为Patches [B, 8, 2] -> [B, num_patches, ps*2]
            num_patches = self.obs_len // ps
            # patches = x.unfold(1, ps, ps).flatten(2)
            patches = x.reshape(B, num_patches, ps * x.shape[-1])

            # 投影到隐空间并添加位置编码
            embed = self.patch_ops[f'size{ps}'](patches) + self.pos_embeds[f'size{ps}']
            patch_embeds.append(embed)  # 各尺度形状: [B, num_patches, latent_dim]

        # 动态权重计算（基于全局上下文）
        global_context = x.reshape(B, -1)
        weights = self.weight_net(global_context)  # [B, 3] (3=len(patch_size_list))

        # 对各尺度嵌入进行零填充以保证对齐最大Patch数 (方便加权求和)
        max_patches = max(self.obs_len // ps for ps in self.patch_list)  # 4 (当ps=2)
        padded_embeds = []
        for embed, ps in zip(patch_embeds, self.patch_list):
            pad_size = max_patches - (self.obs_len // ps)
            if pad_size > 0:
                padded = F.pad(embed, (0, 0, 0, pad_size))  # 在时间维度填充
            else:
                padded = embed
            padded_embeds.append(padded)  # 所有尺度变为 [B, 4, latent_dim]

        # 按权重融合 [B, 4, latent_dim, 3] -> [B, 4, latent_dim]
        stacked = torch.stack(padded_embeds, dim=-1)  # [B, 4, 32, 3]
        weights = weights.view(B, 1, 1, -1)  # [B, 1, 1, 3]
        fused_embed = (stacked * weights).sum(dim=-1)

        return fused_embed


class TopKMoEGate(nn.Module):
    def __init__(self, input_dim, num_experts, num_scales, k=2, temp=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.num_scales = num_scales
        self.k = k
        self.temp = temp
        self.proj = nn.Linear(input_dim, num_experts * num_scales)

    def forward(self, x):
        # x: [B, input_dim]
        logits = self.proj(x).view(x.size(0), self.num_experts, self.num_scales)  # [B, E, S]

        # Gumbel-Topk可微分选择
        if self.training:
            # 训练时使用可微Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
            noisy_logits = (logits + gumbel_noise) / self.temp
            topk_mask = torch.sigmoid(noisy_logits - noisy_logits.topk(self.k, dim=1)[0][:, -1:])
        else:
            # 推理时硬截断
            topk_mask = (logits >= logits.topk(self.k, dim=1)[0][:, -1:]).float()

        # 归一化保持概率总和不变
        sparse_gates = F.softmax(logits, dim=1) * topk_mask
        sparse_gates = sparse_gates / (sparse_gates.sum(dim=1, keepdim=True) + 1e-6)

        return sparse_gates  # [B, E, S]


class DynamicPatchingMoE(nn.Module):
    def __init__(self, in_dim: int = 2, obs_len: int = 8, latent_dim: int = 64,
                 patch_list: list = [2, 4, 8], num_experts: int = 4, top_k: int=2,
                 mask_mode: str = 'feature', mask_ratio: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.obs_len = obs_len
        self.latent_dim = latent_dim
        self.patch_list = patch_list
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio

        # 1. 多尺度Patch投影层 (每个专家独立参数)
        self.expert_proj = nn.ModuleDict({
            f'size{ps}': nn.ModuleList([
                nn.Linear(in_dim * ps, latent_dim)
                for _ in range(num_experts)
            ]) for ps in patch_list
        })

        # 2. 专家专属的位置编码
        self.pos_embeds = nn.ParameterDict({
            f'size{ps}': nn.ParameterList([
                nn.Parameter(torch.randn(obs_len // ps, latent_dim))
                for _ in range(num_experts)
            ]) for ps in patch_list
        })

        # 3. 门控网络 (动态路由)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim * obs_len, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_experts * len(patch_list)),
            nn.Softmax(dim=-1)
        )
        # 3. topk门控网络 (动态路由)
        # self.gate_net = TopKMoEGate(
        #     input_dim=in_dim * obs_len,
        #     num_experts=num_experts,
        #     num_scales=len(patch_list),
        #     k=top_k
        # )

        # 4. 专家输出融合层
        self.fusion = nn.Linear(latent_dim * num_experts, latent_dim)

    def mask_patches(self, patches, patch_size):
        """
        对已划分的patches进行mask
        Args:
            patches: [B, num_patches, patch_dim]
            patch_size: 当前patch的时间步长
        Returns:
            masked_patches: [B, num_patches, patch_dim]
        """
        B, num_patches, _ = patches.shape

        if self.mask_mode == 'random':
            # 随机选择要mask的patch索引
            mask = torch.rand(B, num_patches, 1, device=patches.device) < self.mask_ratio
            return patches * ~mask

        elif self.mask_mode == 'temporal':
            # 时序遮挡（遮挡连续时间段）
            mask_width = max(1, int(patch_size * 0.5))  # 遮挡patch内部分时间步
            mask = torch.zeros(B, num_patches, device=patches.device).bool()
            for b in range(B):
                start = torch.randint(0, num_patches - mask_width + 1, (1,))
                mask[b, start:start + mask_width] = True
            return patches * ~mask.unsqueeze(-1)

        elif self.mask_mode == 'feature':
            # 按特征维度遮挡
            feat_mask = torch.rand(patches.size(-1), device=patches.device) < 0.2
            return patches * ~feat_mask.view(1, 1, -1)

        return patches

    def forward(self, x):
        """
         Args:
             x: input trajs [B, T, D] (obs part)
         Returns:
             fused_embed: 融合的多尺度Patch嵌入 [B, num_patches, latent_dim]
         """
        B = x.shape[0]
        max_patches = self.obs_len // min(self.patch_list)  # 计算最大patch数

        # 计算门控权重 [B, num_experts * num_scales]
        global_context = x.reshape(B, -1)
        gates = self.gate_net(global_context).view(B, -1, len(self.patch_list))  # [B, num_experts, num_scales]

        # # Top-k选择
        # if self.top_k < self.num_experts:
        #     topk_val, topk_idx = gates.topk(self.top_k, dim=1)
        #     sparse_gates = torch.zeros_like(gates).scatter_(1, topk_idx, topk_val)
        #     gates = F.softmax(sparse_gates, dim=1) * (sparse_gates > 0).float()

        expert_outputs = []
        # 每个专家独立处理多尺度特征
        for expert_id in range(self.num_experts):
            scale_features = []
            expert_weight = gates[:, expert_id]  # [B, S]

            for scale_idx, ps in enumerate(self.patch_list):
                # 跳过未被选中的专家-尺度组合
                if expert_weight[:, scale_idx].sum() == 0:
                    continue

                # 分割序列 [B, T, D] -> [B, num_patches, ps*D]
                num_patches = self.obs_len // ps
                patches = x.reshape(B, num_patches, ps * x.shape[-1])
                # if self.training and self.mask_ratio > 0:
                #     patches = self.mask_patches(patches, ps)

                # 当前专家对当前尺度的处理
                proj = self.expert_proj[f'size{ps}'][expert_id]
                pos = self.pos_embeds[f'size{ps}'][expert_id]
                embed = proj(patches) + pos[:num_patches]  # [B, num_patches, latent_dim]

                # 统一填充到最大patch数
                if num_patches < max_patches:
                    pad_size = max_patches - num_patches
                    embed = F.pad(embed, (0, 0, 0, pad_size))  # 在时间维度填充

                # 应用门控权重
                # weight = expert_weight[:, scale_idx].view(B, 1, 1)  # [B, 1, 1]
                weight = gates[:, expert_id, scale_idx].view(B, 1, 1)  # [B, 1, 1]
                scale_features.append(weight * embed)

            # 专家级多尺度融合 (sum pooling)
            expert_out = sum(scale_features) if scale_features else 0 # [B, max_patches, latent_dim]
            expert_outputs.append(expert_out)

        # 处理无专家激活的极端情况
        if len(expert_outputs) == 0:
            expert_outputs.append(torch.zeros(B, max_patches, self.latent_dim, device=x.device))

        # 跨专家特征拼接与融合
        fused = torch.cat(expert_outputs, dim=-1)  # [B, max_patches, latent_dim*num_experts]
        fused_embed = self.fusion(fused)  # [B, max_patches, latent_dim]

        moe_outputs = {
            'gates': gates,
            'experts': expert_outputs
        }

        return fused_embed, moe_outputs


class DynamicPatchingMoEWithFPN(nn.Module):
    def __init__(self, in_dim: int = 2, obs_len: int = 8, latent_dim: int = 64,
                 patch_list: list = [2, 4, 8], num_experts: int = 4, top_k: int = 2,
                 mask_mode: str = 'feature', mask_ratio: float = 0.3,
                 patch_stride_mode: str = 'nonoverlap'):
        super().__init__()
        self.in_dim = in_dim
        self.obs_len = obs_len
        self.latent_dim = latent_dim
        self.patch_list = patch_list
        self.patch_stride_mode = patch_stride_mode
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.num_patches = {ps: self.get_num_patches(ps) for ps in patch_list}

        # 多尺度Patch投影层 (每个专家独立参数)
        self.expert_proj = nn.ModuleDict({
            f'size{ps}': nn.ModuleList([
                nn.Linear(in_dim * ps, latent_dim)
                for _ in range(num_experts)
            ]) for ps in patch_list
        })

        # 专家专属的位置编码
        self.pos_embeds = nn.ParameterDict({
            f'size{ps}': nn.ParameterList([
                nn.Parameter(torch.randn(self.num_patches[ps], latent_dim))
                for _ in range(num_experts)
            ]) for ps in patch_list
        })

        # 门控网络 (动态路由)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim * obs_len, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_experts * len(patch_list)),
            nn.Softmax(dim=-1)
        )

        # # 专家输出融合层
        # self.fusion = nn.Linear(latent_dim * num_experts, latent_dim)

        # 使用FPN融合多尺度嵌入特征
        self.fpn_lateral = nn.ModuleDict({
            f'size{ps}': nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
            for ps in patch_list
        })

        self.fpn_output = nn.ModuleDict({
            f'size{ps}': nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
            for ps in patch_list
        })

        # 上采样和下采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)

    def get_patch_stride(self, patch_size):
        if self.patch_stride_mode == 'nonoverlap':
            return patch_size
        if self.patch_stride_mode == 'overlap':
            return max(1, patch_size // 2)
        raise ValueError(f'Unsupported patch_stride_mode={self.patch_stride_mode}')

    def get_num_patches(self, patch_size):
        if self.patch_stride_mode == 'nonoverlap' and self.obs_len % patch_size != 0:
            raise ValueError(
                f'patch_size={patch_size} must divide obs_len={self.obs_len} '
                'when patch_stride_mode=nonoverlap.'
            )
        stride = self.get_patch_stride(patch_size)
        return (self.obs_len - patch_size) // stride + 1

    def make_patches(self, x, patch_size):
        stride = self.get_patch_stride(patch_size)
        patches = x.unfold(1, patch_size, stride).permute(0, 1, 3, 2)
        return patches.reshape(x.shape[0], self.num_patches[patch_size], patch_size * x.shape[-1])

    def forward(self, x):
        B = x.shape[0]

        # 计算门控权重
        global_context = x.reshape(B, -1)
        gates = self.gate_net(global_context).view(B, -1, len(self.patch_list))

        # 存储各尺度的原始特征
        scale_features = {ps: [] for ps in self.patch_list}

        # 每个专家独立处理多尺度特征
        for expert_id in range(self.num_experts):
            expert_weight = gates[:, expert_id]  # [B, S]

            for scale_idx, ps in enumerate(self.patch_list):
                if expert_weight[:, scale_idx].sum() == 0:
                    continue

                # 分割序列并投影
                num_patches = self.num_patches[ps]
                patches = self.make_patches(x, ps)
                proj = self.expert_proj[f'size{ps}'][expert_id]
                pos = self.pos_embeds[f'size{ps}'][expert_id]
                embed = proj(patches) + pos[:num_patches]  # [B, num_patches, latent_dim]

                # 应用门控权重并存储
                weight = gates[:, expert_id, scale_idx].view(B, 1, 1)
                scale_features[ps].append(weight * embed)

        # 对各尺度特征进行专家聚合
        fpn_features = {}
        for ps in self.patch_list:
            if scale_features[ps]:
                # 聚合当前尺度下所有专家的特征
                aggregated = sum(scale_features[ps]) / len(scale_features[ps])  # [B, num_patches, latent_dim]
                # 转换为Conv1d需要的格式 [B, C, L]
                fpn_features[ps] = aggregated.permute(0, 2, 1)  # [B, latent_dim, num_patches]
            else:
                fpn_features[ps] = torch.zeros(B, self.latent_dim, self.num_patches[ps], device=x.device)

        # 构建FPN金字塔 (从细到粗排序)
        sorted_scales = sorted(self.patch_list, reverse=True)  # 例如 [8,4,2]

        # 初始化FPN处理
        pyramid = {}
        for i, ps in enumerate(sorted_scales):
            # 侧边连接
            lateral = self.fpn_lateral[f'size{ps}'](fpn_features[ps])

            if i == 0:
                # 最精细的尺度直接作为金字塔顶层
                pyramid[ps] = self.fpn_output[f'size{ps}'](lateral)
            else:
                # 上采样前一层的特征并相加
                prev_ps = sorted_scales[i - 1]
                upsampled = self.upsample(pyramid[prev_ps])
                # 调整尺寸可能不匹配的情况
                if upsampled.shape[-1] > lateral.shape[-1]:
                    upsampled = upsampled[..., :lateral.shape[-1]]
                elif upsampled.shape[-1] < lateral.shape[-1]:
                    pad = lateral.shape[-1] - upsampled.shape[-1]
                    upsampled = F.pad(upsampled, (0, pad))

                pyramid[ps] = self.fpn_output[f'size{ps}'](lateral + upsampled)

        # 自顶向下增强特征
        enhanced_features = {}
        for i, ps in enumerate(reversed(sorted_scales)):
            if i == 0:
                # 最粗糙的尺度直接使用金字塔特征
                enhanced_features[ps] = pyramid[ps]
            else:
                # 获取更粗糙尺度的增强特征
                coarser_ps = sorted_scales[len(sorted_scales) - i]  # 前一个(更粗糙的)尺度
                upsampled = self.upsample(enhanced_features[coarser_ps])

                # 调整尺寸
                target_length = pyramid[ps].shape[-1]
                if upsampled.shape[-1] > target_length:
                    upsampled = upsampled[..., :target_length]
                elif upsampled.shape[-1] < target_length:
                    pad = target_length - upsampled.shape[-1]
                    upsampled = F.pad(upsampled, (0, pad))

                # 融合特征
                enhanced_features[ps] = pyramid[ps] + upsampled

        # 选择最精细尺度的增强特征作为输出
        finest_scale = min(self.patch_list)
        output = enhanced_features[finest_scale].permute(0, 2, 1)  # [B, num_patches, latent_dim]

        moe_outputs = {
            'gates': gates,
            'fpn_features': enhanced_features
        }

        return output, moe_outputs
