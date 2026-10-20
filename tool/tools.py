# import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def JRDBdata_process(cfg, data):
    """
    :param data: input one batch data
    :return: traj_mask, past_traj, fut_traj
    """

    traj_mean = traj_mean = torch.FloatTensor([0, 0]).cuda().unsqueeze(0).unsqueeze(0)
    jrdb_scale = cfg.data_scale
    peds_stats = data['peds_num_per_scene'].long()
    past_traj_list = []
    fut_traj_list = []
    pad_traj_list = []
    init_pos_list = []

    traj_mask = torch.zeros(sum(peds_stats), sum(peds_stats)).cuda()

    for i, peds_num in enumerate(peds_stats):
        srt_idx = sum(peds_stats[:i])
        end_idx = sum(peds_stats[:i + 1])

        batch_past_traj = data['pre_motion_3D'][srt_idx:end_idx, :, :].cuda()
        batch_fut_traj = data['fut_motion_3D'][srt_idx:end_idx, :, :].cuda()
        initial_pos = batch_past_traj[:, -1:]
        init_pos_list.append(initial_pos)
        traj = torch.cat((batch_past_traj, batch_fut_traj), dim=1)

        # augment input: absolute position, relative position, velocity
        batch_past_traj_abs = (traj - traj_mean) / jrdb_scale
        batch_past_traj_rel = (traj - initial_pos) / jrdb_scale
        batch_past_traj_vel = torch.cat((
            batch_past_traj_rel[:, 1:] - batch_past_traj_rel[:, :-1], torch.zeros_like(batch_past_traj_rel[:, -1:])),
            dim=1)

        aug_past_traj = torch.cat((batch_past_traj_abs, batch_past_traj_rel, batch_past_traj_vel), dim=-1)
        pad_traj = padding_traj(aug_past_traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        pad_traj_list.append(pad_traj)
        past_traj_list.append(aug_past_traj)
        fut_traj_list.append((batch_fut_traj - initial_pos) / jrdb_scale)

    past_traj = torch.cat(past_traj_list, dim=0)
    pad_traj = torch.cat(pad_traj_list, dim=0)
    fut_traj = torch.cat(fut_traj_list, dim=0)
    init_pos = torch.cat(init_pos_list, dim=0)
    past_traj = past_traj[:, :cfg.obs_len]

    return past_traj, pad_traj, fut_traj, traj_mask, init_pos


def NBAdata_process(cfg, data):
    """
    :param data: input one batch data
    :return: traj_mask, past_traj, fut_traj
    """
    # data = data.values()
    traj_mean = [14, 7.5]
    traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    nba_scale = cfg.data_scale
    batch_size = data['pre_motion_3D'].shape[0]

    trajs = torch.cat((data['pre_motion_3D'], data['fut_motion_3D']), dim=2)
    trajs = trajs.type(torch.float).cuda()
    traj_mask = torch.zeros(batch_size * 11, batch_size * 11).cuda()
    for i in range(batch_size):
        traj_mask[i * 11:(i + 1) * 11, i * 11:(i + 1) * 11] = 1

    initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:]
    # augment input: absolute position, relative position, velocity
    past_traj_abs = ((trajs - traj_mean) / nba_scale).view(-1, cfg.obs_len+cfg.pred_len, 2)
    past_traj_rel = ((trajs - initial_pos) / nba_scale).view(-1, cfg.obs_len+cfg.pred_len, 2)
    past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])),
                              dim=1)
    traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
    pad_traj = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index).view(-1, cfg.obs_len+cfg.pred_len, 6)
    fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos) / nba_scale).view(-1, cfg.pred_len, 2)
    past_traj = traj[:, :cfg.obs_len]

    return past_traj, pad_traj, fut_traj, traj_mask, initial_pos

def ETHdata_process(cfg, data):
    traj_mean = [7.63, 6.95]  # 根据实际load的训练集计算出所有轨迹点x,y的均值, eth: [7.44, 6.66]; hotel: [7.63, 6.95]; univ: [6.09, 5.15]; zara1: [7.47, 6.76]; zara2: [7.67, 6.80]
    traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0)
    eth_scale = cfg.data_scale
    agent_num_list = data['seq_start_end']
    bs_times_num = data['obs_traj'].shape[0]

    traj_mask = torch.zeros(bs_times_num, bs_times_num).cuda()

    for i, element in enumerate(agent_num_list):
        traj_mask[element[0]:element[1], element[0]:element[1]] = 1

    initial_pos = data['obs_traj'].cuda()[:, -1:]
    total_traj = torch.cat((data['obs_traj'], data['pred_traj']), dim=1)

    # augment input: absolute position, relative position, velocity
    traj_abs = (total_traj.cuda() - traj_mean) / eth_scale
    traj_rel = (total_traj.cuda() - initial_pos) / eth_scale
    traj_vel = torch.cat((traj_rel[:, 1:] - traj_rel[:, :-1], torch.zeros_like(traj_rel[:, -1:])), dim=1)
    traj = torch.cat((traj_abs, traj_rel, traj_vel), dim=-1)
    fut_traj = (data['pred_traj'].cuda() - initial_pos) / eth_scale
    pad_traj = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
    past_traj = traj[:, :cfg.obs_len]

    return past_traj, pad_traj, initial_pos, fut_traj, agent_num_list


def SDDdata_process(cfg, data, mode='train'):
    if mode == 'train':
        traj_mean = [700, 842]  # 根据实际load的训练集计算出所有轨迹点x,y的均值
    elif mode == 'test':
        traj_mean = [783, 914]  # 根据实际load的测试集计算出所有轨迹点x,y的均值
    traj_mean = torch.FloatTensor(traj_mean).cuda().unsqueeze(0).unsqueeze(0)

    past_traj = data['past_traj'].type(torch.float).cuda()
    fut_traj = data['fut_traj'].type(torch.float).cuda()

    traj = torch.cat((past_traj, fut_traj), dim=1)

    batch_size = past_traj.shape[0]
    traj_mask = torch.zeros(batch_size, batch_size).cuda()
    for i in range(batch_size):
        traj_mask[i:(i + 1), i:(i + 1)] = 1

    initial_pos_ = past_traj[:, -1:]
    # augment input: absolute position, relative position, velocity
    past_traj_abs = ((traj - traj_mean) / cfg.data_scale)
    past_traj_rel = ((traj - initial_pos_) / cfg.data_scale)
    past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])),
                              dim=1)
    traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
    fut_traj = (fut_traj - initial_pos_) / cfg.data_scale
    pad_traj = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
    past_traj = traj[:, :cfg.obs_len]

    return past_traj, pad_traj, fut_traj, traj_mask, initial_pos_


def noise_motion(x, t, noise_steps=1000):
    betas = make_beta_schedule(schedule='cosine', n_timesteps=noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    sqrt_alpha_hat = torch.sqrt(alphas_prod[t])[:, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_prod[t])[:, None, None]
    z = torch.randn_like(x)

    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * z, z


def reversed_noise(cfg, model, x_t, timesteps, mod_dct_traj):

    t = timesteps
    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    alphas_bar_t_sqrt = extract(alphas_bar_sqrt, t, x_t)
    one_minus_alphas_bar_t_sqrt = extract(one_minus_alphas_bar_sqrt, t, x_t)

    eps_theta = model(x_t, t, mod=mod_dct_traj)
    y = (x_t - one_minus_alphas_bar_t_sqrt * eps_theta) / alphas_bar_t_sqrt

    return y


def sample_timesteps(n, noise_steps=1000):

    return torch.randint(low=1, high=noise_steps, size=(n,))


def make_beta_schedule(schedule: str = 'linear',
                       n_timesteps: int = 1000,
                       start: float = 1e-5, end: float = 1e-2,
                       s: float = 0.008) -> torch.Tensor:
    '''
    Make beta schedule.

    Parameters
    ----
    schedule: ['linear', 'quad', 'sigmoid', 'cosine'],
    n_timesteps: diffusion steps,
    start: beta start, `start<end`,
    end: beta end,

    Returns
    ----
    betas: Tensor with the shape of (n_timesteps)

    '''
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine":
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = n_timesteps + 1
        t = torch.linspace(0, n_timesteps, steps) / n_timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)

    return betas


def condition_mean(cond_fn, mean, var, x, t, label, exp):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.

    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    """
    gradient = cond_fn(x, t, label, exp)
    new_mean = mean + var * gradient

    return new_mean


def uncondition_mean(cond_fn, model, mean, var, x, t, pad_traj):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.

    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    """
    gradient = cond_fn(model, x, t, pad_traj)
    new_mean = mean + var * gradient

    return new_mean


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def mask_complete(cfg, complete_mask, x, t_prev, dct_traj, dct_m, idct_m):

    x_t_prev_known, _ = noise_motion(dct_traj, t_prev)
    x_t_prev_known = idct_transform(x_t_prev_known, idct_m, cfg.n_pre)
    x_t_prev_unknown = idct_transform(x, idct_m, cfg.n_pre)
    x = torch.mul(complete_mask, x_t_prev_known) + torch.mul((1 - complete_mask), x_t_prev_unknown)  # mask
    x = dct_transform(x, dct_m, cfg.n_pre)
    return x


def scale_timesteps(t, num_timesteps):

    return t.float() * (1000.0 / num_timesteps)


def autoregressive_diffusion(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m):
    """
    Implement autoregressive diffusion to generate trajectory point-by-point.
    External diffusion generates initial noise for internal diffusion.
    The function of internal diffusion is to refine the predicted trajectory
    points at timestep t generated by external diffusion.
    :return:
    """

    sample_num = dct_traj.shape[0]

    initial_x = torch.randn(sample_num, 1, 2).cuda()

    generated_seq = []
    for t_pred in reversed(range(cfg.n_pre)):
        x = ddim_loop(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, noise=initial_x)

        # parameter_1 = 0.00001  # scale factor
        # betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
        # alphas = 1 - betas
        # alphas_prod = torch.cumprod(alphas, 0)
        # one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        #
        # t = torch.tensor([t_pred]).cuda()
        # eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
        # # beta = extract(betas, t.repeat(x.shape[0]), dct_traj)
        # eps_theta = model(x, t, mod=mod_dct_traj)
        # mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
        # z = torch.randn_like(x).to(x.device)
        # sigma_t = extract(betas, t, x).sqrt()
        # x = mean + sigma_t * z * parameter_1

        initial_x = x
        generated_seq.append(x)

    return torch.cat(generated_seq, dim=1)


def ddim_loop(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, noise=None):
    # model.eval()
    sample_num = dct_traj.shape[0]
    if noise is not None:
        x = noise
    else:
        x = torch.randn_like(dct_traj).cuda()
        # x = torch.randn((sample_num, cfg.n_pre, 2*3)).cuda()

    # noise_steps // ddim_steps: 用来标记每隔多少步进行采样
    ddim_timestep_seq = np.asarray(list(range(0, cfg.noise_steps, cfg.noise_steps // cfg.ddim_steps))) + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4,
                               end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)

    # with torch.no_grad():
    for i in reversed(range(0, cfg.ddim_steps)):
        t = (torch.ones(sample_num) * ddim_timestep_seq[i]).long().cuda()
        t_prev = (torch.ones(sample_num) * ddim_timestep_prev_seq[i]).long().cuda()

        alpha_t = extract(alphas_prod, t, x)
        alpha_t_prev = extract(alphas_prod, t_prev, x)

        noise_pred = model(x, t, mod=mod_dct_traj)

        x0_pred = (x - torch.sqrt((1. - alpha_t)) * noise_pred) / torch.sqrt(alpha_t)
        x_prev = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred

        x = x_prev

        # x = mask_complete(cfg, complete_mask, x, t_prev, dct_traj, dct_m, idct_m)

    return x


def ddim_sample_iter(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, complete_mask, noise=None):
    model.eval()
    sample_num = dct_traj.shape[0]
    if noise is not None:
        x = noise
    else:
        x = torch.randn_like(dct_traj).cuda()
        # x = torch.randn((sample_num, cfg.obs_len+cfg.pred_len, 2)).cuda()

    # noise_steps // ddim_steps: 用来标记每隔多少步进行采样
    ddim_timestep_seq = np.asarray(list(range(0, cfg.noise_steps, cfg.noise_steps // cfg.ddim_steps))) + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)

    with torch.no_grad():
        for i in reversed(range(0, cfg.ddim_steps)):
            t = (torch.ones(sample_num) * ddim_timestep_seq[i]).long().cuda()
            t_prev = (torch.ones(sample_num) * ddim_timestep_prev_seq[i]).long().cuda()

            alpha_t  = extract(alphas_prod, t, x)
            alpha_t_prev = extract(alphas_prod, t_prev, x)

            noise_pred = model(x, t, mod=mod_dct_traj)

            x0_pred = (x - torch.sqrt((1. - alpha_t)) * noise_pred) / torch.sqrt(alpha_t)
            x_prev = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred

            x = x_prev

            if cfg.mask_complete == True:
                x = mask_complete(cfg, complete_mask, x, t_prev, dct_traj, dct_m, idct_m)

            yield x


def ddim_sample_loop(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, complete_mask):
    '''
    Batch operation to accelerate the denoising process.
    eth data dimension:
    dct_traj: [B, cfg.n_pre, 2]
    agent_mask: [N, N]
    trajs: [B, cfg.num_sample, cfg.n_pre, 2]
    '''

    final = None
    for sample in ddim_sample_iter(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, complete_mask):
        final = sample

    return final


def ddpm_sample_iter(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, complete_mask, noise=None):

    model.eval()
    sample_num = dct_traj.shape[0]
    if noise is not None:
        x = noise
    else:
        # x = torch.randn_like(dct_traj).cuda()
        x = torch.randn((sample_num, cfg.n_pre, 2*3)).cuda()

    parameter_1 = 0.00001  # scale factor
    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    # alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    with torch.no_grad():
        for i in reversed(range(0, cfg.ddpm_steps)):
            # detail denoising producer
            t = torch.tensor([i]).cuda()
            # Factor to the model output, extract(input,t,x)函数的作用就是根据t得到对应t时刻下变量(input)的值
            eps_factor = ((1 - extract(alphas, t, dct_traj)) / extract(one_minus_alphas_bar_sqrt, t, dct_traj))
            # Model output, 这一部分和原论文中计算均值的公式对应
            beta = extract(betas, t.repeat(x.shape[0]), dct_traj)
            eps_theta = model(x, t, mod=mod_dct_traj)
            mean = (1 / extract(alphas, t, dct_traj).sqrt()) * (x - (eps_factor * eps_theta))
            # Generate z
            z = torch.randn_like(dct_traj).to(x.device)
            # Fixed sigma
            sigma_t = extract(betas, t, dct_traj).sqrt()
            # Get timesteps
            # # timesteps = torch.tensor([copy_t] * x.shape[0]).cuda()
            # timesteps = torch.randint(low=1, high=n_steps, size=(x.shape[0],)).cuda()
            # if cond_fn is not None:
            #     new_mean = condition_mean(cond_fn, mean, sigma_t, cur_y, timesteps, label, exp)
            # x = new_mean + sigma_t * z * parameter_1
            x = mean + sigma_t * z * parameter_1

            x = mask_complete(cfg, complete_mask, x, t, dct_traj, dct_m, idct_m)

            yield x


def ddpm_sample_loop(cfg, model, dct_traj, mod_dct_traj, dct_m, idct_m, complete_mask):
    '''
    Batch operation to accelerate the denoising process.
    eth data dimension:
    dct_traj: [B, cfg.n_pre, 2]
    agent_mask: [N, N]
    trajs: [B, cfg.num_sample, cfg.n_pre, 2]
    '''

    final = None
    for sample in ddpm_sample_iter(cfg, model, dct_traj,
                                   mod_dct_traj, dct_m, idct_m, complete_mask):
        final = sample

    return final



def ddpm_uncond_sample_iter(cfg, model, cond_fn, pad_traj, dct_m, idct_m, complete_mask, noise=None):
    model.eval()
    sample_num = pad_traj.shape[0]
    if noise is not None:
        x = noise
    else:
        x = torch.randn_like(pad_traj).cuda()

    parameter_1 = 0.00001  # scale factor
    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    # alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    with torch.no_grad():
        for i in reversed(range(0, cfg.ddpm_steps)):
            # detail denoising producer
            t = torch.tensor([i]).cuda()
            # Factor to the model output, extract(input,t,x)函数的作用就是根据t得到对应t时刻下变量(input)的值
            eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
            # Model output, 这一部分和原论文中计算均值的公式对应
            beta = extract(betas, t.repeat(x.shape[0]), x)
            eps_theta = model(x, t, mod=pad_traj)
            mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
            # Generate z
            z = torch.randn_like(x).to(x.device)
            # Fixed sigma
            sigma_t = extract(betas, t, x).sqrt()
            # Get timesteps
            timesteps = torch.tensor([i] * x.shape[0]).cuda()
            # timesteps = torch.randint(low=1, high=n_steps, size=(x.shape[0],)).cuda()
            if cond_fn is not None:
                new_mean = uncondition_mean(cond_fn, model, mean, sigma_t, x, timesteps, pad_traj)
            x = new_mean + sigma_t * z * parameter_1

            if cfg.mask_complete == True:
                x = mask_complete(cfg, complete_mask, x, t, pad_traj, dct_m, idct_m)

    return x


def ddpm_uncond_sample_loop(cfg, model, cond_fn, pad_traj, dct_m, idct_m, complete_mask):
    '''
    Batch operation to accelerate the denoising process.
    eth data dimension:
    dct_traj: [B, cfg.n_pre, 2]
    agent_mask: [N, N]
    trajs: [B, cfg.num_sample, cfg.n_pre, 2]
    '''
    # timesteps = sample_timesteps(dct_traj.shape[0]).to(dct_traj.device)  # [bs]
    # x_t, _ = noise_motion(dct_traj, timesteps)  # x_t: [bs, 20, 2], noise: [bs, 20, 2]

    output = ddpm_uncond_sample_iter(cfg, model, cond_fn, pad_traj,
                                   dct_m, idct_m, complete_mask)

    return output


def ddim_uncond_sample_iter(cfg, model, cond_fn, pad_traj, dct_m, idct_m, complete_mask, alpha=1.0, noise=None):
    model.eval()
    sample_num = pad_traj.shape[0]
    alpha = cfg.alpha
    if noise is not None:
        x = noise
    else:
        # x = torch.randn_like(dct_traj).cuda()
        x = torch.randn((sample_num, cfg.obs_len+cfg.pred_len, 2)).cuda()

    # noise_steps // ddim_steps: 用来标记每隔多少步进行采样
    ddim_timestep_seq = np.asarray(list(range(0, cfg.noise_steps, cfg.noise_steps // cfg.ddim_steps))) + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

    # make beta schedule and calculate the parameters used in denoising process.
    betas = make_beta_schedule(schedule='cosine', n_timesteps=cfg.noise_steps, start=1e-4, end=5e-2).cuda()  # 标准差
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)

    with torch.no_grad():
        for i in reversed(range(0, cfg.ddim_steps)):
            t = (torch.ones(sample_num) * ddim_timestep_seq[i]).long().cuda()
            t_prev = (torch.ones(sample_num) * ddim_timestep_prev_seq[i]).long().cuda()

            alpha_t  = extract(alphas_prod, t, x)
            alpha_t_prev = extract(alphas_prod, t_prev, x)

            if cond_fn is not None:
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    noise_pred = model(x_in, t, mod=pad_traj)
                    grad = torch.autograd.grad(torch.log(noise_pred), x_in, retain_graph=True, grad_outputs=torch.ones_like(noise_pred))[0] * cfg.scale
                noise_pred = noise_pred - alpha * grad
            else:
                noise_pred = model(x, t, mod=pad_traj)

            x0_pred = (x - torch.sqrt((1. - alpha_t)) * noise_pred) / torch.sqrt(alpha_t)
            x_prev = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred

            x = x_prev

            if cfg.mask_complete == True:
                x = mask_complete(cfg, complete_mask, x, t, pad_traj, dct_m, idct_m)

            yield x


def ddim_uncond_sample_loop(cfg, model, cond_fn, pad_traj, dct_m, idct_m, complete_mask):
    '''
    Batch operation to accelerate the denoising process.
    eth data dimension:
    dct_traj: [B, cfg.n_pre, 2]
    agent_mask: [N, N]
    trajs: [B, cfg.num_sample, cfg.n_pre, 2]
    '''

    final = None
    for sample in ddim_uncond_sample_iter(cfg, model, cond_fn, pad_traj, dct_m, idct_m, complete_mask):
        final = sample

    return final


def print_log(print_str, log, same_line=False, display=True):
    '''
    print a string to a log file

    parameters:
        print_str:          a string to print
        log:                a opened file to save the log
        same_line:          True if we want to print the string without a new next line
        display:            False if we want to disable to print the string onto the terminal
    '''
    if display:
        if same_line:
            print('{}'.format(print_str), end='')
        else:
            print('{}'.format(print_str))

    if same_line:
        log.write('{}'.format(print_str))
    else:
        log.write('{}\n'.format(print_str))
    log.flush()


def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,7, 0, 1, 2,....,7, 0, 1, 2,...., 7...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,7, 7, 7,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index


def padding_traj(traj, padding, idx_pad, zero_index):
    if padding == 'Zero':
        traj_tmp = traj
        traj_tmp[..., zero_index, :] = 0
        traj_pad = traj_tmp[..., idx_pad, :]
    else:
        traj_pad = traj[..., idx_pad, :]

    return traj_pad


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)  # DCT-II型 计算公式
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m


def dct_transform(time_seq, dct_matrix, n_frame):
    return torch.matmul(dct_matrix[:n_frame], time_seq)


def idct_transform(freq_seq, idct_matrix, n_frame):
    return torch.matmul(idct_matrix[:, :n_frame], freq_seq)


def normalize_traj(x_t, traj_sample, dis_mean, dis_std):
    """
    Normalize sample trajectory by dis_mean and dis_std
    :param traj_sample: [B, N, T, 2]
    :param dis_mean: [B, T, 2]
    :param dis_std: [B, T, 1]
    :return: normalized sample trajectory [B, N, T, 2]
    """

    sample_prediction = torch.exp(dis_std / 2)[:, None] * traj_sample / traj_sample.std(
        dim=1).mean(
        dim=(1, 2))[:, None, None, None]
    trajs = sample_prediction + dis_mean[:, None]


    return trajs


def moe_aug_loss_func(moe_outputs, input):
    loss_weights = {'traj': 1.0, 'balance': 0.1, 'consist': 0.5,
                    'special': 0.3, 'sparse': 0.2, 'recon': 0.2}
    total_loss = 0

    # 解构MOE输出
    gates = moe_outputs['gates']  # [B, E, S]
    experts = moe_outputs['experts']  # List[[B,P,D]]

    # 辅助损失
    if 'balance' in loss_weights:
        total_loss += loss_weights['balance'] * scale_aware_balancing_loss(gates)

    if 'consist' in loss_weights:
        total_loss += loss_weights['consist'] * temporal_consistency_loss(experts)

    if 'special' in loss_weights:
        total_loss += loss_weights['special'] * expert_specialization_loss(input, experts, gates)

    if 'sparse' in loss_weights:
        total_loss += loss_weights['sparse'] * sparsity_loss(gates)

    return total_loss


def scale_aware_balancing_loss(gates):
    """
    gates: [B, num_experts, num_scales] 门控权重
    """
    # 计算每个专家-尺度组合的平均使用率
    scale_expert_usage = gates.mean(dim=0)  # [num_experts, num_scales]

    # 鼓励各尺度专家利用率均衡
    scale_std = torch.std(scale_expert_usage, dim=0, unbiased=False).mean()  # 各尺度内部专家均衡
    expert_std = torch.std(scale_expert_usage, dim=1, unbiased=False).mean()  # 各专家跨尺度均衡

    return 0.5 * (scale_std + expert_std)


def temporal_consistency_loss(expert_outputs):
    """
        expert_outputs: List[Tensor] - 各专家的输出 [B, max_patches, D]
        patch_list: List[int] - 使用的patch尺寸列表，如[2,4,8]
        obs_len: int - 观测序列长度（如8帧）
        """
    losses = [] # 最大patch数

    for i in range(len(expert_outputs) - 1):
        # 对相邻尺度专家的输出计算相似度
        cos_sim = F.cosine_similarity(
            expert_outputs[i],
            expert_outputs[i + 1],
            dim=-1
        )
        losses.append(1 - cos_sim.mean())

    return sum(losses) / len(losses) if losses else 0.0


def expert_specialization_loss(x, expert_outputs, gates):
    """
    x: 输入轨迹 [B, T, 2]
    expert_outputs: 各专家输出列表
    gates: 门控权重 [B, num_experts, num_scales]
    """
    # 1. 提取运动特征（速度/加速度）
    velocity = x[:, 1:, 2:4] - x[:, :-1, 2:4]  # [B, T-1, 2]
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 2]

    # 2. 为每个样本分配运动模式标签
    motion_type = (torch.norm(acceleration, dim=-1).mean(-1) > 0.5).float() # [B]

    # 3. 鼓励专家对特定模式响应更强
    loss = 0

    for expert_id in range(gates.size(1)):
        expert_gates = gates[:, expert_id].mean(-1)  # [B]
        loss += nn.CrossEntropyLoss()(expert_gates.unsqueeze(0), motion_type.unsqueeze(0))

    return loss / gates.size(1)


def sparsity_loss(gates, target_sparsity=0.3):
    """
    采用L1约束+Top-k稀疏化
    """
    avg_gate = gates.mean(dim=[0, 2])  # [num_experts]

    # L1正则
    l1_loss = torch.norm(avg_gate, p=1)

    # Top-k稀疏
    k = int(gates.size(1) * target_sparsity)
    topk_val = torch.topk(avg_gate, k=k).values
    if k > 0:
        sparse_loss = topk_val[-1] - target_sparsity  # 鼓励刚好激活k个专家
    else:
        sparse_loss = 0

    return 0.5 * l1_loss + sparse_loss


def moe_regularization(gates, lambda_balance=0.1, lambda_entropy=0.01):
    """
    gates: [B, num_experts, num_scales] 门控权重
    """
    # 1. 专家级负载均衡
    expert_usage = gates.mean(dim=[0, 2])  # [num_experts]
    balance_loss = torch.std(expert_usage)

    # 2. 尺度级熵最大化
    scale_dist = gates.mean(dim=[0, 1])  # [num_scales]
    entropy_loss = -(scale_dist * torch.log(scale_dist + 1e-10)).sum()

    return lambda_balance * balance_loss + lambda_entropy * entropy_loss
