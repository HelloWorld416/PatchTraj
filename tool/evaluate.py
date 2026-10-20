import time

import numpy as np
import torch
from tqdm import tqdm
from datasets.data_process import ETHDataset, NBADataset, SDDDataset, seq_collate
from torch.utils.data import DataLoader
from tool.tools import ETHdata_process, SDDdata_process, NBAdata_process, \
    dct_transform, JRDBdata_process



def calculate_collision(cfg, fut_traj, pred_traj, seq_start_end_list, threshold=0.1):
    # calculate the best one with the smallest error
    distances = torch.norm(pred_traj - fut_traj.unsqueeze(1), dim=-1) * cfg.data_scale
    distances = torch.sum(distances, dim=-1)
    best_index = torch.argmin(distances, dim=-1)
    best_preds = [pred_traj[i][j] for i, j in enumerate(best_index)]
    best_preds = torch.stack(best_preds, dim=0)
    # # use fut_traj instead pred_traj to calculate the GT Collision Rate
    # best_preds = fut_traj
    total_collisions = 0
    total_possible_collisions = 0
    for seq_num, seq in enumerate(seq_start_end_list):
        start, end = seq[0], seq[1]
        trajs = best_preds[start:end]
        N = end - start
        for i in range(N):
            for j in range(i + 1, N):
                # calculate the distance between traj[i] and traj[j]
                distances_ = torch.norm(trajs[i] - trajs[j], dim=-1) * cfg.data_scale
                # calculate the occurred collisions
                collisions = torch.sum(distances_ < threshold)
                total_collisions += collisions
        # calculate the total possible collision frequency
        # total_possible_collisions_ = (N * (N - 1) // 2) * cfg.pred_len
        total_possible_collisions_ = torch.div(N * (N - 1), 2, rounding_mode='floor') * cfg.pred_len
        total_possible_collisions += total_possible_collisions_
    # 计算碰撞率
    # collision_rate = total_collisions / total_possible_collisions

    return total_collisions, total_possible_collisions



def val_model(cfg, model, dct_m, idct_m, scene=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dct_m = dct_m.float().to(device)
    idct_m = idct_m.float().to(device)

    # Prepare for Loading Dataset, opts: 'eth', 'sdd', 'nba'
    if cfg.dataset_type == 'ETH':
        test_dataset = ETHDataset(data_path=cfg.data_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                                      mode='test', scene_names=scene)
    elif cfg.dataset_type == 'SDD':
        test_dataset = SDDDataset(data_path=cfg.data_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len, flip_aug=False, mode='test')
    elif cfg.dataset_type == 'NBA':
        test_dataset = NBADataset(data_path=cfg.data_path, obs_len=cfg.obs_len, pred_len=cfg.pred_len, training=False)

    # Package dataset for training and testing
    if cfg.dataset_type == 'ETH':
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False, num_workers=1,
                                      collate_fn=seq_collate)
    else:
        test_loader = DataLoader(test_dataset, batch_size=cfg.get('test_bs', 256), shuffle=False, num_workers=1)

    if not cfg.dataset_type == 'NBA':
        performance = {'APD': 0, 'FDE': 0, 'ADE': 0,
                       'JADE': 0, 'JFDE': 0}
    else:
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0],
                       'JADE': [0, 0, 0, 0],
                       'JFDE': [0, 0, 0, 0]}

    exp = 'Test'
    samples = 0
    count = 0
    val_count = 0
    total_collisions = 0
    total_possible_collisions = 0
    if cfg.dataset_type == 'ETH':
        scene = scene[0]
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            if cfg.dataset_type == 'ETH':
                past_traj, pad_traj, initial_pos, fut_traj, agent_num_list = ETHdata_process(cfg, data)
            elif cfg.dataset_type == 'SDD':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = SDDdata_process(cfg, data, mode='train')
            elif cfg.dataset_type == 'NBA':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = NBAdata_process(cfg, data)
            elif cfg.dataset_type == 'JRDB':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = JRDBdata_process(cfg, data)

            dct_traj = dct_transform(pad_traj, dct_m, cfg.n_pre)
            mod_dct_traj = dct_transform(pad_traj, dct_m,
                                         cfg.n_pre)
            # start_time = time.time()
            fut_pred, _ = model(past_traj, dct_traj, exp=exp, mod=mod_dct_traj)
            # end_time = time.time()
            # inference_time = end_time - start_time
            # print('Inference time: {:.4f}ms'.format(inference_time * 1000))

            distances = torch.norm(fut_pred - fut_traj.unsqueeze(1), dim=-1) * cfg.data_scale
            samples += fut_pred.shape[0]

            if cfg.dataset_type == 'ETH':
                collisions, possible = calculate_collision(cfg, fut_traj, fut_pred, agent_num_list)
                total_collisions += collisions
                total_possible_collisions += possible

            diversity = torch.zeros((), device=device)
            if fut_pred.shape[1] == 1:
                diversity = 0.0
            for i in range(len(fut_pred)):
                dist_diverse = torch.pdist(fut_pred[i].reshape(cfg.num_sample, -1))
                diversity += dist_diverse.mean()

            ade = distances.mean(dim=-1).min(dim=-1)[0].sum()
            fde = distances[:, :, -1].min(dim=-1)[0].sum()
            jade = distances.mean(dim=-1).mean(dim=0).min()
            jfde = distances[:, :, -1].mean(dim=0).min()

            performance['APD'] += diversity.item()
            performance['ADE'] += ade.item()
            performance['FDE'] += fde.item()
            performance['JADE'] += jade.item()
            performance['JFDE'] += jfde.item()

            val_count += 1

    if cfg.dataset_type == 'ETH':
        print(
            'Evaluation on subset {}\t--APD: {:.4f}\t--ADE: {:.4f}\t--FDE: {:.4f}\t--JADE: {:.4f}\t--JFDE: {:.4f}\t--CR: {:.4f}'.format(
                scene,
                performance['APD'] / samples, performance['ADE'] / samples,
                performance['FDE'] / samples,
                performance['JADE'] / len(test_loader), performance['JFDE'] / len(test_loader), (total_collisions / total_possible_collisions if total_possible_collisions else 0.0))
            )
    else:
        print(
            'Evaluation on {}\t--APD: {:.4f}\t--ADE: {:.4f}\t--FDE: {:.4f}\t--JADE: {:.4f}\t--JFDE: {:.4f}'.format(
                cfg.dataset_type,
                performance['APD'] / samples, performance['ADE'] / samples,
                performance['FDE'] / samples,
                performance['JADE'] / len(test_loader), performance['JFDE'] / len(test_loader))
        )
