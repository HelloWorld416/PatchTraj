"""Evaluate JRDB trajectories at four prediction horizons."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.data_process import JrdbDataset, jrdb_seq_collate
from tool.tools import JRDBdata_process, dct_transform, get_dct_matrix


def val_jrdb(cfg, model, scene=None, dct_m=None, idct_m=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    if dct_m is None:
        dct_m, _ = get_dct_matrix(cfg.obs_len + cfg.pred_len)
    dct_m = dct_m.float().to(device)
    dataset = JrdbDataset(data_path=cfg.data_path, obs_len=cfg.obs_len,
                        pred_len=cfg.pred_len, training=False)
    loader = DataLoader(dataset, batch_size=cfg.get('test_bs', 128),
                        shuffle=False, num_workers=1, collate_fn=jrdb_seq_collate)
    performance = {metric: [0.0] * 4 for metric in ('ADE', 'FDE', 'JADE', 'JFDE')}
    samples = 0
    with torch.no_grad():
        for data in tqdm(loader):
            past_traj, pad_traj, fut_traj, _, _ = JRDBdata_process(cfg, data)
            dct_traj = dct_transform(pad_traj, dct_m, cfg.n_pre)
            fut_pred, _ = model(past_traj, dct_traj, exp='Test', mod=dct_traj)
            distances = torch.norm(fut_pred - fut_traj.unsqueeze(1), dim=-1) * cfg.data_scale
            for index in range(4):
                horizon = 3 * (index + 1)
                ade = distances[:, :, :horizon].mean(dim=-1)
                fde = distances[:, :, horizon - 1]
                performance['ADE'][index] += ade.min(dim=-1)[0].sum().item()
                performance['FDE'][index] += fde.min(dim=-1)[0].sum().item()
                performance['JADE'][index] += ade.mean(dim=0).min().item()
                performance['JFDE'][index] += fde.mean(dim=0).min().item()
            samples += distances.shape[0]
    if not samples:
        raise ValueError('The evaluation dataset is empty')
    for index in range(4):
        horizon_seconds = 1.2 * (index + 1)
        metrics = {key: values[index] / (samples if key in ('ADE', 'FDE') else len(loader))
                   for key, values in performance.items()}
        print('Evaluation\t' + '\t'.join(
            f'--{key}({horizon_seconds:g}s): {value:.4f}' for key, value in metrics.items()))
    return performance
