import os
import numpy as np
import random
import torch
import time
import copy
from torch import nn
from tqdm import tqdm
from datasets.data_process import ETHDataset, NBADataset, SDDDataset, JrdbDataset, seq_collate, jrdb_seq_collate
from torch.utils.data import DataLoader
from tool.tools import ETHdata_process, SDDdata_process, NBAdata_process, JRDBdata_process, print_log, scale_aware_balancing_loss, \
    moe_regularization, moe_aug_loss_func, dct_transform, idct_transform
from model.transformer_model import EMA
from tool.evaluate import val_model


class Trainer:
    def __init__(self, cfg, model, dct_m, idct_m, log, log_path):
        super().__init__()

        # Setting device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.model = model.to(self.device)
        self.log = log
        self.log_path = log_path

        self.dct_m = dct_m.float().to(self.device)
        self.idct_m = idct_m.float().to(self.device)

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        self.iter = 0
        self.init_loss = 100000
        self.best_ade = 100000
        self.best_fde = 100000
        self.best_jade = 100000
        self.best_jfde = 100000
        self.model_name = self.cfg.get('model_name', 'patchformer')
        self.freq_recon_lambda = float(self.cfg.get('freq_recon_lambda', 0.0))
        if self.model_name != 'patchformer':
            self.freq_recon_lambda = 0.0

    def frequency_reconstruction_loss(self, fut_pred, past_traj, fut_traj):
        obs_traj = past_traj[..., 2:4]
        pred_complete = torch.cat(
            [obs_traj.unsqueeze(1).expand(-1, fut_pred.shape[1], -1, -1), fut_pred],
            dim=2,
        )
        gt_complete = torch.cat([obs_traj, fut_traj], dim=1)

        low_freq = min(self.cfg.n_pre, gt_complete.shape[1])
        dct_basis = self.dct_m[:low_freq]
        pred_coeff = torch.einsum('lt,bktd->bkld', dct_basis, pred_complete)
        gt_coeff = torch.einsum('lt,btd->bld', dct_basis, gt_complete)
        freq_dist = torch.linalg.vector_norm(pred_coeff - gt_coeff.unsqueeze(1), ord=2, dim=(2, 3))
        return freq_dist.min(dim=1)[0].mean()

    def get_balance_losses(self, moe_outputs, device):
        zero = torch.zeros((), device=device)
        if not moe_outputs or 't' not in moe_outputs or 'f' not in moe_outputs:
            return zero, zero
        if 'gates' not in moe_outputs['t'] or 'gates' not in moe_outputs['f']:
            return zero, zero
        return (
            scale_aware_balancing_loss(moe_outputs['t']['gates']),
            scale_aware_balancing_loss(moe_outputs['f']['gates']),
        )

    def initilization(self):
        # Prepare for Loading Dataset, opts: 'eth', 'sdd', 'nba'
        if self.cfg.dataset_type == 'ETH':
            self.train_dataset = ETHDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len,
                                       mode='train', flip_aug=True, scene_names=self.cfg.scene_names)
            self.test_dataset = ETHDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len,
                                      mode='test', flip_aug=False, scene_names=self.cfg.scene_names)
        elif self.cfg.dataset_type == 'SDD':
            self.train_dataset = SDDDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, flip_aug=True, mode='train')
            self.test_dataset = SDDDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, flip_aug=False, mode='test')
        elif self.cfg.dataset_type == 'NBA':
            self.train_dataset = NBADataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=True)
            self.test_dataset = NBADataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=False)
        elif self.cfg.dataset_type == 'JRDB':
            self.train_dataset = JrdbDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=True)
            self.test_dataset = JrdbDataset(data_path=self.cfg.data_path, obs_len=self.cfg.obs_len, pred_len=self.cfg.pred_len, training=False)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone, gamma=self.cfg.decay_gamma)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

        self.criterion = nn.MSELoss()
        self.count = 0

        # self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10  # 21 is used for nba, 13 is used for eth

    def prepare_to_train(self):
        self.model.train()
        self.train_loss = 0
        # Package dataset for training and testing
        if self.cfg.dataset_type == 'ETH':
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batchsize, shuffle=True, num_workers=1,
                                      collate_fn=seq_collate)
            self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=1,
                                     collate_fn=seq_collate)
        elif self.cfg.dataset_type == 'JRDB':
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batchsize, shuffle=True, num_workers=1,
                                      collate_fn=jrdb_seq_collate)
            self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=1,
                                     collate_fn=jrdb_seq_collate)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batchsize, shuffle=True, num_workers=1)
            self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=1)
        # self.logger.info(f"Starting training epoch {self.iter}:")

    def begin_to_train(self):
        exp = 'Train'
        train_count = 0
        loss_total, total_m_loss, total_j_loss, total_freq_loss, count = 0, 0, 0, 0, 0
        train_loader = tqdm(self.train_loader)
        for _, data in enumerate(train_loader):
            if self.cfg.dataset_type == 'ETH':
                past_traj, pad_traj, _, fut_traj, agent_num_list = ETHdata_process(self.cfg, data)
            elif self.cfg.dataset_type == 'SDD':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = SDDdata_process(self.cfg, data, mode='train')
            elif self.cfg.dataset_type == 'NBA':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = NBAdata_process(self.cfg, data)
            elif self.cfg.dataset_type == 'JRDB':
                past_traj, pad_traj, fut_traj, traj_mask, initial_pos = JRDBdata_process(self.cfg, data)

            # if self.cfg.dataset_type == 'ETH':
            #     fut_pred = []
            #     moe_outputs = {'t_gates': [],
            #                    'f_gates': [],
            #                    'experts': []
            #                    }
            #     for (start, end) in agent_num_list:
            #         dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
            #         mod_dct_traj = dct_transform(pad_traj, self.dct_m,
            #                                      self.cfg.n_pre)
            #         curr_pred, curr_moe_out = self.model(past_traj[start:end], dct_traj[start:end], exp=exp, mod=mod_dct_traj)
            #         fut_pred.append(curr_pred)
            #         moe_outputs['t_gates'].append(curr_moe_out['t']['gates'])
            #         moe_outputs['f_gates'].append(curr_moe_out['f']['gates'])
            #     fut_pred = torch.cat(fut_pred, dim=0)
            #     t_moe_gates = torch.cat(moe_outputs['t_gates'], dim=0)
            #     f_moe_gates = torch.cat(moe_outputs['f_gates'], dim=0)
            #     balance_loss1 = scale_aware_balancing_loss(t_moe_gates)
            #     balance_loss2 = scale_aware_balancing_loss(f_moe_gates)
            #     # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
            #     # fut_pred = fut_pred[:, :, self.cfg.obs_len:]
            # else:
            past_traj_copy = past_traj.cpu().numpy()
            # # FFT
            # fft_result = np.fft.fft(past_traj_copy, axis=1)
            # real_part = np.real(fft_result)  # 实部
            # imag_part = np.imag(fft_result)  # 虚部
            # freq_component = torch.from_numpy(np.concatenate([real_part, imag_part], axis=-1)).float().cuda()
            # # Wavelet
            # wavelet = 'db1'
            # level = 2
            # all_coeffs = []
            # for sample in past_traj_copy:  # 每个样本 shape: (8, 2)
            #     # 多通道小波分解 (直接处理 (x,y) 二维信号)
            #     coeffs = pywt.wavedec(sample, wavelet, level=level, axis=0)  # 沿时间轴(axis=0)分解
            #     flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])
            #     all_coeffs.append(flattened_coeffs.reshape(8, 6))
            # freq_component = np.stack(all_coeffs)
            # freq_component = torch.from_numpy(freq_component).cuda()
            # DCT
            dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
            mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                                         self.cfg.n_pre)
            # start_time = time.time()
            fut_pred, moe_outputs = self.model(past_traj, dct_traj, exp=exp, mod=mod_dct_traj)
            # end_time = time.time()
            # time_cost = end_time - start_time
            # print('inference time: {:.4f} ms'.format(time_cost*1000))
            balance_loss1, balance_loss2 = self.get_balance_losses(moe_outputs, fut_pred.device)
            # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
            # fut_pred = fut_pred[:, :, self.cfg.obs_len:]

            # # 计算样本一致性损失
            # pred_traj_std = fut_pred.reshape(-1, self.cfg.num_sample, self.cfg.pred_len * 2).std(dim=-1)
            # fut_traj_std = fut_traj.reshape(-1, self.cfg.pred_len * 2)[:, None].std(dim=-1)
            # loss_sample_consistency = torch.norm((pred_traj_std - fut_traj_std), p=2, dim=-1).mean()

            freq_recon_loss = self.frequency_reconstruction_loss(fut_pred, past_traj, fut_traj)
            marginal_dist_loss = (fut_pred - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).sum(axis=-1).min(dim=1)[
                0].mean()
            joint_dist_loss = (fut_pred - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).sum(axis=-1).mean(dim=0).min()
            loss = marginal_dist_loss * self.cfg.hyper_param1 + joint_dist_loss * self.cfg.hyper_param2 + \
                   self.freq_recon_lambda * freq_recon_loss + balance_loss1 + balance_loss2
            loss_total += loss.item()
            total_m_loss += marginal_dist_loss.item() * self.cfg.hyper_param1
            total_j_loss += joint_dist_loss.item() * self.cfg.hyper_param2
            total_freq_loss += freq_recon_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            train_count += 1
            # if train_count == 1:
            #     break
            # del dct_traj, mod_dct_traj, x_t, norm_trajs, initialized_trajs, trajs, pred_traj, loss

        self.iter += 1
        print_log('[{}] Epoch: {}\t\tLoss: {:.4f}\t\tm_loss: {:.4f}\t\tj_loss: {:.4f}\t\tfreq_loss: {:.4f}\t\tlambda: {:.2f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            self.iter, loss_total/train_count, total_m_loss/train_count, total_j_loss/train_count,
            total_freq_loss/train_count, self.freq_recon_lambda), self.log)

        self.train_loss = loss_total/train_count

    def prepare_to_val(self):
        self.scheduler.step()
        self.model.eval()
        self.val_loss = 0
        self.performance = {'APD': 0, 'ADE': 0, 'FDE': 0,
                       'JADE': 0, 'JFDE': 0}

    def begin_to_val(self):
        exp = 'test'
        val_count = 0
        loss_total, total_m_loss, total_j_loss, total_freq_loss = 0, 0, 0, 0
        with torch.no_grad():
            samples = 0
            for _, data in enumerate(tqdm(self.test_loader)):
                if self.cfg.dataset_type == 'ETH':
                    past_traj, pad_traj, _, fut_traj, agent_num_list = ETHdata_process(self.cfg, data)
                elif self.cfg.dataset_type == 'SDD':
                    past_traj, pad_traj, fut_traj, traj_mask, initial_pos = SDDdata_process(self.cfg, data, mode='train')
                elif self.cfg.dataset_type == 'NBA':
                    past_traj, pad_traj, fut_traj, traj_mask, initial_pos = NBAdata_process(self.cfg, data)
                elif self.cfg.dataset_type == 'JRDB':
                    past_traj, pad_traj, fut_traj, traj_mask, initial_pos = JRDBdata_process(self.cfg, data)

                # if self.cfg.dataset_type == 'ETH':
                #     fut_pred = []
                #     moe_outputs = {'t_gates': [],
                #                    'f_gates': [],
                #                    'experts': []
                #                    }
                #     for (start, end) in agent_num_list:
                #         dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
                #         mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                #                                      self.cfg.n_pre)
                #         curr_pred, curr_moe_out = self.model(past_traj[start:end], dct_traj[start:end], exp=exp, mod=mod_dct_traj)
                #         fut_pred.append(curr_pred)
                #         moe_outputs['t_gates'].append(curr_moe_out['t']['gates'])
                #         moe_outputs['f_gates'].append(curr_moe_out['f']['gates'])
                #     fut_pred = torch.cat(fut_pred, dim=0)
                #     t_moe_gates = torch.cat(moe_outputs['t_gates'], dim=0)
                #     f_moe_gates = torch.cat(moe_outputs['f_gates'], dim=0)
                #     balance_loss1 = scale_aware_balancing_loss(t_moe_gates)
                #     balance_loss2 = scale_aware_balancing_loss(f_moe_gates)
                #     # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
                #     # fut_pred = fut_pred[:, :, self.cfg.obs_len:]
                # else:
                past_traj_copy = past_traj.cpu().numpy()
                # # FFT
                # fft_result = np.fft.fft(past_traj_copy, axis=1)
                # real_part = np.real(fft_result)  # 实部
                # imag_part = np.imag(fft_result)  # 虚部
                # freq_component = torch.from_numpy(np.concatenate([real_part, imag_part], axis=-1)).float().cuda()
                # # Wavelet
                # wavelet = 'db1'
                # level = 2
                # all_coeffs = []
                #
                # for sample in past_traj_copy:  # 每个样本 shape: (8, 2)
                #     # 多通道小波分解 (直接处理 (x,y) 二维信号)
                #     coeffs = pywt.wavedec(sample, wavelet, level=level, axis=0)  # 沿时间轴(axis=0)分解
                #     flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])
                #     all_coeffs.append(flattened_coeffs.reshape(8, 6))
                # freq_component = np.stack(all_coeffs)
                # freq_component = torch.from_numpy(freq_component).cuda()
                # DCT
                dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
                mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                                             self.cfg.n_pre)
                fut_pred, moe_outputs = self.model(past_traj, dct_traj, exp=exp, mod=mod_dct_traj)
                balance_loss1, balance_loss2 = self.get_balance_losses(moe_outputs, fut_pred.device)
                # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
                # fut_pred = fut_pred[:, :, self.cfg.obs_len:]

                # # 计算样本一致性损失
                # pred_traj_std = fut_pred.reshape(-1, self.cfg.num_sample, self.cfg.pred_len * 2).std(dim=-1)
                # fut_traj_std = fut_traj.reshape(-1, self.cfg.pred_len * 2)[:, None].std(dim=-1)
                # loss_sample_consistency = torch.norm((pred_traj_std - fut_traj_std), p=2, dim=-1).mean()

                freq_recon_loss = self.frequency_reconstruction_loss(fut_pred, past_traj, fut_traj)
                marginal_dist_loss = (fut_pred - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).sum(axis=-1).min(dim=1)[
                    0].mean()
                joint_dist_loss = (fut_pred - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).sum(axis=-1).mean(
                    dim=0).min()
                loss = marginal_dist_loss * self.cfg.hyper_param1 + joint_dist_loss * self.cfg.hyper_param2 + \
                       self.freq_recon_lambda * freq_recon_loss + balance_loss1 + balance_loss2
                loss_total += loss.item()
                total_m_loss += marginal_dist_loss.item() * self.cfg.hyper_param1
                total_j_loss += joint_dist_loss.item() * self.cfg.hyper_param2
                total_freq_loss += freq_recon_loss.item()
                val_count += 1
                # if val_count > 1:
                #     break

        self.val_loss = loss_total / val_count
        print_log('Evaluation\t--Val_Loss: {:.4f}\t--val_m_loss: {:.4f}\t--val_j_loss: {:.4f}\t--val_freq_loss: {:.4f}\t--lambda: {:.2f}'.format(self.val_loss, total_m_loss/val_count,
                                                                                                      total_j_loss/val_count,
                                                                                                      total_freq_loss/val_count,
                                                                                                      self.freq_recon_lambda), self.log)

    def after_val(self):
        exp = 'test'
        if self.iter % self.cfg.eval_iter == 0 and self.iter != 0:
            with torch.no_grad():
                samples = 0
                count = 0
                for _, data in enumerate(tqdm(self.test_loader)):
                    if self.cfg.dataset_type == 'ETH':
                        past_traj, pad_traj, _, fut_traj, agent_num_list = ETHdata_process(self.cfg, data)
                    elif self.cfg.dataset_type == 'SDD':
                        past_traj, pad_traj, fut_traj, traj_mask, initial_pos = SDDdata_process(self.cfg, data, mode='train')
                    elif self.cfg.dataset_type == 'NBA':
                        past_traj, pad_traj, fut_traj, traj_mask, initial_pos = NBAdata_process(self.cfg, data)
                    elif self.cfg.dataset_type == 'JRDB':
                        past_traj, pad_traj, fut_traj, traj_mask, initial_pos = JRDBdata_process(self.cfg, data)

                    # dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
                    # mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                    #                              self.cfg.n_pre)

                    # if self.cfg.dataset_type == 'ETH':
                    #     fut_pred = []
                    #     for (start, end) in agent_num_list:
                    #         dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
                    #         mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                    #                                      self.cfg.n_pre)
                    #         curr_pred, _ = self.model(past_traj[start:end], dct_traj[start:end], exp=exp, mod=mod_dct_traj)
                    #         fut_pred.append(curr_pred)
                    #     fut_pred = torch.cat(fut_pred, dim=0)
                    #     # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
                    #     # fut_pred = fut_pred[:, :, self.cfg.obs_len:]
                    # else:
                    past_traj_copy = past_traj.cpu().numpy()
                    # # FFT
                    # fft_result = np.fft.fft(past_traj_copy, axis=1)
                    # real_part = np.real(fft_result)  # 实部
                    # imag_part = np.imag(fft_result)  # 虚部
                    # freq_component = torch.from_numpy(np.concatenate([real_part, imag_part], axis=-1)).float().cuda()
                    # # Wavelet
                    # wavelet = 'db1'
                    # level = 2
                    # all_coeffs = []
                    #
                    # for sample in past_traj_copy:  # 每个样本 shape: (8, 2)
                    #     # 多通道小波分解 (直接处理 (x,y) 二维信号)
                    #     coeffs = pywt.wavedec(sample, wavelet, level=level, axis=0)  # 沿时间轴(axis=0)分解
                    #     flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])
                    #     all_coeffs.append(flattened_coeffs.reshape(8, 6))
                    # freq_component = np.stack(all_coeffs)
                    # freq_component = torch.from_numpy(freq_component).cuda()
                    # DCT
                    dct_traj = dct_transform(pad_traj, self.dct_m, self.cfg.n_pre)
                    mod_dct_traj = dct_transform(pad_traj, self.dct_m,
                                                 self.cfg.n_pre)
                    fut_pred, moe_outputs = self.model(past_traj, dct_traj, exp=exp, mod=mod_dct_traj)
                    # fut_pred = idct_transform(fut_pred, self.idct_m, self.cfg.n_pre)
                    # fut_pred = fut_pred[:, :, self.cfg.obs_len:]

                    if self.iter % self.cfg.eval_iter == 0 and self.iter != 0:
                        distances = torch.norm(fut_pred - fut_traj.unsqueeze(1), dim=-1) * self.cfg.data_scale

                        ade = distances.mean(dim=-1).min(dim=-1)[0].sum()
                        fde = distances[:, :, -1].min(dim=-1)[0].sum()
                        jade = distances.mean(dim=-1).mean(dim=0).min()
                        jfde = distances[:, :, -1].mean(dim=0).min()

                        self.performance['ADE'] += ade.item()
                        self.performance['FDE'] += fde.item()
                        self.performance['JADE'] += jade.item()
                        self.performance['JFDE'] += jfde.item()

                        samples += distances.shape[0]
                        count += 1

            if self.best_ade > self.performance['ADE'] / samples:
                self.best_ade = self.performance['ADE'] / samples
            if self.best_fde > self.performance['FDE'] / samples:
                self.best_fde = self.performance['FDE'] / samples
            if self.best_jade > self.performance['JADE'] / count:
                self.best_jade = self.performance['JADE'] / count
            if self.best_jfde > self.performance['JFDE'] / count:
                self.best_jfde = self.performance['JFDE'] / count
            print_log('Evaluation\t--ADE: {:.4f}\t--FDE: {:.4f}\t--JADE: {:.4f}\t--JFDE: {:.4f}'.format(
                self.performance['ADE'] / samples, self.performance['FDE'] / samples,
                self.performance['JADE'] / len(self.test_loader), self.performance['JFDE'] / len(self.test_loader)), self.log)
            if self.iter % 10 == 0:
                print_log('Epoch {}: |----Best ADE: {:.4f}----|----Best FDE: {:.4f}----|----Best JADE: {:.4f}----|----Best JFDE: {:.4f}----|\t'.format(
                    self.iter, self.best_ade, self.best_fde, self.best_jade, self.best_jfde), self.log)

        # save model parameters
        # scene_names = self.cfg.scene_names
        scene_names = None
        if self.init_loss > self.train_loss:
            self.init_loss = self.train_loss
            if scene_names is not None:
                subset_path = self.log_path + '/' + self.cfg.scene_names[0]
                if not os.path.exists(subset_path):
                    os.makedirs(subset_path)
                model_path = subset_path + '/' + 'model_opt'
            else:
                model_path = self.log_path + '/' + 'model_opt'
            if self.cfg.ema is True:
                model_state = {'model_dict': self.ema_model.state_dict()}
                torch.save(model_state, model_path)
            else:
                model_state = {'model_dict': self.model.state_dict()}
                torch.save(model_state, model_path)

    def loop(self):
        self.initilization()
        for self.iter in range(0, self.cfg.num_epoch):
            self.prepare_to_train()
            self.begin_to_train()
            self.prepare_to_val()
            self.begin_to_val()
            self.after_val()
