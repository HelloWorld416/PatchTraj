import math
import os
import pickle
import random

import pandas as pd
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

data_type = ['train', 'val', 'test']


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_array(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class ETHDataset(Dataset):
    """Dataloader for Trajectory dataset (ETH/UCY)"""
    def __init__(self, data_path, obs_len=8, pred_len=12, mode='train', flip_aug=False, scene_names=None):
        """
        :param data_path: dataset path
        :param obs_len: number of timestamps in input trajectories
        :param pred_len: number of timestamps in output trajectories
        :param mode: choose mode for dataset prepare
        """
        super(ETHDataset, self).__init__()

        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.mode = mode
        self.flip_aug = flip_aug
        self.max_peds_in_frame = 0
        self.norm_lap_matr = True
        skip = 1
        min_ped = 1
        threshold = 0.002

        if self.mode == 'train':
            self.data_type = data_type[0]
        elif self.mode == 'val':
            self.data_type = data_type[1]
        elif self.mode == 'test':
            self.data_type = data_type[2]

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for scene in scene_names:
            for subdir, dirs, files in os.walk(os.path.join(data_path, scene, self.data_type)):
                for file in files:
                    if not file.endswith('.txt'):
                        break
                    each_file_name = os.path.join(subdir, file)

                    data = pd.read_csv(each_file_name, sep='\t', index_col=False, header=None)
                    data = np.array(data)

                    frames = np.unique(data[:, 0]).tolist()
                    frame_data = []
                    for frame in frames:
                        frame_data.append(data[frame == data[:, 0], :])
                    # it calculates how many possible starting positions there are for sequence of length 'seq_len'
                    num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                    for idx in range(0, num_sequences * skip + 1, skip):
                        curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                        self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                                 self.seq_len))
                        curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                        curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                                   self.seq_len))
                        num_peds_considered = 0
                        _non_linear_ped = []

                        for _, ped_id in enumerate(peds_in_curr_seq):

                            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                         ped_id, :]
                            curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                            if pad_end - pad_front != self.seq_len:
                                continue

                            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                            curr_ped_seq = curr_ped_seq
                            # Make coordinates relative
                            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                            rel_curr_ped_seq[:, 1:] = \
                                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                            _idx = num_peds_considered
                            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                            curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                            # Linear vs Non-Linear Trajectory
                            _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                            curr_loss_mask[_idx, pad_front:pad_end] = 1
                            num_peds_considered += 1

                        if num_peds_considered > min_ped:
                            non_linear_ped += _non_linear_ped
                            num_peds_in_seq.append(num_peds_considered)
                            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                            seq_list.append(curr_seq[:num_peds_considered])
                            seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        if self.flip_aug:
            seq_list_ = seq_list[:, :, ::-1]
            seq_list = np.concatenate((seq_list, seq_list_), axis=0)
            num_peds_in_seq_ = num_peds_in_seq.copy()
            num_peds_in_seq = num_peds_in_seq + num_peds_in_seq_
            self.num_seq = self.num_seq * 2
        print("Total {} datas are: {}".format(self.mode, self.num_seq))

        # Convert numpy -> Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        results = {
            'obs_traj': self.obs_traj[start:end, :].permute(0, 2, 1),
            'pred_traj': self.pred_traj[start:end, :].permute(0, 2, 1),
        }

        return results


class NBADataset(Dataset):
    """Dataloader for the Trajectory datasets (NBA)"""
    def __init__(self, obs_len=8, pred_len=12, training=True, flip_aug=True, data_path="./datasets"):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        """

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.flip_aug = flip_aug

        if training:
            mode = 'train'
            data_root = os.path.join(data_path, "nba", "nba_train.npy")
        else:
            mode = 'test'
            data_root = os.path.join(data_path, "nba", "nba_test.npy")

        self.trajs = np.load(data_root)  # (N,30,11,2)
        self.trajs /= (94 / 28)
        if training:
            self.trajs = self.trajs[:32500]
            if self.flip_aug:
                flipped_data = np.flip(self.trajs, axis=1)
                self.trajs = np.concatenate((self.trajs, flipped_data), axis=0)
        else:
            self.trajs = self.trajs[:12500]

        self.batch_len = len(self.trajs)
        print("Total {} datas are: {}".format(mode, self.batch_len))

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3)
        self.traj_norm = self.traj_norm.permute(0, 2, 1, 3)
        self.actor_num = self.traj_abs.shape[1]  # num_agent

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        # labels = self.labels.reshape(self.batch_len, 11)

        results = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
        }
        return results


def initial_pos(traj_batches):
    batches = []
    for b in traj_batches:
        starting_pos = b[:,7,:].copy()/1000  # starting pos is end of past, start of future. scaled down.
        batches.append(starting_pos)

    return batches


class SDDDataset(Dataset):
    def __init__(self, data_path, obs_len, pred_len, flip_aug=True, mode='train'):
        super(SDDDataset, self).__init__()

        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mode = mode
        self.sdd_dataset, self.split_marks = [], [0]
        s = 0

        if self.mode == 'train':
            data_path = data_path + '/' + 'sdd/train_8_12.npy'
        elif self.mode == 'test':
            data_path = data_path + '/' + 'sdd/val_8_12.npy'
        data = np.load(data_path)

        assert data.shape[1] == obs_len + pred_len
        self.sdd_dataset.append(data[:, :, 0:2])
        s += len(data)
        self.split_marks.append(s)

        if self.mode == 'train':
            if flip_aug:
                flipped_data = np.flip(data, axis=1)
                self.sdd_dataset.append(flipped_data[:, :, 0:2])
                s += len(flipped_data)
                self.split_marks.append(s)

        self.sdd_dataset = np.concatenate(self.sdd_dataset, axis=0)
        print("Total {} datas are: {}".format(self.mode, len(self.sdd_dataset)))
        assert len(self.sdd_dataset) == s
        self.len = len(self.sdd_dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        results = {
            'past_traj': self.sdd_dataset[index, :self.obs_len, :],
            'fut_traj': self.sdd_dataset[index, self.obs_len:, :],
        }
        return results


class JrdbDataset(Dataset):
    """Dataloder for the JRDB dataset"""

    def __init__(
            self, obs_len=9, pred_len=12, training=True, data_path="./datasets"
    ):

        super(JrdbDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.training = training

        if self.training:
            data_root = os.path.join(data_path, "jrdb", "trajectories_jrdb_train.pkl")
            print("Loading training dataset: ", data_root)
        else:
            data_root = os.path.join(data_path, "jrdb", "trajectories_jrdb_val.pkl")
            print("Loading validaton or testing datasets: ", data_root)

        with open(data_root, "rb") as f:
            self.raw_data = pickle.load(f)

        self.data = self.raw_data[0]
        self.numPeds_in_sequence = self.raw_data[1]  # sum = data.shape[0]
        self.numSeqs_in_scene = self.raw_data[2]  # sum = len(numPeds in each sequence)

        self.data_len = sum(self.numSeqs_in_scene)
        print(self.data_len)

        self.traj_abs = torch.from_numpy(self.data).type(torch.float)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # index in self.numPeds_in_sequence: number of peds in current scene
        peds_num = self.numPeds_in_sequence[index]
        srt_idx = sum(self.numPeds_in_sequence[:index])
        end_idx = sum(self.numPeds_in_sequence[:index + 1])

        curr_traj = self.traj_abs[srt_idx:end_idx, :, :]

        if self.training:
            th = random.random() * np.pi
            cur_ori = curr_traj.clone()
            curr_traj[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            curr_traj[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)

        pre_motion_3D = curr_traj[:, :self.obs_len, :]
        fut_motion_3D = curr_traj[:, self.obs_len:, :]

        # # add random noise to gt history
        # for line in range(peds_num):
        #     select_idx = np.random.choice(self.obs_len, int(self.obs_len/2))
        #     pre_motion_3D[line, select_idx, :] += 0.2 * torch.rand(int(self.obs_len/2), 2)

        out = [
            torch.Tensor([peds_num]),
            pre_motion_3D, fut_motion_3D,
        ]
        return out


def seq_collate(data):
    databox = []
    for i in range(len(data)):
        values = data[i].values()
        databox.append(values)

    # (obs_seq_list, pred_seq_list, label_list) = zip(*databox)
    (obs_seq_list, pred_seq_list) = zip(*databox)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    obs_traj = torch.cat(obs_seq_list, dim=0)
    pred_traj = torch.cat(pred_seq_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    results = {
        'obs_traj': obs_traj,
        'pred_traj': pred_traj,
        'seq_start_end': seq_start_end,
    }

    return results


def jrdb_seq_collate(batch):
    peds_num_list = []
    pre_motion_3D_list = []
    fut_motion_3D_list = []

    for idx, sample in enumerate(batch):
        (peds_num, pre_motion_3D, fut_motion_3D) = sample

        peds_num_list.append(peds_num)
        pre_motion_3D_list.append(pre_motion_3D)
        fut_motion_3D_list.append(fut_motion_3D)

    peds_num = torch.Tensor(peds_num_list).reshape(-1)
    pre_motion_3D = torch.cat(pre_motion_3D_list, dim=0)
    fut_motion_3D = torch.cat(fut_motion_3D_list, dim=0)

    data = {
        'peds_num_per_scene': peds_num,
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'seq': 'jrdb',
    }

    return data
