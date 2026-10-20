"""Release smoke tests with generated data; no external datasets or weights."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from datasets.data_process import NBADataset, JrdbDataset
from model.denoising_model import PatchFormer

ROOT = Path(__file__).resolve().parents[1]


def make_model(cfg):
    return PatchFormer(
        input_feats=6, obs_len=cfg.obs_len, pred_len=cfg.pred_len,
        patch_size=cfg.patch_size, patch_list=cfg.patch_list, stride=cfg.stride,
        num_frames=cfg.n_pre, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims, num_experts=cfg.num_experts,
        patch_embed=cfg.patch_embed, dropout=cfg.dropout, num_sample=cfg.num_sample,
        dynamic_patch=cfg.dynamic_patch, fusion_mode=cfg.fusion_mode,
        patch_stride_mode=cfg.patch_stride_mode,
    )


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for directory in ('sdd', 'nba', 'jrdb', 'hotel/test'):
            (self.root / directory).mkdir(parents=True)
        for name in ('train_8_12', 'val_8_12'):
            np.save(self.root / 'sdd' / name, rng.normal(size=(2, 20, 2)))
        for name in ('nba_train', 'nba_test'):
            np.save(self.root / 'nba' / name, rng.normal(size=(1, 30, 11, 2)))
        for name in ('trajectories_jrdb_train', 'trajectories_jrdb_val'):
            with (self.root / 'jrdb' / (name + '.pkl')).open('wb') as stream:
                pickle.dump((rng.normal(size=(2, 21, 2)), [2], [1]), stream)
        rows = [[frame, agent, frame * .1 + agent, frame * .2]
                for frame in range(20) for agent in range(2)]
        np.savetxt(self.root / 'hotel/test/sample.txt', rows, delimiter='\t')

    def run_cli(self, *args, env=None):
        result = subprocess.run([sys.executable, 'main.py', *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True,
                                timeout=120, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_help_without_cuda(self):
        output = self.run_cli('--help', env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.assertIn('--data_path', output)

    def test_custom_data_root(self):
        nba = NBADataset(obs_len=10, pred_len=20, training=False, data_path=self.root)
        self.assertEqual(tuple(nba[0]['fut_motion_3D'].shape), (11, 20, 2))
        jrdb = JrdbDataset(training=False, data_path=self.root)
        self.assertEqual(tuple(jrdb[0][2].shape), (2, 12, 2))

    def test_model_interface(self):
        torch.set_num_threads(1)
        for name in ('sdd', 'nba', 'jrdb', 'hotel'):
            with self.subTest(dataset=name):
                cfg = OmegaConf.load(ROOT / 'configs' / (name + '.yml'))
                model = make_model(cfg).eval()
                with torch.no_grad():
                    freq = torch.randn(2, cfg.n_pre, 6)
                    pred, _ = model(torch.randn(2, cfg.obs_len, 6), freq,
                                    exp='Test', mod=freq)
                self.assertEqual(tuple(pred.shape), (2, cfg.num_sample, cfg.pred_len, 2))
                self.assertTrue(torch.isfinite(pred).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required by the data pipeline')
    def test_training_and_evaluation_entry_points(self):
        for name in ('sdd', 'nba', 'jrdb', 'hotel'):
            with self.subTest(dataset=name):
                cfg = OmegaConf.load(ROOT / 'configs' / (name + '.yml'))
                cfg.batchsize = 2
                cfg.test_bs = 2
                # Exercise the real pipeline with small models on shared GPUs.
                cfg.latent_dims = 16
                cfg.patch_embed = 16
                cfg.num_layers = 2
                path = self.root / (name + '.yml')
                OmegaConf.save(cfg, path)
                weights = self.root / (name + '.pt')
                torch.save({'model_dict': make_model(cfg).state_dict()}, weights)
                common = ['--cfg', path, '--data_path', self.root,
                          '--log_dir', self.root / 'logs', '--exp_name', name]
                if name == 'sdd':
                    self.run_cli(*common, '--mode', 'train', '--num_epoch', 1)
                    weights = self.root / 'logs/SDD/sdd/model_opt'
                    self.assertTrue(weights.is_file())
                output = self.run_cli(*common, '--mode', 'test', '--model_ckpt', weights)
                self.assertIn('Evaluation', output)
                self.assertIn('ADE', output)
                if name == 'hotel':
                    self.assertIn('subset hotel', output)
                self.assertNotIn('nan', output.lower())


if __name__ == '__main__':
    unittest.main()
