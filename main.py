import argparse
import os

import random

import numpy as np
import torch.cuda
from omegaconf import OmegaConf
from model.denoising_model import PatchFormer
from model.time_series_baselines import PatchTSTTrajectory, TimesNetTrajectory, FEDformerTrajectory
from build_trainer import Trainer
from tool.tools import generate_pad, get_dct_matrix
from tool.evaluate import val_model
from tool.evaluate_nba import val_nba
from tool.evaluate_jrdb import val_jrdb



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          type=str,       default='./datasets')
    parser.add_argument('--mode',               type=str,       default='train', choices=['train', 'test'], help="train or test")
    parser.add_argument('--model_name',         type=str,       default=None,
                        choices=['patchformer', 'patchtst', 'timesnet', 'fedformer'],
                        help='Model for trajectory prediction.')
    parser.add_argument('--log_dir',            type=str,       default='./logs')
    parser.add_argument('--manual_seed',        type=int,       default=0)
    parser.add_argument('--eval_iter',          type=int,       default=2)
    parser.add_argument('--cfg',                type=str,       default='./configs/sdd.yml', help="eth, sdd, nba or jrdb")
    parser.add_argument('--model_ckpt',         type=str,       default='')
    parser.add_argument('--fusion_mode',        type=str,       default=None,
                        choices=['concat', 'freq_to_time', 'time_to_freq', 'bidirectional'],
                        help='Fusion ablation mode for time/frequency streams.')
    parser.add_argument('--exp_name',           type=str,       default=None,
                        help='Optional subdirectory under log_dir/dataset_type.')
    parser.add_argument('--num_epoch',          type=int,       default=None,
                        help='Optional override for cfg.num_epoch.')
    parser.add_argument('--freq_recon_lambda',  type=float,     default=None,
                        help='Weight for frequency reconstruction loss.')
    parser.add_argument('--patch_list',         type=str,       default=None,
                        help='Comma-separated patch scale set, e.g. 2,5,10.')
    parser.add_argument('--patch_stride_mode',  type=str,       default=None,
                        choices=['nonoverlap', 'overlap'],
                        help='Patch extraction mode for DPM.')

    args = parser.parse_args()
    if args.mode == 'test' and not args.model_ckpt:
        parser.error('--model_ckpt is required in test mode')
    if not torch.cuda.is_available():
        parser.error('Training and evaluation require a CUDA-enabled PyTorch installation and GPU')
    torch.cuda.set_device(0)

    def prepare_seed(rand_seed):
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        torch.backends.cudnn.deterministic = True  # Add deterministic algorithm
        torch.backends.cudnn.benchmark = False  # Turn off benchmark optimization

    # initialize the seed
    prepare_seed(args.manual_seed)

    cfg = OmegaConf.load(args.cfg)
    cfg.data_path = args.data_path
    cfg.mode = args.mode
    cfg.model_name = args.model_name or cfg.get('model_name', 'patchformer')
    cfg.log_dir = args.log_dir
    cfg.manual_seed = args.manual_seed
    cfg.eval_iter = args.eval_iter
    cfg.fusion_mode = args.fusion_mode or cfg.get('fusion_mode', 'bidirectional')
    cfg.freq_recon_lambda = args.freq_recon_lambda if args.freq_recon_lambda is not None else cfg.get('freq_recon_lambda', 0.0)
    if cfg.model_name != 'patchformer':
        cfg.freq_recon_lambda = 0.0
        cfg.num_sample = 10
    if args.patch_list is not None:
        cfg.patch_list = [int(scale.strip()) for scale in args.patch_list.split(',') if scale.strip()]
    cfg.patch_stride_mode = args.patch_stride_mode or cfg.get('patch_stride_mode', 'nonoverlap')
    if args.num_epoch is not None:
        cfg.num_epoch = args.num_epoch
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    cfg.idx_pad, cfg.zero_index = generate_pad(cfg.padding, cfg.obs_len, cfg.pred_len)

    # get DCT matrix
    dct_m, idct_m = get_dct_matrix(cfg.obs_len + cfg.pred_len)

    model_kwargs = dict(
        input_feats=2 * 3,  # 2 means x, y
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        patch_size=cfg.patch_size,
        patch_list=cfg.patch_list,
        stride=cfg.stride,
        num_frames=cfg.n_pre,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        num_experts=cfg.num_experts,
        patch_embed=cfg.patch_embed,
        ff_size=cfg.get('ff_size', 1024),
        dropout=cfg.dropout,
        num_sample=cfg.num_sample,
        dynamic_patch=cfg.dynamic_patch,
        fusion_mode=cfg.fusion_mode,
        patch_stride_mode=cfg.patch_stride_mode,
    )
    model_registry = {
        'patchformer': PatchFormer,
        'patchtst': PatchTSTTrajectory,
        'timesnet': TimesNetTrajectory,
        'fedformer': FEDformerTrajectory,
    }
    model = model_registry[cfg.model_name](**model_kwargs)

    # Count total trainable parameter number
    model_parameters = filter(lambda a: a.requires_grad, model.parameters())
    parameters = sum([np.prod(a.size()) for a in model_parameters])
    print('>>>>>> Total trainable parameters are: {:.2f}M'.format(parameters / 1000000.0))

    # Log prepare
    log_path = os.path.join(cfg.log_dir, cfg.dataset_type)
    if cfg.get('exp_name', None):
        log_path = os.path.join(log_path, cfg.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = open(os.path.join(log_path, 'log.txt'), 'a+')

    OmegaConf.save(cfg, os.path.join(log_path, 'configs.yaml'))
    print(f'>>>>>> Model: {cfg.model_name}')
    print(f'>>>>>> Fusion mode: {cfg.fusion_mode}')
    print(f'>>>>>> Frequency reconstruction lambda: {cfg.freq_recon_lambda}')
    print(f'>>>>>> Patch list: {list(cfg.patch_list)} ({cfg.patch_stride_mode})')
    log.write(f'Model: {cfg.model_name}\n')
    log.write(f'Fusion mode: {cfg.fusion_mode}\n')
    log.write(f'Frequency reconstruction lambda: {cfg.freq_recon_lambda}\n')
    log.write(f'Patch list: {list(cfg.patch_list)} ({cfg.patch_stride_mode})\n')
    log.flush()

    # Train stage
    if cfg.mode == 'train':
        trainer = Trainer(
                cfg=cfg,
                model=model,
                dct_m=dct_m,
                idct_m=idct_m,
                log=log,
                log_path=log_path
            )
        trainer.loop()
    # Test stage
    elif cfg.mode == 'test':
        ckpt = torch.load(args.model_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_dict'])
        if cfg.dataset_type == 'ETH':
            scene_names = cfg.scene_names
            for scene in scene_names:
                scene = scene.split()
                val_model(cfg, model, dct_m, idct_m, scene)
        elif cfg.dataset_type == 'SDD':
            val_model(cfg, model, dct_m, idct_m)
        elif cfg.dataset_type == 'NBA':
            val_nba(cfg, model, dct_m=dct_m)
        elif cfg.dataset_type == 'JRDB':
            val_jrdb(cfg, model, dct_m=dct_m)


    log.close()
