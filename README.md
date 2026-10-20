# PatchTraj

PyTorch implementation of PatchTraj for multimodal trajectory prediction, combining multi-scale trajectory patches, mixture-of-experts patch embedding, and time/frequency feature fusion. The frequency stream uses the discrete cosine transform (DCT).

## Environment

- Python 3.10
- PyTorch 1.12 or later with CUDA support
- An NVIDIA GPU is required for training and evaluation; 24 GB GPU memory is recommended for the default batches.

Run commands from the repository root:

```bash
git clone https://github.com/HelloWorld416/PatchTraj.git
cd PatchTraj
conda create -n patchtraj python=3.10 -y
conda activate patchtraj
# Original PyTorch/CUDA environment; alternatively install a CUDA build
# appropriate for your GPU and driver before the other dependencies.
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

Release smoke tests use Python 3.10 and PyTorch 1.12.0. Dependency ranges are not a claim that every version combination has been tested. Mamba, torchvision, and torchaudio are not required.

## Data Preparation

Obtain the preprocessed [ETH/UCY](https://drive.google.com/file/d/16ZBPBDo0rPxhNqt5h1bD-0rZfW6PdO9B/view), [SDD](https://drive.google.com/file/d/15DcLh6ZJygNp2r-CNM41IZ-N5PbzK2l8/view), and [NBA](https://drive.google.com/file/d/1_FRQywx76V2MVIX29ulPLEczuhqK-QMF/view) data using the links supplied with the original project. Extract them into this layout:

```text
datasets/
├── __init__.py
├── data_process.py
├── sdd/
│   ├── train_8_12.npy
│   └── val_8_12.npy
├── nba/
│   ├── nba_train.npy
│   └── nba_test.npy
├── jrdb/
│   ├── trajectories_jrdb_train.pkl
│   └── trajectories_jrdb_val.pkl
├── eth/{train,val,test}/*.txt
├── hotel/{train,val,test}/*.txt
├── univ/{train,val,test}/*.txt
├── zara1/{train,val,test}/*.txt
└── zara2/{train,val,test}/*.txt
```

Dataset files are not distributed in this repository. For JRDB, supply the preprocessed files from the original project distribution; this release does not provide a public JRDB download or a raw-data preprocessing pipeline. `--data_path /path/to/datasets` overrides the root for all datasets.

Expected formats:

- **SDD:** NumPy arrays `(N, 20, C)`, with x/y positions in the first two channels. The supplied validation file is used for evaluation.
- **NBA:** NumPy arrays `(N, 30, 11, 2)`. The loader converts coordinates using `94 / 28` and uses at most 32,500 training and 12,500 test sequences before augmentation.
- **JRDB:** A pickle containing trajectories `(total_agents, 21, 2)`, agent counts per sequence, and sequence counts per scene, in that order.
- **ETH/UCY:** Tab-separated frame ID, pedestrian ID, x, and y. Each scene directory contains its training and evaluation splits.

## Configuration

| Dataset | Configuration | Observed frames | Predicted frames | DCT coefficients | Patch scales | Samples |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| SDD | `configs/sdd.yml` | 8 | 12 | 8 | 2, 4, 8 | 20 |
| NBA | `configs/nba.yml` | 10 | 20 | 10 | 2, 5, 10 | 20 |
| JRDB | `configs/jrdb.yml` | 9 | 12 | 9 | 3, 9 | 20 |
| ETH/UCY | `configs/{eth,hotel,univ,zara1,zara2}.yml` | 8 | 12 | 8 | 2, 4, 8 | 20 |

The default model is named `patchformer` in the code. YAML files control dimensions, optimizer settings, batches, and training duration. Defaults use bidirectional fusion and a frequency reconstruction loss weight of `1.0`.

## Training and Evaluation

Runs write `log.txt`, `configs.yaml`, and `model_opt` into `logs/<DATASET>/`. `--exp_name NAME` adds a subdirectory; `--log_dir PATH` changes the output root. `model_opt` is selected by improving training loss, not validation ADE. Use distinct experiment names to preserve runs. Select a GPU with `CUDA_VISIBLE_DEVICES=0`.

### SDD

#### Training

```bash
python main.py --cfg configs/sdd.yml --mode train --manual_seed 0
```

#### Evaluation

```bash
python main.py --cfg configs/sdd.yml --mode test --model_ckpt logs/SDD/model_opt --exp_name eval --manual_seed 0
```

### NBA

#### Training

```bash
python main.py --cfg configs/nba.yml --mode train --manual_seed 0
```

#### Evaluation

```bash
python main.py --cfg configs/nba.yml --mode test --model_ckpt logs/NBA/model_opt --exp_name eval --manual_seed 0
```

### JRDB

#### Training

```bash
python main.py --cfg configs/jrdb.yml --mode train --manual_seed 0
```

#### Evaluation

```bash
python main.py --cfg configs/jrdb.yml --mode test --model_ckpt logs/JRDB/model_opt --exp_name eval --manual_seed 0
```

### ETH/UCY

Train and evaluate each split separately, with its own weights and logs.

#### Training

```bash
for scene in eth hotel univ zara1 zara2; do
  python main.py --cfg "configs/${scene}.yml" --mode train --exp_name "${scene}" --manual_seed 0
done
```

#### Evaluation

```bash
for scene in eth hotel univ zara1 zara2; do
  python main.py --cfg "configs/${scene}.yml" --mode test --model_ckpt "logs/ETH/${scene}/model_opt" --exp_name "${scene}_eval" --manual_seed 0
done
```

### Evaluation Metrics

Metrics are printed to the console. ADE/FDE select the best predicted sample. NBA reports horizons of 1, 2, 3, and 4 seconds; JRDB reports 1.2, 2.4, 3.6, and 4.8 seconds. ETH/UCY and SDD also report APD. The existing JADE/JFDE implementation selects a shared sample across each evaluation batch, then averages batch values; these metrics depend on batch composition. APD uses normalized model coordinates; displacement errors are rescaled with `data_scale`.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgement

The README organization follows [CondTraj](https://github.com/HelloWorld416/CondTraj). We thank the dataset authors and the PyTorch ecosystem contributors for their work.
