# Lang2Lang: Unsupervised Cross-Lingual Translation via Vec2Vec

> **This is a research fork** extending the original Vec2Vec framework to enable unsupervised cross-lingual translation without parallel data.

## About This Project

This repository is a fork of [vec2vec](https://github.com/rjha18/vec2vec), extended to demonstrate **unsupervised machine translation by leveraging the universal geometry of text embeddings**. While the original vec2vec paper (Jha et al., 2025) demonstrated English→English embedding space translation, this research extends it to **Target Language→English recovery for arbitrary target languages**, with potential applications to dead, fragmentary, and low-resource languages.

### Research Goal

Instead of requiring parallel corpora (paired translations), this approach:
1. Translates source language embeddings into English embedding space via vec2vec
2. Applies zero-shot embedding inversion in the English space to recover text
3. Avoids the need for parallel data during training

**Phase 1: Russian→English validation** - Stress-testing the approach with different scripts, tokenizers, and encoders while reproducing original paper results using ruRoBERTa (Russian) → GTE/E5 (English) translation without paired data.

**Phase 2: Low-resource languages** - Extending to Etruscan, Iberian, Meroitic, Linear Elamite, Old Prussian, Gothic, and other fragmentary corpora.

For complete research proposal and methodology, see [docs/PROPOSAL.md](docs/PROPOSAL.md).

## System Requirements

### Hardware
- **For Training:** NVIDIA GPU with CUDA support
  - Minimum: 16GB VRAM (e.g., RTX 4090, A100 40GB)
  - Recommended: A100 80GB or multiple GPUs
- **For Development:** Any machine (CPU-only for code inspection)
- **RAM:** 32GB+ recommended
- **Storage:** ~100GB (conda environment, datasets, models, checkpoints)

### Software
- Python 3.11.4
- Conda (Anaconda or Miniconda)
- PyTorch 2.0.1 with CUDA support (environment.yml has CPU-only; must upgrade for training)
- Key dependencies: `transformers`, `datasets`, `accelerate`, `wandb`, `vec2text`

**Important:** The provided `environment.yml` contains CPU-only PyTorch. For GPU training, you must upgrade:
```bash
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Quick Start
```bash
# Clone and setup
git clone <this-repo>
cd lang2lang
conda env create -f environment.yml
conda activate base

# Upgrade to GPU PyTorch (required for training)
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install vec2text sentence-transformers

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

For detailed system requirements, cloud GPU options, and installation troubleshooting, see [docs/SYSTEM_REQUIREMENTS.md](docs/SYSTEM_REQUIREMENTS.md).

For step-by-step reproduction instructions with pretrained models and training from scratch, see [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

# Original Vec2Vec Documentation

> **Project page:** https://vec2vec.github.io/

Vec2Vec is a framework for training GANs (Generative Adversarial Networks) to convert between different embedding models. It allows for the transformation of embeddings from one latent space to another while preserving the semantic relationships between vectors.

## Overview

Vec2Vec uses adversarial training to learn mappings between different embedding spaces. It can translate embeddings from unsupervised models like GTE (General Text Embeddings) to supervised models like GTR (General Text Representations), allowing for better alignment and utility of various embedding types.

![Vec2Vec Universal Architecture](universal2.png)

## Configuration

Vec2Vec uses a toml configuration file with sections for general settings, translator architecture, discriminator parameters, training hyperparameters, GAN-specific settings, evaluation metrics, and logging options. These files are stored in the `configs/` folder of the repo.

## Usage

To run the main experiment using the `configs/[EXPERIMENT_NAME].toml` configuration, run:

```bash
python train.py [EXPERIMENT_NAME] --num_points [NUMBER OF POINTS] --epochs [EPOCHS]
```

Most of the experiments in the paper use `configs/unsupervised.toml`.

> [!NOTE]
> GAN training is *very* unstable (especially across backbones!) You may need to try multiple seeds for convergence!

### Command Line Arguments

Each entry in the toml configuration can be altered in two ways: (1) by directly changing the configuration file, or (2) adding a flag to the run command above.
The `train.py` script with accepts various parameters, including:

#### General Settings
- `--num_points`: Number of points to allocate to each encoder
- `--unsup_points`: Number of points to allocate to the unsupervised encoder (the supervised recieves the rest)
- `--unsup_emb`: Unsupervised embedding model (e.g., 'gte')
- `--sup_emb`: Supervised embedding model (e.g., 'gtr')
- `--dataset`: Dataset to use (e.g., "nq")
- `--epochs`: Number of epochs to train for
- `--seed`: Random seed for reproducibility
- `--sampling_seed`: Seed for sampling operations
- `--train_dataset_seed`: Seed for training dataset generation
- `--val_dataset_seed`: Seed for validation dataset generation

Please refer to the example `.toml` files for all possible settings (there are a lot!).

## Model Release
We are releasing the trained weights of the models used in the paper [here](https://github.com/rjha18/vec2vec/releases/tag/v1.0.0). To use the trained weights, use `translator.load_state_dict()`. For an example on usage, please refer to `eval.py`.

## The Paper
Our paper is available on ArXiv: [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540) (Jha, Zhang, Shmatikov, and Morris, 2025). If you find the code useful, please use the following citation:

```
@misc{jha2025harnessinguniversalgeometryembeddings,
      title={Harnessing the Universal Geometry of Embeddings}, 
      author={Rishi Jha and Collin Zhang and Vitaly Shmatikov and John X. Morris},
      year={2025},
      eprint={2505.12540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.12540}, 
}
```
