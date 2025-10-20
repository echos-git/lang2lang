# System Requirements for Vec2Vec Validation

This document outlines the system requirements needed to validate the original paper's findings of recovering hidden source text from a provided embedding space given a known autoencoder.

## Hardware Requirements

### Current Development Environment
- **Local Machine**: M4 Mac (ARM64 architecture)
- **Use Case**: Code development, small-scale testing, environment setup
- **Limitations**:
  - No NVIDIA GPU support on macOS
  - CPU-only PyTorch training (extremely slow for GAN training)
  - Not suitable for full reproduction experiments

### Production/Training Environment (Recommended)
- **GPU**: NVIDIA GPU with CUDA support
  - **Minimum**: 1x GPU with 16GB VRAM (e.g., RTX 4090, A100 40GB, or similar)
  - **Recommended**: 1x A100 80GB or multiple GPUs for faster training
  - **Why**: GAN training is computationally intensive and unstable without GPU acceleration
- **RAM**: 32GB+ system RAM recommended
  - Dataset streaming and multi-encoder tokenization are memory-intensive
  - Larger batch sizes require more RAM
- **Storage**:
  - ~50GB for conda environment and dependencies
  - ~1.1GB for pretrained model weights from release
  - ~10-50GB for datasets (depending on corpus size)
  - Additional space for checkpoints and logging

### Cloud GPU Options
Since you're planning to rent GPUs, consider:
- **AWS**: p3.2xlarge (V100), p4d.24xlarge (A100), g5.xlarge (A10G)
- **Google Cloud**: a2-highgpu-1g (A100)
- **Lambda Labs**: Often more cost-effective for ML workloads
- **Vast.ai**: Budget-friendly GPU rentals

## Software Requirements

### Python Environment
- **Python Version**: 3.11.4 (as specified in environment.yml)
- **Package Manager**: Conda (Anaconda or Miniconda)
- **Environment Size**: ~500+ packages via conda

### Critical Dependencies

#### Core ML Libraries
```yaml
- pytorch=2.0.1=cpu_py311h6d93b4c_0  # NOTE: CPU-only in environment.yml
- numpy=1.24.3
- datasets=2.12.0      # HuggingFace datasets for data loading
- transformers=4.29.2  # HuggingFace transformers for models
```

**IMPORTANT**: The provided `environment.yml` specifies **CPU-only PyTorch**. For GPU training, you'll need to replace this with CUDA-enabled PyTorch:
```bash
# After creating the environment, upgrade to GPU PyTorch:
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Training Infrastructure
```yaml
- accelerate  # HuggingFace Accelerate for distributed training
- wandb       # Weights & Biases for experiment tracking (optional but recommended)
```

#### Embedding Models
```yaml
- sentence-transformers  # Not explicitly in environment.yml, installed via pip or imported dependency
```

**Required for validation**: `sentence-transformers` library to load encoders:
- GTE (gte-base): `thenlper/gte-base`
- GTR (gtr-t5-base): `sentence-transformers/gtr-t5-base`
- Other supported models listed in `utils/model_utils.py::HF_FLAGS`

#### Text Inversion (Critical for Validation)
```python
# From utils/utils.py:18
from vec2text.models import InversionModel
```

**NOT included in environment.yml** - must be installed separately:
```bash
pip install vec2text
```

The paper's validation requires:
- `vec2text` library for embedding inversion (recovering text from embeddings)
- Pretrained inversion models from HuggingFace:
  - `ielabgroup/vec2text_gtr-base-st_inversion` (used in code)
  - `jxm/gte-32-noise-0.001` (for GTE)

#### Data Processing
```yaml
- pandas=1.5.3
- scipy=1.10.1
- scikit-learn=1.3.0
- tqdm=4.65.0
```

#### Configuration
```yaml
- toml=0.10.2  # For reading .toml config files
- pyyaml=6.0
```

### Operating System
- **Linux**: Preferred (environment.yml targets Linux with `_libgcc_mutex`, `_openmp_mutex`)
- **macOS**: Works for development, but GPU training not possible
- **Windows**: Should work with WSL2 + CUDA, but not explicitly tested

## Installation Steps

### 1. Create Conda Environment

**On M4 Mac (for development only):**
```bash
cd /Users/ephraim/Code/lang2lang
conda env create -f environment.yml
conda activate base
```

**On Linux GPU Server (for training):**
```bash
# Clone repository
git clone https://github.com/rjha18/vec2vec.git
cd vec2vec

# Create environment
conda env create -f environment.yml
conda activate base

# CRITICAL: Upgrade to GPU PyTorch
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. Install Additional Dependencies

```bash
# Install vec2text for embedding inversion
pip install vec2text

# Optional: Install sentence-transformers if not already installed
pip install sentence-transformers

# Optional: Verify installation
python -c "from vec2text.models import InversionModel; print('vec2text installed')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers installed')"
```

### 3. Download Pretrained Models

**For validation without training:**
```bash
# Download pretrained vec2vec models (1.1 GB)
wget https://github.com/rjha18/vec2vec/releases/download/v1.0.0/final_model_release.zip
unzip final_model_release.zip
```

**Models will auto-download from HuggingFace on first use:**
- Embedding models (GTE, GTR, etc.) - typically 100-500MB each
- Inversion models (vec2text) - typically 500MB-1GB each

### 4. Configure Experiment Tracking (Optional)

```bash
# If using Weights & Biases
wandb login

# Or disable wandb in config
# Edit configs/unsupervised.toml: use_wandb = false
```

## Validation Workflow

### Quick Validation (Pretrained Models)

**Objective**: Verify embedding→text recovery works with pretrained models

```bash
# Download pretrained model
wget https://github.com/rjha18/vec2vec/releases/download/v1.0.0/final_model_release.zip
unzip final_model_release.zip

# Run evaluation on pretrained model
python eval.py ./path/to/pretrained/model/
```

**Expected behavior**:
1. Loads pretrained vec2vec translator
2. Loads English embedding models (GTE, GTR)
3. Translates embeddings between spaces
4. Computes intrinsic metrics (cosine similarity, top-k accuracy)
5. Optionally performs text inversion to recover source text
6. Reports results in `results/` directory

### Full Training Reproduction

**Warning**: GAN training is highly unstable and requires:
- Multiple runs with different random seeds
- GPU acceleration (CPU training impractical)
- Extensive hyperparameter tuning
- Significant compute time (hours to days per model)

```bash
# Run main experiment with unsupervised config
python train.py unsupervised --num_points 100000 --epochs 10

# Monitor training
# - Loss curves in wandb (if enabled)
# - Checkpoints saved to ./finetuning_unsupervised/
# - Early stopping based on validation metrics
```

## Dataset Requirements

### Default Dataset: Natural Questions (NQ)
- **Source**: HuggingFace `datasets` library
- **Size**: Varies by subset, typically 10-100GB
- **Auto-download**: First run will download and cache
- **Format**: Streaming dataset (no need to download entirely)

### UN Parallel Corpus (for Russian→English)
- **Purpose**: Evaluation only (held out from training)
- **Access**: Must be downloaded separately
- **Format**: Parallel Russian-English sentences

### MIMIC Dataset (Medical)
- **Purpose**: Domain-specific experiments
- **Script**: `create_mimic.py`
- **Storage**: Preprocessed in `data/mimic/`

## Expected Resource Usage

### Training (per experiment)
- **GPU Memory**: 10-16GB VRAM
  - Batch size 256, embedding dim 768, typical translator
  - Smaller batch sizes for less memory (slower training)
- **GPU Time**: 5-20 hours per model
  - Depends on epochs, dataset size, convergence
  - May need multiple runs with different seeds
- **Disk Space**: 5-10GB
  - Checkpoints, logs, intermediate results
  - Model weights ~500MB-1GB per checkpoint

### Evaluation (pretrained)
- **GPU Memory**: 4-8GB VRAM
- **GPU Time**: 10-30 minutes
- **Disk Space**: 1-2GB for results

## Known Issues & Considerations

### 1. CPU vs GPU PyTorch
The `environment.yml` has **CPU-only PyTorch**. You MUST upgrade to GPU PyTorch for training:
```bash
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Mixed Precision
Config supports `fp16` and `bf16` mixed precision:
- **bf16**: Requires Ampere GPUs (A100, RTX 30xx+)
- **fp16**: Works on older GPUs (V100, RTX 20xx)
- Automatic fallback if bf16 not supported

### 3. GAN Training Instability
From README:
> GAN training is *very* unstable (especially across backbones!) You may need to try multiple seeds for convergence!

**Recommendations**:
- Start with original paper's config (`configs/unsupervised.toml`)
- Try seeds: 5, 10, 42, 123 if training fails to converge
- Monitor discriminator/generator loss balance
- Check gradient norms (should not explode)

### 4. M4 Mac Limitations
- **No CUDA**: Cannot train GANs efficiently
- **MPS backend**: PyTorch MPS backend unstable for this codebase
- **CPU-only**: Useful for code inspection, config editing, small tests
- **Recommendation**: Use M4 for development, rent GPU for experiments

### 5. Data Streaming
HuggingFace `datasets` streams data to avoid loading entire corpus:
- First run may be slow (downloading + caching)
- Subsequent runs are faster (cached)
- Configurable via `utils/streaming_utils.py`

## Minimal Setup for Code Inspection (M4 Mac)

If you only want to inspect code without running experiments:

```bash
# Create lightweight environment
conda create -n vec2vec python=3.11
conda activate vec2vec

# Install only essentials
pip install torch numpy pandas toml pyyaml

# Browse code, read configs, understand architecture
# No heavy dependencies needed for code review
```

## Next Steps

1. **Local (M4 Mac)**: Set up conda environment, inspect code, understand architecture
2. **GPU Server**: Rent GPU instance, transfer code, upgrade PyTorch to CUDA
3. **Validation**: Download pretrained models, run `eval.py` to verify setup
4. **Training**: Run small experiment (reduced `--num_points`) to verify pipeline
5. **Reproduction**: Full training run with paper's hyperparameters

## Support & Documentation

- **Paper**: https://arxiv.org/abs/2505.12540
- **Project Page**: https://vec2vec.github.io/
- **Original Repo**: https://github.com/rjha18/vec2vec
- **Pretrained Models**: https://github.com/rjha18/vec2vec/releases/tag/v1.0.0
- **Vec2Text (Inversion)**: https://github.com/jxmorris12/vec2text
