# Reproducing Vec2Vec Core Results

This guide provides minimal steps to reproduce the key findings from [vec2vec.github.io](https://vec2vec.github.io) using this repository on a cloud environment with an A100 GPU.

## Overview

The core vec2vec result demonstrates that GANs can learn to translate between embedding spaces (e.g., GTE → GTR) while preserving semantic relationships, enabling:
1. **Cross-model embedding translation** with high fidelity
2. **Text recovery from embeddings** via translation to invertible spaces
3. **Preservation of semantic attributes** across embedding spaces

## Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended)
- **RAM:** 32GB+ system memory
- **Storage:** 100GB+ free space for datasets and models

### Cloud Platform Setup

Choose one of these cloud providers:

**Lambda Labs:**
```bash
# Rent A100 instance via Lambda Labs dashboard
# SSH into instance
ssh ubuntu@<instance-ip>
```

**Google Cloud Platform:**
```bash
# Create A100 instance
gcloud compute instances create vec2vec-reproduce \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# SSH into instance
gcloud compute ssh vec2vec-reproduce --zone=us-central1-a
```

**AWS (p4d.24xlarge for A100):**
```bash
# Launch instance via AWS Console with Deep Learning AMI
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>
```

## Step 1: Environment Setup

### 1.1 Clone Repository
```bash
git clone https://github.com/rjha18/supervised_disc.git
cd supervised_disc
```

### 1.2 Install Conda (if not present)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
```

### 1.3 Create Environment
```bash
# Create conda environment from provided YAML
conda env create -f environment.yml
conda activate base
```

### 1.4 Verify GPU Access
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

## Step 2: Download Pre-trained Models (Optional)

To skip training and directly evaluate published results:

### 2.1 Download Released Weights
```bash
# Download from GitHub releases
wget https://github.com/rjha18/vec2vec/releases/download/v1.0.0/unsupervised_gte_gtr.pt -O pretrained_model.pt
```

### 2.2 Verify Model Loading
```bash
python -c "
import torch
state_dict = torch.load('pretrained_model.pt', map_location='cpu')
print(f'Model keys: {list(state_dict.keys())[:5]}...')
print('Model loaded successfully')
"
```

## Step 3: Train Vec2Vec Translator (GTE → GTR)

This reproduces the main unsupervised translation experiment from the paper.

### 3.1 Review Training Configuration
```bash
cat configs/unsupervised.toml
```

Key parameters:
- **Encoders:** GTE (source) → GTR (target)
- **Dataset:** Natural Questions (NQ)
- **Architecture:** Residual MLP with adapters
- **Training:** Multiple discriminators with VSP and cycle-consistency losses

### 3.2 Run Training
```bash
# Train with 100k data points for 10 epochs (reproduces paper setup)
python train.py unsupervised --num_points 100000 --epochs 10

# Monitor training progress
# Training will log to Weights & Biases (W&B) if configured
# Checkpoints saved to ./finetuning_unsupervised/
```

**Expected training time:** 4-8 hours on A100 (depending on batch size and mixed precision)

### 3.3 Monitor Training Metrics

Key metrics to track (printed during training):
- **Generator loss:** Should decrease and stabilize
- **Discriminator loss:** Should hover around 0.5-0.7 (balanced GAN)
- **Reconstruction loss:** MSE between input and reconstructed embeddings
- **VSP loss:** Vector space preservation metric
- **Validation retrieval accuracy:** Top-k accuracy on held-out set

### 3.4 Handle Training Instability

If training diverges (common with GAN training):
```bash
# Try different random seed
python train.py unsupervised --num_points 100000 --epochs 10 --seed 43

# Or adjust learning rate
python train.py unsupervised --num_points 100000 --epochs 10 --lr 5e-5
```

## Step 4: Evaluate Trained Model

### 4.1 Run Evaluation Script
```bash
# Evaluate your trained model
python eval.py ./finetuning_unsupervised/unsupervised_<timestamp>/

# Or evaluate pre-trained model (if downloaded)
python eval.py ./finetuning_unsupervised/pretrained/
```

### 4.2 Key Evaluation Metrics

The evaluation script computes:

1. **Retrieval Accuracy:**
   - Top-1, Top-5, Top-10 accuracy
   - Mean Reciprocal Rank (MRR)
   - Measures whether translated embeddings retrieve correct documents

2. **Embedding Similarity:**
   - Mean cosine similarity between translated and ground-truth embeddings
   - Should be > 0.85 for successful translation

3. **Vector Space Preservation:**
   - Correlation of pairwise distances before/after translation
   - Should be > 0.9 for geometric preservation

Expected results (from paper):
```
Top-1 Retrieval Accuracy: ~75-85%
Top-5 Retrieval Accuracy: ~90-95%
Mean Cosine Similarity: ~0.87-0.92
VSP Correlation: ~0.93-0.96
```

## Step 5: Test Text Recovery via Embedding Inversion

This validates the key application: recovering text from translated embeddings.

### 5.1 Install Vec2Text (Embedding Inversion Tool)
```bash
pip install vec2text
```

### 5.2 Create Inversion Test Script

Create `test_inversion.py`:
```python
import torch
from utils.model_utils import load_encoder
from utils.utils import load_translator_from_path
from vec2text import invert_embeddings

# Load encoders
gte_encoder = load_encoder('gte')  # Source encoder
gtr_encoder = load_encoder('gtr')  # Target encoder (invertible)

# Load trained translator
translator = load_translator_from_path('./finetuning_unsupervised/unsupervised_<timestamp>/')
translator.eval()

# Test text
test_text = "The capital of France is Paris."

# Step 1: Encode with GTE
with torch.no_grad():
    gte_embedding = gte_encoder.encode([test_text], convert_to_tensor=True)

# Step 2: Translate GTE → GTR using vec2vec
with torch.no_grad():
    translated_embedding = translator.translate(
        gte_embedding,
        source_encoder='gte',
        target_encoder='gtr'
    )

# Step 3: Invert GTR embedding to recover text
recovered_text = invert_embeddings(
    embeddings=translated_embedding,
    encoder=gtr_encoder,
    num_steps=20
)

print(f"Original: {test_text}")
print(f"Recovered: {recovered_text[0]}")
```

### 5.3 Run Inversion Test
```bash
python test_inversion.py
```

### 5.4 Evaluate Inversion Quality

Expected behavior:
- **High-quality translation:** Recovered text should preserve semantic meaning
- **Exact recovery:** Unlikely, but semantic equivalence expected
- **Typical accuracy:** ~60-80% semantic preservation (LLM judge or human eval)

Example outputs:
```
Original: "The capital of France is Paris."
Recovered: "Paris is the capital city of France."  # Semantically equivalent

Original: "Machine learning enables computers to learn from data."
Recovered: "Computers can learn from data using machine learning."  # Paraphrased
```

## Step 6: Run Baseline Comparisons

### 6.1 Optimal Transport Baseline
```bash
# Run OT baseline with same data
python ot_baseline.py unsupervised --num_points 100000

# Evaluate OT results
# OT provides a simpler alignment baseline without adversarial training
```

### 6.2 Compare Results

Vec2vec should outperform OT on:
- Retrieval accuracy (+5-10% typical improvement)
- Semantic attribute preservation
- Robustness to distribution shift

## Step 7: Validate Key Claims

### 7.1 Claim 1: Cross-Model Translation Fidelity

**Validation:**
```bash
# Measure cosine similarity between translated and ground-truth embeddings
# Should achieve > 0.85 mean similarity
python eval.py ./finetuning_unsupervised/unsupervised_<timestamp>/ | grep "Mean Cosine"
```

### 7.2 Claim 2: Semantic Attribute Preservation

**Validation:**
- Downstream task transfer (classification, clustering)
- Attribute inference accuracy in translated space
- See `eval.py` for attribute preservation metrics

### 7.3 Claim 3: Zero-Shot Inversion Enables Text Recovery

**Validation:**
```bash
# Run inversion test (Step 5)
# Compute semantic similarity (BERTScore, LLM judge)
python test_inversion.py
```

## Troubleshooting

### Training Fails to Converge
- **Solution 1:** Reduce learning rate (`--lr 1e-5`)
- **Solution 2:** Try different seed (`--seed 42`, `--seed 43`, etc.)
- **Solution 3:** Increase gradient clipping (`--max_grad_norm 0.5`)
- **Solution 4:** Disable some discriminators temporarily (edit config)

### Out of Memory (OOM)
- **Solution 1:** Reduce batch size (`--bs 256` or `--bs 128`)
- **Solution 2:** Enable gradient checkpointing (edit `train.py`)
- **Solution 3:** Use mixed precision (`--mixed_precision bf16`)

### Dataset Download Slow
- **Solution:** Pre-download via HuggingFace CLI:
```bash
huggingface-cli download BeIR/nq
```

### W&B Login Issues
```bash
# Disable W&B if not needed
export WANDB_MODE=disabled
python train.py unsupervised --num_points 100000 --epochs 10
```

## Expected Outcomes

After completing these steps, you should have:

1. ✅ **Trained GTE→GTR translator** achieving ~80% Top-1 retrieval accuracy
2. ✅ **Validated geometric preservation** with VSP correlation > 0.9
3. ✅ **Demonstrated text recovery** via embedding inversion pipeline
4. ✅ **Reproduced core paper results** on unsupervised embedding translation

## Minimal Validation Checklist

- [ ] Environment setup complete (conda, GPU verified)
- [ ] Training runs without errors for at least 1 epoch
- [ ] Evaluation script produces metrics (retrieval accuracy, cosine similarity)
- [ ] Inversion test recovers semantically similar text
- [ ] Results approximately match paper claims (within 5% margin)

## Time Estimate

| Step | Time (A100) |
|------|-------------|
| Environment setup | 15-30 min |
| Training (100k points, 10 epochs) | 4-8 hours |
| Evaluation | 10-20 min |
| Inversion testing | 5-10 min |
| **Total** | **5-9 hours** |

## Citation

If you use this reproduction in your research:

```bibtex
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

## Support

For issues reproducing results:
- Check original repo: https://github.com/rjha18/vec2vec
- Project page: https://vec2vec.github.io
- Paper: https://arxiv.org/abs/2505.12540
