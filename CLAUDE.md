# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vec2Vec is a framework for training GANs to convert between different embedding models. The project trains adversarial networks to learn mappings between embedding spaces (e.g., unsupervised GTE to supervised GTR embeddings) while preserving semantic relationships.

**Project page:** https://vec2vec.github.io/

## Development Commands

### Training
```bash
# Main training command
python train.py [EXPERIMENT_NAME] --num_points [NUMBER] --epochs [EPOCHS]

# Example with unsupervised config
python train.py unsupervised --num_points 100000 --epochs 10
```

### Evaluation
```bash
# Evaluate a trained model
python eval.py [MODEL_DIR]

# Example
python eval.py ./finetuning_unsupervised/model_name/
```

### Baselines
```bash
# Run optimal transport baseline
python ot_baseline.py [EXPERIMENT_NAME] --num_points [NUMBER]
```

### Environment Setup
The project uses conda for environment management:
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate base
```

## Architecture

### Core Components

**Translator Models** (`translators/`)
- `TransformTranslator.py`: Main translator using a transform module to map between embedding spaces
- `AbsNTranslator.py`: Abstract base class for N-to-N translators (many encoder to many encoder)
- `MLPWithResidual.py`: Residual MLP blocks used as adapters and transforms
- `LinearTranslator.py`: Simple linear translation baseline
- `Discriminator.py`: Discriminator networks for adversarial training
- Transform modules in `translators/transforms/`: UNet and other architectural variants

**Training Pipeline**
- Input adapters: Project each embedding type into a shared latent space (d_adapter)
- Transform: Core transformation in latent space (residual MLPs, UNets, etc.)
- Output adapters: Project latent representations back to target embedding space
- Multiple discriminators: Embedding space discriminator, latent space discriminator, supervised discriminator, and similarity discriminator

**Key Training Mechanisms**
1. **Reconstruction loss**: Encoder → latent → same encoder (identity mapping)
2. **Translation loss**: Source encoder → latent → target encoder
3. **Adversarial losses**: Multiple GANs operating on different aspects:
   - Main discriminator on translated embeddings
   - Supervised discriminator on reverse translations
   - Latent discriminator on latent representations
   - Similarity discriminator on pairwise similarity matrices
4. **VSP (Vector Space Preservation) loss**: Preserves geometric relationships
5. **Consistency losses** (cc_*): Cycle-consistency style losses for translation robustness

### Configuration System

All experiments use TOML configuration files in `configs/`:
- `configs/unsupervised.toml`: Main paper configuration
- `configs/ot.toml`: Optimal transport baseline configuration

Config sections:
- `[general]`: Seeds, datasets, embedding models, normalization
- `[translator]`: Architecture (style, depths, dimensions, normalization)
- `[discriminator]`: GAN type and discriminator architecture
- `[train]`: Batch size, learning rates, loss coefficients, early stopping
- `[gan]`: GAN-specific hyperparameters
- `[eval]`: Validation set size and evaluation parameters
- `[logging]`: W&B project settings and save directories

### Data Pipeline

**Datasets** (`utils/streaming_utils.py`, `utils/collate.py`)
- Streaming embeddings from HuggingFace datasets
- `MultiencoderTokenizedDataset`: Handles multiple encoder tokenization
- Supports text datasets: "nq" (Natural Questions), custom datasets
- MIMIC dataset support via `create_mimic.py`
- Dataset splits: Training uses shuffled subsets, validation uses separate seed

**Encoders**
- Loaded via `utils/model_utils.py::load_encoder()`
- Commonly used: 'gte' (GTE), 'gtr' (GTR), 'stella'
- Each encoder has its own input/output adapters in the translator

### Utilities

**utils/** directory:
- `gan.py`: Three GAN implementations (VanillaGAN, LeastSquaresGAN, RelativisticGAN)
- `train_utils.py`: Loss functions (rec_loss_fn, trans_loss_fn, vsp_loss_fn)
- `eval_utils.py`: Evaluation loop with retrieval metrics, early stopping
- `model_utils.py`: Model loading, translator instantiation, argument parsing
- `streaming_utils.py`: Dataset streaming and batch processing
- `wandb_logger.py`: Weights & Biases logging wrapper
- `collate.py`: Multi-encoder batching and collation

## Important Training Details

**GAN Training Stability**
- GAN training is highly unstable, especially across different encoder backbones
- May need multiple seeds for convergence
- Uses gradient clipping (max_grad_norm in config)
- R1 penalty regularization available for discriminators

**Mixed Precision**
- Supports fp16 and bf16 (automatically falls back to fp16 if bf16 unavailable)
- Configured via `mixed_precision` in config
- Uses HuggingFace Accelerate for distributed training

**Command-Line Override**
Any config parameter can be overridden via command line:
```bash
python train.py unsupervised --lr 1e-4 --bs 512 --loss_coefficient_gen 2.0
```

**Model Loading**
- Trained weights released at: https://github.com/rjha18/vec2vec/releases/tag/v1.0.0
- Load via `translator.load_state_dict()`
- See `eval.py` for usage example
- HuggingFace Hub support via `load_translator_from_hf()` in `utils/utils.py`

## File Reference Patterns

When implementing features or debugging:
- Loss computations: `train.py:114-154`
- Discriminator steps: `train.py:76-112`
- Translation forward pass: `translators/TransformTranslator.py:69-102`
- Adapter creation: `translators/TransformTranslator.py:52-57`
- Config loading: `utils/utils.py:92-107`
- Translator instantiation: `utils/utils.py:21-73`

## RESEARCH DIRECTION

This repository is a fork of the original vec2vec implementation, extended to enable **unsupervised cross-lingual translation without parallel data**. The original vec2vec demonstrated English→English embedding space translation; this research extends it to Target Language→English recovery for arbitrary target languages, with potential applications to dead, fragmentary, and low-resource languages.

### Core Research Goal

Leverage the universal geometry of text embeddings to perform unsupervised machine translation by:
1. Translating source language embeddings into English embedding space via vec2vec
2. Applying zero-shot embedding inversion in the English space to recover text
3. Avoiding the need for parallel corpora during training

### Phase 1: Russian→English (Validation)

**Objective:** Stress-test the approach with different scripts, tokenizers, and encoders while reproducing original paper results.

**Setup:**
- **Source encoder:** ruRoBERTa (Russian autoencoder/bi-encoder)
- **Target encoder:** Strong English model (GTE or E5)
- **Training data:** Unpaired Russian and English monolingual corpora
- **Evaluation data:** UN Parallel Corpus (Russian–English) held out exclusively for evaluation
- **Why Russian:** Non-Latin script, well-resourced for validation, author familiarity

**Pipeline:**
1. Russian text `d_ru` → ruRoBERTa → embedding `u`
2. Translate `u` via vec2vec: `F_ru→en(u)` → English embedding space
3. Apply zero-shot English embedding inversion to `F_ru→en(u)` → candidate English text `d̂_en`
4. Key insight: Reuse existing English inversion models, avoid training Russian decoders

**Evaluation Metrics:**
- **Intrinsic:** Mean cosine similarity, Top-1 nearest-neighbor accuracy, mean rank against true English embeddings on UN test pairs
- **Extrinsic:**
  - Zero-shot attribute inference accuracy in English space (semantic preservation)
  - Inversion judge accuracy (LLM rubric): Does `d̂_en` convey entities/relations from reference?
  - Human fluency/accuracy assessment from bilingual speakers

**Ablation Studies:**
- Remove adversarial terms, cycle-consistency, or VSP individually to measure contribution
- Test cross-backbone robustness (ruRoBERTa→GTE vs. ruRoBERTa→E5)
- Verify no paired sentences leak into training via data provenance audit

### Phase 2: Low-Resource & Partially Deciphered Languages

Once validated on Russian, extend to languages with extremely limited or no parallel data:
- **Target languages:** Etruscan, Iberian, Meroitic, Linear Elamite, Old Prussian, Gothic, other fragmentary corpora
- **Approach:**
  - Embed raw target-language text using language-appropriate autoencoder or multilingual model constrained to target segments
  - Translate into English embedding space via vec2vec
  - Use dictionary fragments, onomastic lists, or parallel glosses **only for evaluation**
- **Research questions:**
  - Sample efficiency: How much target-language data is minimally required?
  - Regularization regimes for extremely sparse corpora
  - Characterize limits: When is latent geometry learnable but lexical recovery impossible?

### Technical Foundation

**Modeling objective** (from original vec2vec, Jha et al. 2025):
1. **Space-specific adapters:** Input/output adapters around shared MLP backbone
2. **Adversarial losses:** Match distributions at latent and output levels
3. **Reconstruction:** Identity mapping (source → latent → source)
4. **Cycle-consistency:** Round-trip translation with low distortion (ru→en→ru and en→ru→en)
5. **VSP (Vector Space Preservation):** Preserve pairwise geometric relationships under translation

This objective induces a usable universal latent space enabling high-fidelity cross-space mapping without paired supervision.

### Expected Impact

If successful, this work:
- Converts universal geometric priors over embeddings into a practical unsupervised translation pipeline
- Enables content recovery from embeddings when parallel data is unavailable
- May enable partial recovery/translation of dead or fragmentary languages for the first time
- Technical feasibility supported by: vec2vec's cross-model alignment results, attribute inference preservation, and ~80% zero-shot inversion rates for some model pairs (Jha et al., 2025; Zhang, Morris, Shmatikov, 2025)

### Key Related Work

- **Vec2vec foundation:** Jha, Zhang, Shmatikov, Morris (2025). "Harnessing the Universal Geometry of Embeddings"
- **Embedding inversion:** Morris et al. (2023), Zhang, Morris, Shmatikov (2025)
- **Cross-lingual alignment without pairs:** Conneau et al. (2018), Artetxe et al. (2018), Lample et al. (2018)
- **Decipherment:** Ravi & Knight (2011)

### Implementation Notes for Developers

When extending vec2vec for cross-lingual work:
- **Data provenance is critical:** Audit training sets to ensure zero parallel data leakage
- **Encoder selection matters:** Choose encoders trained on appropriate monolingual corpora for target language
- **Inversion models:** Leverage existing English inversion models (e.g., vec2text) rather than training target-language decoders
- **GAN instability:** Cross-lingual translation may exhibit even greater instability than cross-model translation; expect extensive hyperparameter tuning and multiple seeds
- **Evaluation:** Always hold out parallel data (when available) exclusively for evaluation; never use for training

## Paper Citations

**Original Vec2Vec:**
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

**This Research (Proposal):**
Shalunov, E. & Neur, J. (2025). "Leveraging the universal geometry of natural language embeddings for unsupervised translation without pairs." University of California, Santa Barbara.
