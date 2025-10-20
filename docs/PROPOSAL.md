# Leveraging the universal geometry of natural language embeddings for unsupervised translation without pairs

**Ephraim Shalunov¹, Jeremi Neur²**

¹Department of Computer Science
²Department of Electrical and Computer Engineering
University of California, Santa Barbara
{shalunov,jeremineur}@ucsb.edu

---

> **Note:** This proposal is proprietary. Please do not distribute without the consent of the authors.

> **Note:** An LLM pipeline was used to edit this document for grammar and style consistency. Content is human.

---

## Proposed Work

Training language models for translation between languages typically requires paired data: given an input in the source language, the model is supervised to produce the aligned target sentence. Classic and recent lines of work include decipherment and unsupervised MT using monolingual corpora with dual objectives and back-translation; cross-lingual word-embedding alignment via orthogonal Procrustes and Gromov–Wasserstein matching; and optimal-transport or adversarial/cycle-consistency formulations for sequence or representation space alignment [representative prior: Ravi & Knight 2011; Artetxe et al. 2018; Lample et al. 2018; Conneau et al. 2018; Alvarez-Melis & Jaakkola 2018].

This supervised/paired paradigm is unsuitable for dead or scarcely attested languages. In the absence of human translations, large classes of languages (e.g., Etruscan, Iberian, Meroitic; partially attested scripts like Linear Elamite; and fragmentary corpora for Old Prussian, Gothic) remain un- or only partially translated despite enormous LLM progress in translation.

Our proposed work leverages a recent result on the **universal geometry of text embeddings** to remove the need for pairs. In *Harnessing the Universal Geometry of Embeddings* (Jha, Zhang, Shmatikov, Morris, 2025), the authors introduce **vec2vec**, an unsupervised translator between embedding spaces that learns space-specific adapters around a shared latent "universal" representation and optimizes adversarial objectives plus reconstruction, cycle-consistency, and vector-space-preservation (VSP) constraints. Empirically, unpaired embeddings from different models are mapped into near-identical latents and translated across spaces with high cosine similarity and exact-match rates—moreover, translated embeddings retain semantics sufficient for zero-shot attribute inference and for text inversion (content reconstruction) in as many as ~80% of documents for some model pairs.

In the original paper, the demonstration is within modern languages and model families (e.g., BERT/T5-style English encoders), establishing English→English recovery via translation and inversion. We seek to extend this to **Target→English recovery for arbitrary target languages**, enabling unsupervised translation. Our first demonstration will be **Russian→English**, chosen to stress-test different scripts, tokenizers, and encoders, as well as reproduce the paper's results. Russian is chosen from among the five non-English UN languages due to its non-Latin script alongside author Shalunov's familiarity with the language.

## Methodology

### 1. Encoders and corpora

We will use a Russian autoencoder/bi-encoder (specifically, ruRoBERTa) to produce fixed-length embeddings for monolingual Russian text. For the English target space, we will use a strong public English embedding model (e.g., GTE or E5). After validating on the easier, paired case, we will attempt unpaired; then, training of vec2vec will consume unpaired Russian and English embedding samples drawn from large monolingual corpora. The UN Parallel Corpus (Russian–English) will be then held out **only for evaluation**, providing human ground truth to measure translation quality without leaking pairs into training.

### 2. Modeling objective

Following Jha et al., we will learn space-specific input/output adapters around a shared MLP backbone and optimize:

- **(i)** Adversarial losses at latent and output levels to match distributions
- **(ii)** **Reconstruction** (identity through the latent back into the same space)
- **(iii)** **Cycle-consistency** (round-trip Russian → English → Russian and vice versa with low distortion)
- **(iv)** **VSP** to preserve pairwise geometry under translation

This objective is precisely the mechanism that induces a usable universal latent and high-fidelity cross-space mapping.

### 3. Russian→English translation pipeline

For a Russian sentence *d_ru*, compute *u = M_ru(d_ru)*. Translate *u* into the English space via *F_ru→en*. To obtain text, we apply a **zero-shot embedding inversion** method trained for the English embedding space to *F_ru→en(u)*, yielding a candidate English sentence *d̂_en*. Because inversion models operate in the known English space, we avoid training a decoder for Russian and exploit existing inversion techniques shown to recover salient content from embeddings. Intermediate zero-shot attribute inference in the English space will quantify semantic preservation prior to full inversion.

### 4. Evaluation

**Intrinsic:** Mean cosine similarity, Top-1 nearest-neighbor accuracy, and mean rank of *F_ru→en(u)* against true *v = M_en(d_en)* on the UN test pairs (not used in training).

**Extrinsic:**
- **(a)** Zero-shot attribute inference accuracy in English
- **(b)** Inversion judge accuracy (LLM-assisted rubric) for whether *d̂_en* conveys entities/relations from the reference
- **(c)** Human assessment by those fluent in both languages

Metrics mirror those used by Jha et al. for comparability, though the human judgment is of course subjective.

### 5. Ablations and controls

We will ablate adversarial terms, cycle-consistency, and VSP to quantify each term's contribution to Russian→English fidelity; test cross-backbone robustness (e.g., ruRoBERTa→GTE vs. ruRoBERTa→E5); and verify no paired sentences appear during training by auditing data provenance.

## Extension to Low-Resource Languages

Once validated on Russian, we will attempt **low-resource and partially deciphered targets** by embedding raw target-language text with a language-appropriate autoencoder (or multilingual model constrained to target-language segments) and translating into the English embedding space, using dictionary fragments, onomastic lists, or parallel glosses **only for evaluation**. For languages with extremely limited corpora (e.g., Etruscan, Iberian), we will study sample-efficiency/regularization regimes and characterize limits of inversion when latent geometry is learnable but lexical content is sparse.

## Impact

If successful, our work converts the universal geometric prior over language embeddings into a practical pipeline for **unsupervised translation without pairs**, enabling content recovery from embeddings even when parallel data is unavailable. Perhaps, if successful, we will be able to recover portions of dead or fragmentary languages. Results reported by Jha et al. on cross-model alignment, attribute inference, and zero-shot inversion support the technical feasibility of this approach.

## Related Work

**Conneau, Alexis et al. (2018).** "Word Translation Without Parallel Data". In: *International Conference on Learning Representations (ICLR)*. Originally arXiv:1710.04087. URL: https://openreview.net/forum?id=H196sainb.

**Huang, Yu-Hsiang et al. (2024).** "Transferable Embedding Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries". In: *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*. Association for Computational Linguistics, pp. 4193–4205. URL: https://aclanthology.org/2024.acl-long.230/.

**Jha, Rishi et al. (2025).** "Harnessing the Universal Geometry of Embeddings". In: arXiv: 2505.12540 [cs.CL]. URL: https://arxiv.org/abs/2505.12540.

**Morris, John X. et al. (2023).** "Text Embeddings Reveal (Almost) As Much As Text". In: *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*. Association for Computational Linguistics. DOI: 10.48550/arXiv.2310.06816. URL: https://arxiv.org/abs/2310.06816.

**Ravi, Sujith and Knight, Kevin (2011).** "Deciphering Foreign Language". In: *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)*. Portland, Oregon, USA: Association for Computational Linguistics, pp. 12–21. URL: https://aclanthology.org/P11-1002.pdf.

**Zhang, Collin, Morris, John X., and Shmatikov, Vitaly (2025).** "Universal Zero-shot Embedding Inversion". In: arXiv: 2504.00147 [cs.CL]. URL: https://arxiv.org/abs/2504.00147.
