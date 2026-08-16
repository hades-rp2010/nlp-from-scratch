# NLP From Scratch

Building core Natural Language Processing algorithms from first principles in PyTorch.

These are implementations done from scratch , without using any high-level libraries or pre-trained models. The goal is to understand the underlying mechanics of NLP algorithms and architectures. The implementations are not optimized for production use but are designed to be educational and illustrative.


---

## Roadmap

- [x] **`embeddings/` — Word Embeddings**
  - [x] CBOW (Continuous Bag-of-Words)
  - [x] Skip-gram (Word2Vec)
  - [ ] GloVe (Global Vector Matrix Factorization)
  - [ ] FastText (Subword N-gram Embeddings)
- [ ] **`sequence_models/` — Sequential Architectures**
  - [ ] RNN (Recurrent Neural Networks from Scratch)
  - [ ] LSTM (Long Short-Term Memory)
  - [ ] GRU (Gated Recurrent Unit)
- [ ] **`attention/` — Attention Mechanisms**
  - [ ] Seq2Seq (Sequence-to-Sequence with Bahdanau / Luong Attention)
  - [ ] Scaled Dot-Product & Multi-Head Attention
- [ ] **`transformers/` — Transformer Architectures**
  - [ ] Encoder-Only** (BERT-style Masked Language Model)
  - [ ] Decoder-Only** (GPT-style Causal Language Model)

---

## Quick Start

```bash
# Run Word2Vec (Skip-gram + Negative Sampling)
uv run python embeddings/word2vec.py
```
