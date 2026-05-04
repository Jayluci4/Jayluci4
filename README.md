# Jayant Lohia
**Systems Researcher | Hardware-Aware Optimization | AI from First Principles**

I focus on **numerical stability** in low-precision inference and building **reference implementations** of modern AI architectures from scratch (NumPy-only).

---

## ⚡ Featured Research: NOVA 
**[NOVA: Rational Winograd Transforms for FP16/INT8 Stability](https://github.com/Jayluci4/winograd-fp16-stability-analysis)**

Standard Winograd convolutions ($F(6,3)$) are numerically unstable in FP16, causing accuracy to collapse to random chance on deep networks.
* **The Problem:** The condition number of the Cook-Toom transform matrix explodes ($\kappa > 10^5$).
* **My Solution:** I developed a method to discover **rational coefficients** (e.g., $\pm 5/6$) using Evolution Strategies.
* **The Result:** Reduced condition number by **400x**, restoring VGG16 accuracy from 4.7% $\to$ 77.5% in pure FP16.

<div align="center">
  <img src="https://github.com/Jayluci4/winograd-fp16-stability-analysis/blob/master/images/accuracy-analysis.png?raw=true" alt="NOVA Accuracy Recovery VGG16" width="100%" />
</div>

---

## 📚 Educational Architectures: "AI From First Principles"
I build clean-room, NumPy-only implementations of state-of-the-art architectures to demonstrate how they work at the mathematical level. No black boxes.

**15 Repositories | ~5600 Lines of Code | Verified Implementation**

### Part 1: The Modern Stack (2020-2025)
| Repo | Concept | Architecture Detail |
| :--- | :--- | :--- |
| **[`micro-instruct`](#)** | **LLM Training** | Full instruction-tuning pipeline (RoPE, RMSNorm) |
| **[`micro-transformer`](#)** | **Transformers** | The GPT-architecture implemented without PyTorch |
| **[`micro-attention`](#)** | **Attention** | Multi-head self-attention vectorized manually |
| **[`micro-diffusion`](#)** | **GenAI** | DDPM and Stable Diffusion internals |
| **[`micro-rlhf`](#)** | **Alignment** | PPO and Reward Modeling logic |
| **[`micro-lora`](#)** | **Fine-Tuning** | Low-Rank Adaptation matrix math |

### Part 2: Foundational Components
| Repo | Concept | Key Implementation |
| :--- | :--- | :--- |
| **[`micro-lstm`](#)** | **Gating** | Manual backprop through time (BPTT) |
| **[`micro-seq2seq`](#)** | **Translation** | Encoder-Decoder with attention |
| **[`micro-embedding`](#)** | **Word2Vec** | Skip-gram and negative sampling |

---

## 🛠️ Systems & Production Engineering
Beyond educational code, I build autonomous systems and optimization tools.

* **[mandrake-agent](#):** Intelligent agent framework for autonomous task execution.
* **[kg-chatbot](#):** Integrating Knowledge Graphs with LLM reasoning.
* **[india-grants-oracle](#):** RAG system for analyzing government documentation.

---

## 📄 Foundational Papers Implemented
My implementations are grounded in the original literature:
* *Attention Is All You Need* (Vaswani et al., 2017)
* *LoRA: Low-Rank Adaptation* (Hu et al., 2021)
* *Denoising Diffusion Probabilistic Models* (Ho et al., 2020)
* *LSTM* (Hochreiter & Schmidhuber, 1997)

---

## 📬 Contact
I am currently open to **Research Engineering** and **Systems Optimization** roles (Remote/India).

* **Email:** jayantlohia16@gmail.com
* **Research:** [arXiv:2512.18453](https://arxiv.org/abs/2512.18453)
* **Focus:** CUDA Kernels, Winograd Optimization, Edge Inference, Transformer Architecture.
