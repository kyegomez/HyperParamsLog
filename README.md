
# Large Language Models – Hyperparameter Comparison & Deep Dive

This repository provides a comparative overview of several well‑known large language models (LLMs) along with a detailed explanation of their key hyperparameters. The goal is to serve as a reference for researchers and practitioners who wish to understand how architectural and training hyperparameters influence LLM performance.

## Table of Contents

- [Overview](#overview)
- [Comparative Hyperparameter Table](#comparative-hyperparameter-table)
- [In‑Depth Explanation of Hyperparameters](#in‑depth-explanation-of-hyperparameters)
  - [Model Size & Layers](#model-size--layers)
  - [Hidden/Embedding Dimension](#hiddenembedding-dimension)
  - [Attention Heads](#attention-heads)
  - [Context Window](#context-window)
  - [Learning Rate & Schedules](#learning-rate--schedules)
  - [Batch Size](#batch-size)
- [Additional Considerations](#additional-considerations)
- [References](#references)

## Overview

Large language models have rapidly evolved over the last few years, with parameter counts ranging from a few billion to hundreds of billions. Key hyperparameters such as the number of layers, hidden dimension size, attention heads, and training-specific parameters (learning rate and batch size) play a critical role in determining a model’s capacity, efficiency, and eventual performance.

This document provides a snapshot of hyperparameters for models such as GPT‑3, LLaMA, PaLM, Gopher, OPT, BLOOM, Chinchilla, T5, CodeGen, Falcon 40B, and GPT‑4. Where details are not fully disclosed, approximate or inferred values are provided.

## Comparative Hyperparameter Table

> **Note:**  
> - The **Hidden/Embedding Dimension** represents the size of the input embedding, which is typically equivalent to the model’s hidden size.  
> - Learning rate and batch size values are approximations, often inferred from scaling studies and publicly available literature.  
> - “Undisc.” indicates details that remain undisclosed for proprietary models.

| **Model**                 | **Parameters** | **Layers**         | **Hidden/Embedding Dimension** | **Attention Heads** | **Context Window**  | **Learning Rate**      | **Batch Size (tokens)**  |
|---------------------------|----------------|--------------------|--------------------------------|---------------------|---------------------|------------------------|--------------------------|
| **GPT‑3**                 | ~175B          | 96                 | 12,288                         | 96                  | ~2048 tokens        | ~3×10⁻⁴                | ~3.2M (approx.)          |
| **GPT‑2 XL**              | ~1.5B          | 48                 | 1,600                          | ~25                 | ~1024 tokens        | ~1×10⁻⁴                | Not disclosed            |
| **LLaMA (65B)**           | ~65B           | 80                 | 8,192                          | 64                  | ~2048 tokens        | ~1.5×10⁻⁴              | 4M                       |
| **PaLM**                  | ~540B          | 64                 | 18,432                         | 128                 | ~2048 tokens        | ~3×10⁻⁴*               | Not disclosed            |
| **Gopher**                | ~280B          | 96                 | 12,288                         | 96                  | ~2048 tokens        | ~3×10⁻⁴*               | Not disclosed            |
| **OPT (175B)**            | ~175B          | 96                 | 12,288                         | 96                  | ~2048 tokens        | ~3×10⁻⁴*               | Not disclosed            |
| **Megatron‑Turing NLG**   | ~530B          | ~64                | ~14,336                        | ~56*                | ~2048 tokens        | ~3×10⁻⁴*               | Not disclosed            |
| **Jurassic‑1 Jumbo**      | ~178B          | Undisc.            | Undisc.                        | Undisc.             | ~2048 tokens        | Undisc.                | Undisc.                  |
| **BLOOM (176B)**          | ~176B          | ~70                | ~12,288*                       | ~112*               | ~2048 tokens        | ~3×10⁻⁴*               | Not disclosed            |
| **Chinchilla (70B)**      | ~70B           | (estimated)        | ~7,000*                        | Undisc.             | ~2048 tokens        | ~3×10⁻⁴* (compute‑opt.) | Not disclosed            |
| **T5 (11B)**              | ~11B           | Encoder: 24<br>Decoder: 24 | ~1024                   | ~16                 | ~512 tokens         | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **CodeGen (16B)**         | ~16B           | ~48                | ~4096                          | ~32                 | ~2048 tokens        | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **Falcon 40B**            | ~40B           | ~40                | ~5120*                         | ~40*                | ~2048 tokens        | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **GPT‑4**                 | Undisc.        | Undisc.            | Undisc.                        | Undisc.             | ~8K–32K tokens*     | Undisc.                | Undisc.                  |

*Values marked with an asterisk (*) are estimated or based on compute‑optimal scaling studies.

## In‑Depth Explanation of Hyperparameters

### Model Size & Layers

- **Parameters:**  
  The total number of model parameters determines the model’s capacity to store knowledge and perform complex reasoning. Models like GPT‑3 with 175 billion parameters have an enormous capacity compared to earlier models.
  
- **Layers:**  
  The number of layers (or transformer blocks) dictates the depth of the model. Each additional layer adds more complexity to the model’s ability to capture hierarchical representations in the data. A higher number of layers usually improves performance, although it also increases computational cost and training complexity.

### Hidden/Embedding Dimension

- **Definition:**  
  This value represents the size of the token embeddings as well as the dimensionality of the internal hidden states. It is critical because it influences how much information each token can encode.
  
- **Impact:**  
  A larger hidden dimension allows the model to capture more nuanced semantic and syntactic features. However, increasing this dimension also significantly raises the model’s overall parameter count and the computational resources required for training.

### Attention Heads

- **Definition:**  
  Attention heads allow the transformer model to attend to different parts of the input simultaneously. The number of heads determines how many parallel attention mechanisms the model uses.
  
- **Impact:**  
  More heads generally improve the model’s ability to capture various aspects of the input context concurrently. However, beyond a certain point, additional heads may yield diminishing returns while increasing complexity.

### Context Window

- **Definition:**  
  The context window is the maximum sequence length (in tokens) that the model can process at one time. It determines how much context the model can consider when making predictions.
  
- **Impact:**  
  A larger context window enables the model to capture long-range dependencies in text, which is critical for tasks like long-form generation or document-level understanding. It also affects the memory footprint during inference and training.

### Learning Rate & Schedules

- **Learning Rate (LR):**  
  The learning rate controls how much the model weights are updated during each training step. It is a critical hyperparameter that affects both the convergence speed and the stability of the training process.
  
- **LR Schedules:**  
  Models typically use a dynamic learning rate that changes over the course of training. Common strategies include:
  
  - **Warmup Phase:**  
    The learning rate starts small and gradually increases to a maximum value to avoid instability in the early training stages.
  
  - **Cosine Decay / Stable Decay:**  
    After the warmup, the learning rate gradually decays following a cosine or stable-decay schedule. This allows the model to fine-tune its parameters as training nears convergence.
  
  - **Cyclical Schedules:**  
    Some approaches use cyclical cosine schedules or warmup-stable-decay cycles, where the learning rate periodically increases again to potentially escape local minima.

- **Impact:**  
  Selecting the right learning rate and schedule is crucial for ensuring efficient convergence. A rate that is too high may cause the model to diverge, while a rate that is too low can slow down training significantly.

### Batch Size

- **Definition:**  
  Batch size (often measured in the number of tokens processed per update) determines how many training examples are processed simultaneously.
  
- **Impact:**  
  A larger batch size can stabilize training and improve hardware utilization by providing better gradient estimates. However, excessively large batches may require careful adjustment of the learning rate (as indicated by scaling laws) and can increase memory demands. For many LLMs, the batch size is dynamically increased during training to maximize efficiency.

## Additional Considerations

- **Optimizer Choice:**  
  Most of these models use variants of the Adam or AdamW optimizers. Fine-tuning the optimizer’s hyperparameters (such as β₁, β₂, and weight decay) is essential for achieving stable training.

- **Scaling Laws:**  
  Empirical scaling laws indicate that increasing model size, training data, and compute jointly improves performance. However, balancing these factors requires careful hyperparameter tuning, particularly for learning rate and batch size.

- **Compute Budget:**  
  Training LLMs is computationally intensive. As such, many hyperparameters are chosen based on available resources and may be adjusted dynamically (e.g., using population-based training or Bayesian optimization) to maximize efficiency.

## References

For further reading on LLM hyperparameters and optimization strategies, consider the following resources:

- [Hyperparameter Optimization For LLMs: Advanced Strategies](https://neptune.ai/blog/hyperparameter-optimization-for-llms) – An in-depth discussion on selecting optimal hyperparameters.
- [A Comprehensive Overview of Large Language Models (arXiv)](https://arxiv.org/abs/2307.06435) – Survey paper covering LLM architectures, scaling laws, and training techniques.
- [LLaMA: Open and Efficient Foundation Language Models (PDF)](https://parsa.epfl.ch/course-info/cs723/papers/llama.pdf) – Paper providing details on model architecture and optimization hyperparameters.

---

This README.md is intended to serve as both a quick reference and a deeper guide for understanding the critical aspects of LLM hyperparameters. Adjustments to these values are often made based on empirical evidence and available compute, making hyperparameter tuning an essential part of model development and research.
