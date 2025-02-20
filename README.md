# HyperParamsLog



> **Important:**  
> • “Hidden/Embedding Dimension” here is taken as the input embedding size (typically equal to the hidden size).  
> • Learning rate and batch size values are approximate and inferred from related literature or scaling studies.  
> • “Undisc.” indicates details that are not publicly disclosed.

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
| **Chinchilla (70B)**      | ~70B           | ~? (estimated)     | ~7,000*                        | Undisc.             | ~2048 tokens        | ~3×10⁻⁴* (compute‐opt.) | Not disclosed            |
| **T5 (11B)**              | ~11B           | Encoder: 24<br>Decoder: 24 | ~1024                   | ~16                 | ~512 tokens         | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **CodeGen (16B)**         | ~16B           | ~48                | ~4096                          | ~32                 | ~2048 tokens        | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **Falcon 40B**            | ~40B           | ~40                | ~5120*                         | ~40*                | ~2048 tokens        | ~(1–3)×10⁻⁴*           | Not disclosed            |
| **GPT‑4**                 | Undisc.        | Undisc.            | Undisc.                        | Undisc.             | ~8K–32K tokens*     | Undisc.                | Undisc.                  |

*Notes on additional models:  
• **BLOOM:** Developed as a multilingual open‐source LLM with architecture similar to GPT‑3.  
• **Chinchilla:** A compute‑optimal model from DeepMind, tuned with far more tokens relative to its parameter count.  
• **T5:** A text‑to‑text model with an encoder–decoder architecture; note that its maximum input length is lower than decoder‑only models.  
• **CodeGen:** Designed specifically for code generation tasks.  
• **Falcon 40B:** An open‑source model optimized for both performance and efficiency.  
• **GPT‑4:** A next‑generation model with many details remaining proprietary.

These hyperparameters are based on publicly available literature and scaling analyses. Actual training details (including learning rate schedules, batch sizes, and optimizer tweaks) are typically fine‑tuned based on available compute budgets and task requirements.
