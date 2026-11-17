---
tags: [reading-list, llama-cpp, local-models]
---

# Reading list: running models locally

Ordered from "read first" to "only if curious". All links checked 2026-10-15. Concepts: [[Running Models Locally - Concepts]].

## Start here (most useful for the project)
- [ ] Hugging Face, "How to generate text": temperature, top-k and top-p, with pictures. https://huggingface.co/blog/how-to-generate
- [ ] llama-server README: every flag and endpoint. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- [ ] Hugging Face, chat templates. https://huggingface.co/docs/transformers/main/en/chat_templating
- [ ] llama.cpp README. https://github.com/ggml-org/llama.cpp

## How models work inside (visual, beginner-friendly)
- [ ] Andrej Karpathy, "Intro to Large Language Models" (1-hour talk). https://www.youtube.com/watch?v=zjkBMFhNj_g
- [ ] Jay Alammar, "The Illustrated Transformer". https://jalammar.github.io/illustrated-transformer/
- [ ] Jay Alammar, "The Illustrated GPT-2" (writing one token at a time). https://jalammar.github.io/illustrated-gpt2/
- [ ] Hugging Face, KV cache explained. https://huggingface.co/blog/kv-cache
- [ ] Andrej Karpathy, "Let's build the GPT Tokenizer" (video, long). https://www.youtube.com/watch?v=zduSFxRajkE

## File formats and shrinking models
- [ ] Maarten Grootendorst, "A Visual Guide to Quantization". https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
- [ ] Hugging Face, GGUF overview. https://huggingface.co/docs/hub/gguf
- [ ] llama.cpp quantize README (table of formats). https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- [ ] GGUF specification (deep reference only). https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Sampling and diversity (closest to our research)
- [ ] Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration" (where top-p comes from). https://arxiv.org/abs/1904.09751
- [ ] Nguyen et al. 2024, min-p sampling. https://arxiv.org/abs/2407.01082
- [ ] Thinking Machines, "Defeating Nondeterminism in LLM Inference" (why the same seed can give different outputs). https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

## Tools
- [ ] MIT "The Missing Semester" (terminal, shell, git). https://missing.csail.mit.edu/
- [ ] uv docs. https://docs.astral.sh/uv/
- [ ] llama.cpp build guide. https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- [ ] Hugging Face Hub download guide. https://huggingface.co/docs/huggingface_hub/guides/download

## Only if we get a GPU machine
- [ ] Kwon et al. 2023, vLLM / PagedAttention. https://arxiv.org/abs/2309.06180
- [ ] mlx-lm (Apple's alternative). https://github.com/ml-explore/mlx-lm · Ollama: https://ollama.com

## OLMo 3 itself
- [ ] OLMo 3 tech report. https://arxiv.org/abs/2512.13961 (notes: [[OLMo 3 Think - Overview]])
