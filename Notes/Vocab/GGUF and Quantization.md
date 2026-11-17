---
tags: [vocab, tooling]
aliases: [GGUF, quantization, Q8_0, quantized, llama.cpp, safetensors]
---
# GGUF and Quantization (Q8_0)

**In one line:** a compressed file format that lets big models run on a laptop.

**Plain explanation:**
- Models are released as **safetensors** files storing each [[Parameters|parameter]] as a 16-bit number (~14.6 GB for OLMo 7B).
- **Quantization** stores each number with fewer bits. **Q8_0** = 8 bits, so it's about half the size (~8 GB) with nearly no quality loss.
- **GGUF** is the file format used by **llama.cpp**, the program that runs the model on your Mac.

**Why it matters for us:** the paper's OLMo stories were made with a Q8_0 GGUF in llama.cpp. All three checkpoints must use the **same** format and program, so any differences come from training, not setup.

**Related:** [[Checkpoint]]
