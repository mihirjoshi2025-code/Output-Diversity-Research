---
tags: [vocab]
aliases: [parameters, weights, model weights, parameter]
---
# Parameters (Weights)

**In one line:** the billions of numbers inside a model that training adjusts. They *are* the model's knowledge.

**Plain explanation:** A model is a giant math formula with billions of adjustable knobs. Each knob is a number called a parameter (or weight). At the start of training the knobs are random and the model writes gibberish. Training nudges every knob a tiny bit, over and over, until the model predicts text well. "Olmo-3-**7B**" means about 7 billion knobs.

**Analogy:** a mixing board with 7 billion sliders. Training is the sound engineer adjusting sliders until it sounds right.

**In our project:** the SFT, DPO and RLVR [[Checkpoint|checkpoints]] are the *same* model shape with *different* knob settings. Comparing them shows what each training stage changed.

**Related:** [[Checkpoint]], [[Gradient]], [[Learning Rate]], [[GGUF and Quantization]]
