---
tags: [vocab]
aliases: [loss, loss function, objective]
---
# Loss

**In one line:** a single number measuring how wrong the model currently is. Training means making it smaller.

**Plain explanation:** Each training method defines "wrong" differently:
- **SFT loss:** how surprised was the model by the correct next token? (Very surprised = high loss.)
- **DPO loss:** how much does the model fail to prefer chosen over rejected?
- **RL:** roughly, how much reward did it miss?

The [[Gradient|gradient]] of the loss tells you which way to nudge each [[Parameters|parameter]] to reduce it.

**"Length-normalized DPO"** (used by OLMo): divides by answer length so long answers aren't favoured or penalized just for their length.

**Related:** [[Gradient]], [[Learning Rate]]
