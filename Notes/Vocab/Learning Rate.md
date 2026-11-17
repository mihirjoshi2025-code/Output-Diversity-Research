---
tags: [vocab]
aliases: [learning rate, LR, warmup, linear decay]
---
# Learning Rate

**In one line:** how big each training step is, meaning how much the [[Parameters|weights]] change per update.

**Plain explanation:** Too big, and the model overshoots and breaks. Too small, and it barely learns. Post-training uses *tiny* learning rates because the model is already good and you only want small changes. OLMo's DPO used 8 × 10⁻⁸ (0.00000008).

- **Warmup:** start with an even smaller rate and ramp up, to avoid early shocks.
- **Linear decay:** shrink the rate steadily toward zero by the end.
- **Constant:** keep it the same throughout (OLMo's RL).

**Related:** [[Gradient]], [[Epoch]], [[Batch Size]]
