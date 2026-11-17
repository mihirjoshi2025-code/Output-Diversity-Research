---
tags: [vocab]
aliases: [gradient, gradients, gradient descent]
---
# Gradient

**In one line:** the direction in which to nudge every [[Parameters|parameter]] so the [[Loss|loss]] goes down.

**Analogy:** you are blindfolded on a hillside and want to reach the valley. The gradient is the feel of the slope under your feet: it tells you which way is downhill. Each training step takes a small step that way (**gradient descent**). The [[Learning Rate|learning rate]] is the step size.

**Why it comes up:** in [[GRPO]], if all 8 answers to a prompt get the same reward, there is nothing to compare, so the gradient is zero and the model learns nothing from that prompt. OLMo drops such prompts.

**Related:** [[Loss]], [[Learning Rate]]
