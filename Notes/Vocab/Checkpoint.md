---
tags: [vocab]
aliases: [checkpoint, checkpoints, revision]
---
# Checkpoint

**In one line:** a saved snapshot of a model's [[Parameters|weights]] at some point during training.

**Plain explanation:** Like a save point in a video game. Ai2 released the model as it was right after SFT, right after DPO, and after RLVR (the final model). Each one can be downloaded and run on its own.

**In our project:** the three checkpoints are the whole experiment. Same prompts, same settings, three snapshots, so any difference in Core rate comes from the training stage in between.

| Stage | Hugging Face repo |
|---|---|
| sft | `allenai/Olmo-3-7B-Think-SFT` |
| dpo | `allenai/Olmo-3-7B-Think-DPO` |
| rlvr | `allenai/Olmo-3-7B-Think` |

A **revision** is a specific version (a git commit hash) of the repo on Hugging Face. We pin revisions in `helper_scripts/download_models.py` so the files can't silently change under us.

**Related:** [[Post-training]], [[GGUF and Quantization]]
