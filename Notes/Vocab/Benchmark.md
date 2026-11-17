---
tags: [vocab]
aliases: [benchmark, benchmarks, eval, evaluation, AlpacaEval, IFEval, IFBench, AIME, MATH]
---
# Benchmark

**In one line:** a standard test used to score and compare models.

**The ones in the OLMo 3 notes:**
| Benchmark | What it tests | How it's scored |
|---|---|---|
| **AlpacaEval 2 LC** | General chat quality | GPT-4.1 [[LM Judge\|judge]] compares with a reference model. "LC" = length-controlled, so long answers aren't favoured. |
| **IFEval** | Following exact instructions ("use 3 bullets") | Automatic checks |
| **IFBench** | Harder, newer instruction constraints | Automatic checks |
| **AIME** | Hard competition math | Correct final answer |
| **MATH** | School/competition math | Correct final answer |
| **Safety** | Refusing harmful requests, etc. | Mix |

**Why it matters for us:** none of these measure story *diversity*. A model can climb every benchmark while its stories get more repetitive. That gap is part of the paper's point.

**Related:** [[Decontamination]], [[Pass@k]], [[Instruction Following]]
