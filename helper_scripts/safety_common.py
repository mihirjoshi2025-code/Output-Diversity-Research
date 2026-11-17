"""Shared input handling for the safety classifiers (guard_classify.py, jev_classify.py).

Both classifiers must see exactly the same text, so the cut-off rules live here. Limits are sized for one
llama-server slot (8,192 tokens with -c 32768 --parallel 4): Qwen3Guard's instructions take ~400 tokens, and
16K + 10K characters is at most ~7K tokens of English. In the WildChat story sample, no response is longer than
~7.5K characters; long *prompts* (roleplay setups up to ~36K) are what get cut.

A long prompt keeps its first and last halves, so both the setup and the actual request (often at the end) survive.
"""

PROMPT_CHARS = 16_000
RESPONSE_CHARS = 10_000
GAP = "\n[...]\n"


def truncate_pair(prompt: str, response: str) -> tuple[str, str, bool]:
    """Cut prompt (head + tail) and response (head) to the shared limits; the flag says whether either was cut."""
    truncated = len(prompt) > PROMPT_CHARS or len(response) > RESPONSE_CHARS
    if len(prompt) > PROMPT_CHARS:
        half = (PROMPT_CHARS - len(GAP)) // 2
        prompt = prompt[:half] + GAP + prompt[-half:]
    return prompt, response[:RESPONSE_CHARS], truncated
