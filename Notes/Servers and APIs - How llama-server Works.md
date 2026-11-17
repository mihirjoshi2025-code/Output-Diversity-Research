---
tags: [setup, llama-cpp, servers, http, week1]
---

# Servers and APIs: how llama-server works

> [!summary] The whole note in 4 lines
> - **llama-server** loads the model once and then waits for questions. That's the **server**.
> - Our script (or `curl`) sends it a question and gets an answer back. That's the **client**.
> - They talk in **HTTP** (like a website), and the messages are written in **JSON**.
> - Every question is separate: the server remembers nothing between them.

See also: [[Parallel Generation - 4 Stories at Once]] · [[Running Models Locally - Concepts]] · [[Shell and Command Syntax - Cheat Sheet]]

---

## The picture

```
 Terminal 1: SERVER                     Terminal 2: CLIENT
┌──────────────────────┐  question   ┌──────────────────────┐
│ llama-server         │ ◄────────── │ curl or our script   │
│ model loaded, waiting│             │                      │
│ on port 8080         │ ──────────► │ saves the answer     │
└──────────────────────┘   answer    └──────────────────────┘
            both running on your Mac
```

**Restaurant version:** the server is a kitchen that's already fired up (the model is loaded). The client is a waiter carrying orders in and plates out. Firing up the kitchen is slow (loading the model), so you do it once and then send in lots of orders.

---

## Five words to know

| Word | Plain meaning | Our example |
|---|---|---|
| **localhost** | "This same computer" | `http://localhost:8080` |
| **Port** | A numbered door. Each server listens on its own door. | `8080`. Only one program per door, or you get "address already in use". |
| **HTTP** | The language client and server speak: one request, one response | Same thing your browser uses |
| **Endpoint** | The specific address for one job | `/v1/chat/completions` = "write a reply" |
| **JSON** | Text format for the messages. `{}` = named fields, `[]` = list | `{"role": "user", "content": "Write a story."}` |

**"OpenAI-compatible"** just means llama-server accepts the same message shape as ChatGPT's API, so lots of tools work with it.

---

## What we send

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a fiction writer. ..."},
      {"role": "user",   "content": "Write a story."}
    ],
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "seed": 1
  }'
```

- **`messages`**: the conversation. `system` = instructions on how to behave. `user` = the actual prompt.
- **`max_tokens`**: the longest the answer can be.
- **`temperature`, `top_p`, `seed`**: dice settings (see [[Parallel Generation - 4 Stories at Once]]). **Always send these.** If you leave them out, the server uses its own defaults, which you'd then have to go and find.

`curl` pieces: `-s` = quiet, `-H` = add a header line, `-d` = the message body.

---

## What comes back (the parts we use)

```json
{
  "choices": [{
    "message": {
      "content": "The lighthouse keeper...",
      "reasoning_content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage":   {"completion_tokens": 700},
  "timings": {"predicted_per_second": 11}
}
```

| Field | Meaning |
|---|---|
| `content` | **The story.** The only thing we score. |
| `reasoning_content` | The `<think>` part (Think models only). Saved, never scored. |
| `finish_reason` | `stop` = finished on its own ✅. `length` = got cut off at `max_tokens` ⚠️ |
| `completion_tokens` | How long the answer was |
| `predicted_per_second` | Speed |

---

## What happens inside the server (one request)

1. Read the JSON.
2. Turn the messages into the exact text format the model was trained on (the **chat template**, which is why we use `--jinja`).
3. Chop the text into tokens.
4. Read the whole prompt at once (fast).
5. Write the answer one token at a time (slow) until the model says "done" or it hits `max_tokens`.
6. Send the JSON back.

---

## Running the server: the practical stuff

- It **takes over its terminal** and prints a log. Leave that window open and use a second one for everything else.
- **Stop it:** Ctrl-C in its window.
- **Is it ready?** `curl http://localhost:8080/health` says `ok`. Before that, requests get a `503`.
- **Restart it when you change** the model file, `-c`, `--parallel`, `--jinja`, or `--chat-template-file`. Dice settings go in each request, so those don't need a restart.
- **Swapping the model file on disk does nothing** to a running server. It keeps what it loaded. Restart it.
- **What's running?** `ps aux | grep llama-server`. **Who's on port 8080?** `lsof -i :8080`.

### Server flags we use
| Flag | Meaning |
|---|---|
| `-m file.gguf` | Which model |
| `--jinja` | Use the model's own chat template |
| `--chat-template-file x.jinja` | Use this template file instead (needed for the Instruct models) |
| `-c 16384` | Total memory for tokens, shared across slots |
| `--parallel 4` | Write 4 stories at once |
| `--port 8080` | Which door |
| `2>&1 \| tee file.log` | Show the log and save it to a file |

---

## When things go wrong

| You see | It usually means |
|---|---|
| `Connection refused` | Server isn't running, or wrong port |
| `503` | Still loading. Wait a bit. |
| `400` | The server didn't like the request. **Read the message.** Ours was the Instruct chat-template bug (fix: `--chat-template-file chat_templates/olmo3-instruct-fixed.jinja`). |
| `Address already in use` | Something else is on that port |
| Output is `@@@@` or gibberish | Broken model file. Ours was a bad conversion (see `CLAUDE.md`, "Conversion gotcha"). |
| `finish_reason: "length"` | Hit `max_tokens` |
| Really slow | Not enough memory (quit other apps, especially the Llama desktop app) |

---

## How our script fits in

`helper_scripts/generate_stories.py` is just a client that does the `curl` above 500 times, then:
- checks each answer came back OK
- saves one line per story with the prompt, settings, seed, story, `finish_reason`, and so on

Minimal Python version of one request:
```python
import requests
r = requests.post("http://localhost:8080/v1/chat/completions", json={
    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
    "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "seed": 1,
})
r.raise_for_status()                          # stop if it wasn't OK
story = r.json()["choices"][0]["message"]["content"]
```
