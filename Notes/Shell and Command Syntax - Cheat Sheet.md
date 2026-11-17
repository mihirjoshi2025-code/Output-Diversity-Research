---
tags: [setup, shell, cheat-sheet, week1]
---

# Shell and command syntax: cheat sheet

> [!summary] The 6 things that trip people up
> 1. **Spaces split things.** Our folder name has spaces, so write `~/"AI Research - Cornell"`, with the quotes after the `~`.
> 2. **`\` at the end of a line** = "command continues on the next line." If you join the lines into one, **delete the `\`**.
> 3. **Ctrl-C** stops whatever is running in that terminal.
> 4. **`>` overwrites a file**, `>>` adds to the end.
> 5. **`source .venv/bin/activate`** only matters for converting models. Nothing else needs it.
> 6. Any program + **`--help`** lists its options.

See also: [[Running Models Locally - Concepts]] · [[Servers and APIs - How llama-server Works]]

---

## Reading a command

```
uv run --no-project helper_scripts/generate_stories.py instruct-sft --out stories.jsonl --workers 4
└┬┘ └┬┘ └────┬────┘ └───────────────┬───────────────┘ └────┬─────┘ └─┬─┘ └─────┬─────┘ └──┬───┘
program  sub-   flag            the script               plain     flag   its value   flag + value
       command                                          argument
```

- **Flag** = an option starting with `-` or `--`. Some need a value (`-c 8192`); some are on/off (`--jinja`).
- **Plain argument** = no dash. Its meaning depends on its position.

---

## Paths (where files are)

| You write | It means |
|---|---|
| `~` | Your home folder |
| `.` | The folder you're in |
| `..` | The folder above |
| `helper_scripts/x.py` | Starting from where you are now |
| `/Users/...` | Starting from the very top |
| `*.jsonl` | Every file ending in `.jsonl` |

`pwd` = where am I · `cd folder` = go there · `ls` = what's here · `ls -la` = with sizes and hidden files

---

## Quotes

| | What happens | Use it for |
|---|---|---|
| `'single'` | Everything is literal | JSON in `curl -d '{...}'` |
| `"double"` | Literal, except `$things` get filled in | Paths with spaces |
| none | Spaces split; `*`, `?`, `~` get expanded | Normal stuff |

A URL with `?` in it needs quotes, or zsh says `no matches found`.

---

## The `\` line break

```bash
llama-server \
    -m model.gguf \
    --port 8080
```
is the same as
```bash
llama-server -m model.gguf --port 8080
```
⚠️ `llama-server \ -m model.gguf` on **one line** is broken: the `\` turns the next space into part of a word. (That was our `unrecognized arguments:  --outtype` error.)

---

## Connecting commands

| Symbol | Meaning |
|---|---|
| `a && b` | Do `b` only if `a` worked |
| `a ; b` | Do `b` after `a`, no matter what |
| `a \| b` | Feed `a`'s output into `b` |
| `a > file` | Save output to a file (**replaces** it) |
| `a >> file` | Add output to the end of a file |
| `2>&1` | Include error messages with the normal output |
| `a \| tee file` | Show it **and** save it |
| `# words` | A comment; ignored |

---

## Running programs

| Do this | To |
|---|---|
| **Ctrl-C** | Stop it |
| ↑ arrow | Bring back the last command |
| Ctrl-R | Search old commands |
| `ps aux \| grep llama` | See what llama things are running |
| `kill 12345` | Stop process number 12345 |
| `lsof -i :8080` | Who's using port 8080 |
| `caffeinate -i <command>` | Keep the Mac awake while it runs |

---

## Python boxes (venvs)

- `source .venv/bin/activate` = use this project's Python. Your prompt shows `(.venv)`.
- `deactivate` = go back to normal.
- `python --version` = check which Python (the llama.cpp one must be **3.12**).
- `uv run --no-project script.py` = run a script; uv sorts out its packages. No activating needed.

---

## Handy commands

| Command | Does |
|---|---|
| `wc -l file.jsonl` | Count lines (= number of stories) |
| `head -n 5 file` | First 5 lines |
| `tail -f file.log` | Watch a log as it grows (Ctrl-C to stop watching) |
| `du -sh folder` | How big a folder is |
| `df -h` | How much disk is free |
| `grep word file` | Find lines containing a word |
| `python3 -m json.tool file.json` | Print JSON nicely |
| `git status` | What changed in the project |

---

## File types

- **JSON**: one structured blob `{...}`. Server answers.
- **JSONL**: one JSON blob **per line**. Our story files (`wc -l` = number of stories).
- **CSV**: a table with commas.
- **Markdown (`.md`)**: these notes.
