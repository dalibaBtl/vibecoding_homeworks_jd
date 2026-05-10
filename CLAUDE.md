# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

Homework assignments for a vibecoding/LLM course, each in its own directory (`HW01_LLM/`, future `HW02_*/`, etc.). Each homework is an independent Python project managed with `uv`.

## Running homework scripts

Each homework directory uses `uv` with its own `pyproject.toml` and `uv.lock`. Run scripts from within the homework directory:

```bash
cd HW01_LLM
uv --cache-dir .uv-cache run python openai_loop_example.py
```

Environment variables are loaded from a `.env` file in the same directory as the script (copy `.env.example` to `.env` and fill in the key).

## HW01_LLM — OpenAI tool-calling loop

`openai_loop_example.py` demonstrates a manual two-turn tool-call loop with the OpenAI API (`gpt-4o`):

1. **Turn 1** — send user message + tool definitions; model responds with a `tool_calls` request.
2. **Python executes** the requested local function (`blackbox_funkce_1` = sqrt, `blackbox_funkce_2` = square).
3. **Turn 2** — append the assistant message (with `tool_calls`) and a `tool` role message (with the result), then re-call the API to get the final natural-language response.

The `tool_skill` string in the developer message acts as a "skill card" that tells the model which function to call for which intent. `DEBUG = True` prints each step of the loop.

## Dependencies

- Python ≥ 3.10
- Package manager: `uv` (lock files committed per homework)
- Runtime deps declared in each `pyproject.toml`; no global install needed beyond `uv` itself
