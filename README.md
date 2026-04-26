# Concall Analysis Agent

A from-scratch AI agent that analyses company quarterly earnings call transcripts and produces a structured investment report — derived thesis, KPIs, red flags, Hold/Trim/Accumulate decision, and next quarter watchlist.

## What it does

Feed it a concall transcript (PDF or TXT) and a company name. It runs 8 analysis tools in sequence, orchestrated by Claude, and prints a full investment analysis to the terminal.

```bash
.venv/bin/python concall_agent.py \
  --transcript path/to/concall.pdf \
  --company "Tips Music Ltd"
```

## Why it's intentionally bare-bones

This repo exists to demonstrate one concept clearly: **how LLM-powered tool calling works**.

The agent loop mirrors the pattern from first principles —

```
LLM decides → Tool executes → Result feeds back → LLM decides again
```

Every layer is visible in plain Python: the prompt being built, the JSON the LLM returns, the tool being dispatched, the result being appended to the conversation. Nothing is hidden behind a framework.

Keeping it simple was a deliberate choice. Adding abstractions, a framework like LangChain, or a UI would have buried the concept under infrastructure. The goal was to understand what an agent *is* before reaching for tools that do it for you.

## Why no UI or Chrome extension

A UI would shift focus from the agent mechanics to frontend wiring. A Chrome extension would add a layer of browser APIs, messaging, and DOM handling that has nothing to do with tool calling.

The terminal output *is* the interface here — and it's intentional. Every `[LLM]`, `[TOOL CALL]`, and `[TOOL RESULT]` line printed is the agent loop made visible. That's the learning, not the polish.

## Structure

| File | Role |
|---|---|
| `concall_agent.py` | Config, agent loop, report formatter |
| `tools.py` | 8 analysis tools (deterministic + sub-LLM mix) |
| `prompts.py` | System prompt + focused sub-LLM prompts |
| `logger.py` | Logs every LLM call to `llm_calls.json` |
