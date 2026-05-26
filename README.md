# Autonomous Coding Agent with Self-Correction

A ReAct-style agentic system built with LangGraph that writes Python code, executes it in a sandboxed subprocess, reads the output, and self-corrects until all tasks pass — powered entirely by **Ollama (LLaMA 3.2, free & local)**.

## How it works

Task

 │
 
[Node 1] generate_code   ← LLaMA 3.2 writes Python code

 │
 
[Node 2] execute_code    ← subprocess runs it safely (no Docker needed)

 │
 
[Node 3] check_result    ← exact output evaluation (no LLM judge)

 │
 
 ├── PASS ──────────────────────────────────► END
 
 │
 
 └── FAIL + error context ──► back to Node 1 (self-correct)
 
                               (max 5 attempts)

The key insight: on retry, the agent feeds the **exact error and expected output** back into the prompt so the model knows precisely what went wrong — not just that it failed.

## Benchmark results

Evaluated on 25 coding tasks across 3 difficulty tiers (Easy / Medium / Hard):

| Method | Pass Rate | Notes |
|---|---|---|
| Single-pass LLaMA 3.2 (baseline) | 92% | No correction, one shot |
| Self-correcting agent | 96% | Recovers failures automatically |
| **Task failure reduction** | **50%** | Agent recovered 1 in 2 baseline failures |

Hard tasks include: balanced parentheses with stack, binary search, iterative list flattening, LRU cache with OrderedDict, Roman numeral conversion, run-length encoding, two-pointer merge sort.

## Project structure

```
coding_agent/
├── agent.py      # LangGraph state machine — 3 nodes, conditional retry loop
├── tools.py      # Code executor using subprocess + code extractor
├── prompts.py    # System prompt, generate prompt, retry prompt
└── main.py       # 25-task benchmark — baseline vs agent, exact evaluator
```

## Setup (Mac)

### 1. Install Ollama and pull the model
```bash
# Download from https://ollama.com then:
ollama pull llama3.2
ollama serve
```

### 2. Install Python dependencies
```bash
python3 -m venv agent_env
source agent_env/bin/activate
pip install langgraph langchain langchain-ollama langchain-core
```

### 3. Run the benchmark
```bash
python main.py
```

## Key design decisions

**No Docker** — subprocess with tempfile is sufficient for a portfolio benchmark and works natively on Mac with zero setup overhead.

**No LLM judge** — previous versions used an LLM to evaluate outputs which caused false failures (e.g. marking correct output wrong). Replaced with deterministic exact string matching — every task has a hardcoded expected output.

**Error-aware retry prompt** — the retry prompt tells the model exactly what the output was, what was missing, and what was expected. This is what makes self-correction actually work vs. just regenerating the same wrong code.

**Local & free** — zero API cost. Runs entirely on your Mac using Ollama.
