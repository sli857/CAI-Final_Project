# Disclosure Timing and Agent Clarification

Code for the ECE 6256 / CS 6604 (Fall 2026) final project
"Does User Disclosure Timing Change the Measured Benefit of Agent Clarification?"

Status: work in progress. The sandbox, the scripted user, and the conversation
loop exist and are tested with rule-based stub agents. No experiment with a
real model has been run yet, so there are no results in this repository.

## Layout

| File | Purpose |
|---|---|
| `sandbox.py` | Appointment table, three tools, and the goal check |
| `user_sim.py` | Scripted user with the proactive and on-demand rules |
| `episode.py` | One conversation between an agent and the scripted user |
| `agent.py` | The two system prompts (verbatim), tool definitions, and the model call |
| `run.py` | Runs all conditions and appends one JSON line per episode |
| `tasks.jsonl` | Tasks, one JSON object per line |
| `test_sandbox.py`, `test_episode.py`, `test_run.py` | Checks that run without any model |
| `pace/` | Slurm scripts for PACE-ICE |

## Setup

Python 3.12 or newer. The code uses only the standard library, so there is
nothing to install for local work.

```bash
python3 test_sandbox.py
python3 test_episode.py
python3 test_run.py
```

Each script prints a final line that says all checks passed.

## Running on PACE-ICE

Models are served with vLLM on PACE-ICE. `pace/hello_vllm.sbatch` starts a
server, sends two tool-calling requests, and exits. It is a connectivity
check, not an experiment.

```bash
sbatch pace/hello_vllm.sbatch
```

`pace/run_min.sbatch` runs the minimal experiment: one scenario, two agent
strategies, two user rules, five repetitions (20 episodes). Results are
appended to `results/min-llama3.1-8b.jsonl`.

```bash
sbatch pace/run_min.sbatch
```

## AI use

The code in this repository was drafted with Claude Code (Anthropic). Team
members review and test the code and are responsible for it.
