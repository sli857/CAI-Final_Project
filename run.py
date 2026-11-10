"""Run episodes: every task x 2 agent strategies x 2 user rules x repetitions.

Example:
  python run.py --base-url http://127.0.0.1:8000 --model meta-llama/Llama-3.1-8B-Instruct \
      --tasks tasks.jsonl --reps 5 --out results/min.jsonl

One JSON line is appended to --out as soon as an episode ends. Running the same
command again skips episodes that are already in the file, so an interrupted
run can be continued without repeating work.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

from agent import PROMPTS, make_model
from episode import run_episode
from sandbox import load_tasks

AGENTS = ("base", "clarify")
USER_RULES = ("proactive", "ondemand")
MAX_ERRORS_IN_A_ROW = 3  # stop early if the model server looks broken


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def key_of(row):
    return (row["task_id"], row["model"], row["agent"], row["user_rule"], row["rep"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", default="tasks.jsonl")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        done = {key_of(r) for r in rows if "error" not in r}  # failed episodes are run again
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    commit = git_commit()
    total = len(tasks) * len(AGENTS) * len(USER_RULES) * args.reps
    print(f"{total} episodes planned, {len(done)} already done, output: {args.out}")
    ran = errors_in_a_row = 0

    # Conditions are interleaved inside each task, so a partial run stays balanced.
    for task in tasks:
        for rep in range(args.reps):
            for agent in AGENTS:
                for user_rule in USER_RULES:
                    row = {"task_id": task["task_id"], "model": args.model, "agent": agent,
                           "user_rule": user_rule, "rep": rep, "seed": rep}
                    if key_of(row) in done:
                        continue
                    stats = {}
                    model = make_model(args.base_url, args.model, seed=rep, stats=stats)
                    try:
                        row.update(run_episode(task, model, PROMPTS[agent], user_rule, args.max_turns))
                        errors_in_a_row = 0
                    except Exception as error:  # server or network problem, not an agent failure
                        row["error"] = f"{type(error).__name__}: {error}"
                        errors_in_a_row += 1
                    row.update(kind=task["kind"], action=task["action"], stats=stats, code_commit=commit,
                               finished_at=datetime.datetime.now().isoformat(timespec="seconds"))
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    ran += 1
                    outcome = row.get("error") or f"success={row['success']} end={row['end_reason']} " \
                                                  f"turns={row['assistant_turns']}"
                    print(f"[{ran}] {task['task_id']} {agent:8} {user_rule:10} rep{rep}  {outcome}", flush=True)
                    if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                        print(f"stopping: {MAX_ERRORS_IN_A_ROW} episodes failed in a row")
                        return 1
    print(f"finished: ran {ran} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
