"""Checks for agent.py and run.py without a GPU. Run:  python3 test_run.py

Starts a tiny fake model server on this machine. The fake "model" follows fixed
rules (list, ask which appointment, cancel the one the user names). It tests
the HTTP client, the runner, and resuming. Its output is NOT an experimental result.
"""
import http.server
import json
import os
import tempfile
import threading

import run

requests_seen = []


def fake_model(messages):
    def text(content):
        return {"role": "assistant", "content": content}

    def call(name, **args):
        return {"role": "assistant", "content": None, "extra_field": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    tool_results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if not tool_results:
        return call("list_appointments")
    if any(r.get("ok") for r in tool_results):
        return text("Done.")
    said = " ".join(m["content"] for m in messages[2:] if m["role"] == "user")
    for a in tool_results[0]["appointments"]:
        if a["day"] in said and a["time"] in said:
            return call("cancel_appointment", appointment_id=a["id"])
    return text("Which appointment do you mean?")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests_seen.append(body)
        if body["model"] == "broken":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"server exploded")
            return
        reply = json.dumps({"choices": [{"message": fake_model(body["messages"])}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{server.server_port}"
here = os.path.dirname(os.path.abspath(__file__))

with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "results", "test.jsonl")
    common = ["--base-url", base_url, "--tasks", os.path.join(here, "tasks.jsonl"), "--reps", "2", "--out", out]

    # 1 task x 2 agents x 2 user rules x 2 reps = 8 episodes
    assert run.main(common + ["--model", "fake"]) == 0
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 8
    assert {run.key_of(r) for r in rows} == {("min-001", "fake", a, u, rep) for a in run.AGENTS
                                             for u in run.USER_RULES for rep in (0, 1)}
    assert all(r["success"] and r["target_source_at_first_write"] == "asked" for r in rows)
    assert all(r["stats"]["model_calls"] == r["assistant_turns"] == 4 for r in rows)
    assert all(r["stats"]["prompt_tokens"] == 40 for r in rows)

    # The request carries the tools and the seed, and no temperature.
    first = requests_seen[0]
    assert len(first["tools"]) == 3 and first["seed"] == 0 and "temperature" not in first
    # The two strategies differ only in the system prompt.
    prompts = {r["agent"]: r["messages"][0]["content"] for r in rows}
    assert prompts["clarify"].startswith(prompts["base"]) and prompts["clarify"] != prompts["base"]
    # Non-standard fields from the server are not sent back in the history.
    assert all("extra_field" not in m for body in requests_seen for m in body["messages"])

    # Resuming: nothing is run twice.
    seen = len(requests_seen)
    assert run.main(common + ["--model", "fake"]) == 0
    assert len(requests_seen) == seen and len(open(out).readlines()) == 8

    # A broken server: episodes are recorded as errors, and the run stops after 3 in a row.
    assert run.main(common + ["--model", "broken"]) == 1
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 11 and all("HTTP 500" in r["error"] for r in rows[8:])

server.shutdown()
print("all runner checks passed")
