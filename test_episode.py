"""Checks for user_sim.py and episode.py. Run:  python3 experiment/test_episode.py

The "agents" here are rule-based stubs that stand in for a model. Their only job
is to test the conversation loop and the metrics. Their output is NOT an
experimental result.
"""
import json
import os

from episode import run_episode
from sandbox import load_tasks
from user_sim import is_target_question

task = load_tasks(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.jsonl"))[0]


# ---- helpers for writing stubs -------------------------------------------
def say(text):
    return {"role": "assistant", "content": text}


def call(tool, arguments="{}", **args):
    arguments = json.dumps(args) if args else arguments
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}]}


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def listed(messages):
    """Appointments from an earlier list call, or None if the stub has not listed yet."""
    return next((r["appointments"] for r in tool_results(messages) if "appointments" in r), None)


def wrote(messages):
    return any(r.get("ok") for r in tool_results(messages))


def named(messages, appointments):
    """The appointment whose day and time appear in a later user message, if any."""
    said = " ".join(m["content"] for m in messages[2:] if m["role"] == "user")
    return next((a for a in appointments if a["day"] in said and a["time"] in said), None)


# ---- stubs ----------------------------------------------------------------
def guess_first(messages):
    """List, then cancel the first appointment without talking to the user."""
    if listed(messages) is None:
        return call("list_appointments")
    if wrote(messages):
        return say("Done.")
    return call("cancel_appointment", appointment_id=listed(messages)[0]["id"])


def ask_first(messages):
    """List, ask which appointment, then cancel the one the user names."""
    appointments = listed(messages)
    if appointments is None:
        return call("list_appointments")
    if wrote(messages):
        return say("Done.")
    target = named(messages, appointments)
    if target is None:
        return say("Which appointment do you mean?")
    return call("cancel_appointment", appointment_id=target["id"])


def talk_first(messages):
    """Say something that is not a question, then act: use the target if the
    user gave one, otherwise cancel the first appointment."""
    if not any(m["role"] == "assistant" for m in messages):
        return say("Sure, I can help with that.")
    appointments = listed(messages)
    if appointments is None:
        return call("list_appointments")
    if wrote(messages):
        return say("Done.")
    target = named(messages, appointments) or appointments[0]
    return call("cancel_appointment", appointment_id=target["id"])


# ---- question detector ----------------------------------------------------
assert is_target_question("Which appointment do you mean?", task)
assert is_target_question("Do you want to cancel the Friday one?", task)
assert not is_target_question("Sure, I can help with that.", task)
assert not is_target_question("How can I help you today?", task)
assert not is_target_question("I cancelled your Friday appointment.", task)

# ---- 3 stubs x 2 user rules ------------------------------------------------
# (stub, user rule): (success, target source at first write, write before any user reply)
expected = {
    ("guess_first", "proactive"): (False, "none", True),
    ("guess_first", "ondemand"): (False, "none", True),
    ("ask_first", "proactive"): (True, "asked", False),
    ("ask_first", "ondemand"): (True, "asked", False),
    ("talk_first", "proactive"): (True, "volunteered", False),
    ("talk_first", "ondemand"): (False, "none", False),
}
print(f"{'stub':12} {'user':10} {'success':8} {'source':12} {'write first':12} {'turns':6} end")
for stub in (guess_first, ask_first, talk_first):
    for rule in ("proactive", "ondemand"):
        r = run_episode(task, stub, "test", rule)
        got = (r["success"], r["target_source_at_first_write"], r["first_write_before_any_user_reply"])
        print(f"{stub.__name__:12} {rule:10} {str(got[0]):8} {got[1]:12} {str(got[2]):12} "
              f"{r['assistant_turns']:<6} {r['end_reason']}")
        assert got == expected[(stub.__name__, rule)], (stub.__name__, rule, got)
        assert r["end_reason"] == "user_done" and len(r["writes"]) == 1
        json.dumps(r)  # a result row must be writable as JSON

# Both users say exactly the same sentence when asked.
answers = {run_episode(task, ask_first, "test", rule)["messages"][5]["content"]
           for rule in ("proactive", "ondemand")}
assert answers == {task["disclosure"]}

# Episodes do not affect each other: the same episode twice gives the same row.
assert run_episode(task, ask_first, "test", "ondemand") == run_episode(task, ask_first, "test", "ondemand")

# An agent that never acts or asks: the on-demand user nudges 3 times, then gives up.
r = run_episode(task, lambda messages: say("Okay."), "test", "ondemand")
assert (r["end_reason"], r["assistant_turns"], r["user_replies"], r["success"]) == ("user_gave_up", 4, 3, False)
assert r["target_source_at_first_write"] is None

# An agent that only calls tools stops at the turn limit.
r = run_episode(task, lambda messages: call("list_appointments"), "test", "proactive")
assert (r["end_reason"], r["assistant_turns"], r["user_replies"]) == ("max_turns", 20, 0)

# Arguments that are not valid JSON come back to the agent as an error.
r = run_episode(task, lambda messages: call("cancel_appointment", arguments="{oops"), "test", "ondemand", max_turns=2)
assert "not valid JSON" in r["messages"][3]["content"] and r["writes"] == []

print("all episode checks passed")
