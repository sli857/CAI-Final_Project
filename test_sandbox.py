"""Checks for sandbox.py. Run from anywhere:  python3 experiment/test_sandbox.py"""
import os

from sandbox import Sandbox, check_goal, load_tasks

task = load_tasks(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.jsonl"))[0]


def run(calls):
    """Play a list of (tool, args) calls in a fresh sandbox and return it."""
    sandbox = Sandbox(task["appointments"])
    for tool, args in calls:
        sandbox.call(tool, args)
    return sandbox


cancel_wed = ("cancel_appointment", {"appointment_id": "APT-1"})
cancel_fri = ("cancel_appointment", {"appointment_id": "APT-2"})

# Goal check: only "cancel Friday and nothing else" counts as success.
assert check_goal(task, run([cancel_fri])) is True
assert check_goal(task, run([cancel_wed])) is False
assert check_goal(task, run([cancel_wed, cancel_fri])) is False
assert check_goal(task, run([])) is False
# Moving the unrelated appointment also breaks success, even if Friday is cancelled.
move_wed = ("reschedule_appointment", {"appointment_id": "APT-1", "new_day": "Monday", "new_time": "09:00"})
assert check_goal(task, run([cancel_fri, move_wed])) is False

# Reading does not change anything and is not counted as a write.
sandbox = run([("list_appointments", {}), cancel_fri])
assert check_goal(task, sandbox) is True
assert len(sandbox.writes) == 1

# Bad calls return an error and leave the table unchanged.
sandbox = Sandbox(task["appointments"])
before = sandbox.state()
assert "error" in sandbox.call("cancel_appointment", {"appointment_id": "APT-999"})
assert "error" in sandbox.call("reschedule_appointment", {"appointment_id": "APT-1", "new_day": "Sunday", "new_time": "09:00"})
assert "error" in sandbox.call("delete_everything", {})
assert "error" in sandbox.call("cancel_appointment", "APT-1")
assert sandbox.state() == before and sandbox.writes == []
sandbox.call(*cancel_fri)
assert "error" in sandbox.call(*cancel_fri)  # cancelling twice is an error
assert len(sandbox.writes) == 1

# Isolation: a new sandbox starts clean, and the task data is never modified.
run([cancel_wed, cancel_fri])
fresh = Sandbox(task["appointments"])
assert all(a["status"] == "scheduled" for a in fresh.state())
assert "status" not in task["appointments"][0]
listed = fresh.call("list_appointments", {})["appointments"]
listed[0]["status"] = "cancelled"  # editing a returned copy must not leak in
assert fresh.state()[0]["status"] == "scheduled"

print("all sandbox checks passed")
