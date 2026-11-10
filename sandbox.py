"""Appointment sandbox and goal check.

One Sandbox object holds the appointment table for ONE episode. It is built
from a task's initial data and thrown away afterwards, so episodes never share state.
"""
import copy
import json
import re

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WRITE_TOOLS = ("cancel_appointment", "reschedule_appointment")


class Sandbox:
    def __init__(self, appointments):
        # Deep copy: changes made in this episode never touch the task data.
        self.appointments = [dict(copy.deepcopy(a), status="scheduled") for a in appointments]
        self.writes = []  # every successful state-changing call, in order

    def call(self, tool, args):
        """Run one tool call. Always returns a dict; errors are returned, not raised."""
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        if tool == "list_appointments":
            return {"appointments": copy.deepcopy(self.appointments)}
        if tool not in WRITE_TOOLS:
            return {"error": f"unknown tool: {tool}"}

        appt = next((a for a in self.appointments if a["id"] == args.get("appointment_id")), None)
        if appt is None:
            return {"error": "appointment not found"}
        if appt["status"] != "scheduled":
            return {"error": "appointment is already cancelled"}

        if tool == "cancel_appointment":
            appt["status"] = "cancelled"
        else:
            day, time = args.get("new_day"), args.get("new_time")
            if day not in DAYS or not re.fullmatch(r"\d\d:\d\d", str(time)):
                return {"error": "new_day must be a weekday name and new_time must be HH:MM"}
            appt["day"], appt["time"] = day, time
        self.writes.append({"tool": tool, "args": copy.deepcopy(args)})
        return {"ok": True, "appointment": copy.deepcopy(appt)}

    def state(self):
        """The full table, in a fixed order, for comparison."""
        return sorted(copy.deepcopy(self.appointments), key=lambda a: a["id"])


def expected_state(task):
    """The final table the user wants: apply the task's goal to a fresh sandbox."""
    sandbox = Sandbox(task["appointments"])
    for step in task["goal"]:
        result = sandbox.call(step["tool"], step["args"])
        assert "error" not in result, f"bad goal in task {task['task_id']}: {result}"
    return sandbox.state()


def check_goal(task, sandbox):
    """Success = the final table equals the expected table exactly.

    This covers both requirements at once: the intended change was made,
    and no other appointment was touched.
    """
    return sandbox.state() == expected_state(task)


def load_tasks(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
