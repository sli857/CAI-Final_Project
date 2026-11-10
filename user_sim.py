"""Scripted simulated user. No model calls.

Two rules:
  proactive : gives the target the first time it gets to speak.
  ondemand  : gives the target only when the assistant asks about it.
Both rules use the same sentence (task["disclosure"]) when they give the target.
"""
import re

NUDGE = "Please go ahead with my request."
MAX_NUDGES = 3


def is_target_question(text, task):
    """Draft rule: a question that says "which" or mentions a candidate's day, time, or id.

    To be tuned on the development set and then frozen. Every decision is logged.
    """
    if "?" not in text:
        return False
    low = text.lower()
    if re.search(r"\bwhich\b", low):
        return True
    return any(a["day"].lower() in low or a["time"] in low or a["id"].lower() in low
               for a in task["appointments"])


class ScriptedUser:
    def __init__(self, task, rule):
        assert rule in ("proactive", "ondemand"), rule
        self.task, self.rule = task, rule
        # Where the agent got the target from: None (not given yet), "asked",
        # "volunteered", or "request" (control tasks: it was in the first message).
        self.source = "request" if task["kind"] == "control" else None
        self.nudges = 0  # nudges in a row
        self.log = []    # one entry per assistant message the user saw

    def reply(self, assistant_text, wrote):
        """Return the user's next message, or None to end the conversation.

        wrote: True if the agent has already made a state-changing call.
        """
        asked = is_target_question(assistant_text, self.task)
        entry = {"assistant": assistant_text, "target_question": asked, "after_write": wrote}
        self.log.append(entry)

        # The agent has acted and the user has nothing new to say: stop.
        if wrote and (self.source is not None or (self.rule == "ondemand" and not asked)):
            entry["action"] = "user_done"
            return None

        # Give the target: when asked (both rules), or at the first chance (proactive).
        if asked or (self.rule == "proactive" and self.source is None):
            if self.source is None:
                self.source = "asked" if asked else "volunteered"
            self.nudges = 0
            entry["action"] = "answer" if asked else "volunteer"
            return self.task["disclosure"]

        # Otherwise say something that carries no target information.
        if self.nudges >= MAX_NUDGES:
            entry["action"] = "user_gave_up"
            return None
        self.nudges += 1
        entry["action"] = "nudge"
        return NUDGE
