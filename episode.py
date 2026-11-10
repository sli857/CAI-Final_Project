"""One conversation between an agent and the scripted user.

`model` is any function that takes the message list (OpenAI chat format) and
returns the next assistant message as a dict. A real model and a test stub
both fit this interface.
"""
import json

from sandbox import Sandbox, check_goal
from user_sim import ScriptedUser


def run_episode(task, model, system_prompt, user_rule, max_turns=20):
    sandbox = Sandbox(task["appointments"])  # fresh state for this episode only
    user = ScriptedUser(task, user_rule)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task["request"]}]
    user_replies = 0
    first_write = None
    end_reason = "max_turns"
    turns = 0

    while turns < max_turns:
        message = model(messages)
        turns += 1  # every model call counts as one assistant turn
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            # Tool turn: run the calls and go back to the model. The user does not speak.
            for call in tool_calls:
                try:
                    args = json.loads(call["function"]["arguments"])
                except (TypeError, ValueError):
                    result = {"error": "arguments are not valid JSON"}
                else:
                    writes_before = len(sandbox.writes)
                    result = sandbox.call(call["function"]["name"], args)
                    if first_write is None and len(sandbox.writes) > writes_before:
                        first_write = {"target_source": user.source or "none",
                                       "before_any_user_reply": user_replies == 0}
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": json.dumps(result)})
            continue

        # Text turn: now it is the user's turn to speak.
        reply = user.reply(message.get("content") or "", wrote=bool(sandbox.writes))
        if reply is None:
            end_reason = user.log[-1]["action"]  # "user_done" or "user_gave_up"
            break
        messages.append({"role": "user", "content": reply})
        user_replies += 1

    return {
        "task_id": task["task_id"],
        "user_rule": user_rule,
        "success": check_goal(task, sandbox),
        "end_reason": end_reason,
        "assistant_turns": turns,
        "user_replies": user_replies,
        "writes": sandbox.writes,
        # "none" = the target had not been given when the agent first changed the database.
        "target_source_at_first_write": first_write["target_source"] if first_write else None,
        "first_write_before_any_user_reply": first_write["before_any_user_reply"] if first_write else None,
        "target_questions": sum(e["target_question"] for e in user.log),
        "messages": messages,
        "user_log": user.log,
    }
