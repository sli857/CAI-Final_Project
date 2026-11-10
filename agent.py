"""The agent: system prompts, tool definitions, and the call to the model server.

There is one agent. The two strategies differ only in the system prompt.
The prompts below are DRAFTS. They will be tuned on the development set and
then frozen before the main evaluation.
"""
import json
import time
import urllib.error
import urllib.request

from sandbox import DAYS

BASE_PROMPT = (
    "You are an appointment assistant for a clinic. Use the provided tools to help "
    "the current user manage their appointments. Ask the user for clarification when "
    "needed. When you have finished, tell the user what you did."
)
CLARIFY_PROMPT = BASE_PROMPT + (
    "\n\nBefore you cancel or reschedule an appointment, check whether more than one "
    "appointment matches the user's request. If more than one matches and the user has "
    "not said which one they mean, do not guess. Ask the user a question that "
    "distinguishes the candidates, and act only after the user has identified the "
    "appointment."
)
PROMPTS = {"base": BASE_PROMPT, "clarify": CLARIFY_PROMPT}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_appointments",
        "description": "List all appointments of the current user.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "cancel_appointment",
        "description": "Cancel one appointment of the current user.",
        "parameters": {"type": "object",
                       "properties": {"appointment_id": {"type": "string"}},
                       "required": ["appointment_id"]}}},
    {"type": "function", "function": {
        "name": "reschedule_appointment",
        "description": "Move one appointment of the current user to a new day and time.",
        "parameters": {"type": "object",
                       "properties": {"appointment_id": {"type": "string"},
                                      "new_day": {"type": "string", "enum": DAYS},
                                      "new_time": {"type": "string", "description": "24-hour HH:MM"}},
                       "required": ["appointment_id", "new_day", "new_time"]}}},
]


def make_model(base_url, model, seed, stats, timeout=300):
    """Return a function (messages -> next assistant message) backed by an
    OpenAI-compatible server such as vLLM.

    No temperature is sent, so the server applies the model's default sampling
    settings. `seed` makes a repetition reproducible. `stats` is a dict that
    collects the number of calls, tokens, and seconds for the episode.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def call(messages):
        body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                "parallel_tool_calls": False, "seed": seed, "max_tokens": 512}
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {error.code} from model server: {detail}") from None
        usage = data.get("usage") or {}
        stats["model_calls"] = stats.get("model_calls", 0) + 1
        stats["seconds"] = round(stats.get("seconds", 0) + time.time() - start, 3)
        stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
        stats["completion_tokens"] = stats.get("completion_tokens", 0) + usage.get("completion_tokens", 0)

        # Keep only the standard fields, so the message can be sent back as history.
        raw = data["choices"][0]["message"]
        message = {"role": "assistant", "content": raw.get("content")}
        if raw.get("tool_calls"):
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                for c in raw["tool_calls"]]
        return message

    return call
