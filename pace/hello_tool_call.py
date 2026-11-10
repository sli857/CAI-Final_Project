"""Hello world for tool calling against an OpenAI-compatible server (vLLM).

Usage: python hello_tool_call.py http://127.0.0.1:PORT MODEL_NAME

Round 1: send the ambiguous request; we expect a list_appointments tool call.
Round 2: send back a fake tool result; we only look at what the model does next.
This is a connectivity check, not an experiment. Exit code 0 means round 1
returned a well-formed tool call.
"""
import json
import sys
import urllib.request

base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]

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
]
FAKE_LISTING = {"appointments": [
    {"id": "APT-1", "day": "Wednesday", "time": "10:00", "provider": "Dr. Lee", "status": "scheduled"},
    {"id": "APT-2", "day": "Friday", "time": "15:00", "provider": "Dr. Lee", "status": "scheduled"}]}


def chat(messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "temperature": 0, "max_tokens": 512}
    request = urllib.request.Request(base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


messages = [
    {"role": "system", "content": "You are an appointment assistant. Use the tools to help the user. "
                                  "Ask the user for clarification when needed."},
    {"role": "user", "content": "Please cancel my appointment."},
]
ok = False
for round_number in (1, 2):
    response = chat(messages)
    message = response["choices"][0]["message"]
    print(f"--- round {round_number}: assistant message ---")
    print(json.dumps(message, indent=2))
    print("usage:", json.dumps(response.get("usage")))
    calls = message.get("tool_calls") or []
    if not calls:
        print("(no tool call in this round)")
        break
    for call in calls:
        json.loads(call["function"]["arguments"])  # raises if the arguments are not valid JSON
    if round_number == 1:
        ok = True
    messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
    for call in calls:
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(FAKE_LISTING)})

print("RESULT:", "tool calling works" if ok else "round 1 did not return a tool call")
sys.exit(0 if ok else 1)
