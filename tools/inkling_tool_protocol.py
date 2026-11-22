"""CPU-only parsing of Inkling 2's native tool-call syntax (stdlib only).

The checkpoint's chat template writes a model turn as messages, each opened by
`<|message_model|>`:

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>an answer<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

and a completed turn ends with `<|content_model_end_sampling|>` (the model's
EOS, which the adapter strips before parsing). parse_completion() takes that
decoded text, special tokens kept, and the OpenAI function definitions, and
returns the same shape as flint_tool_protocol.parse_completion: content,
reasoning_content, tool_calls, finish_reason. It never executes anything.

Argument checking reuses flint_tool_protocol's schema validator. A completion
that was cut by the token limit yields no calls. A call that is malformed, names
a tool that was not offered, or has arguments the schema rejects is not turned
into a call: its turn comes back as plain text with finish_reason "stop" and the
reason in `invalid_calls`, so the agent sees what the model wrote instead of the
whole request failing.
"""
import json

from flint_tool_protocol import ProtocolError, _schemas, _validate

MESSAGE = "<|message_model|>"
END = "<|end_message|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
INVOKE = "<|content_invoke_tool_json|>"
SAMPLING_END = "<|content_model_end_sampling|>"


def _object_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ProtocolError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def parse_completion(text, tools, stopped=True):
    if not isinstance(text, str):
        raise ProtocolError("Completion text must be a string")
    if not stopped:
        # Cut by the token limit: never executable, whatever it contains.
        visible = text.replace(SAMPLING_END, "")
        return {"content": visible, "reasoning_content": "", "tool_calls": [],
                "finish_reason": "length", "invalid_calls": []}
    schemas = _schemas(tools)
    content, reasoning, calls, invalid = [], [], [], []
    for index, part in enumerate(text.replace(SAMPLING_END, "").split(MESSAGE)):
        if not part.strip():
            continue
        if not part.endswith(END):
            invalid.append("message {} is not closed".format(index))
            content.append(part)
            continue
        body = part[:-len(END)]
        if body.startswith(THINKING):
            reasoning.append(body[len(THINKING):])
        elif body.startswith(TEXT):
            content.append(body[len(TEXT):])
        elif INVOKE in body:
            name, _, raw = body.partition(INVOKE)
            try:
                if name not in schemas:
                    raise ProtocolError("Unknown or non-whitelisted function: " + name)
                call = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=lambda c: (_ for _ in ()).throw(
                    ProtocolError("Invalid JSON constant: " + c)))
                if (not isinstance(call, dict) or set(call) != {"name", "args"} or call["name"] != name
                        or not isinstance(call["args"], dict)):
                    raise ProtocolError("Tool call JSON must be {name, args} for " + name)
                _validate(call["args"], schemas[name])
            except (ValueError, ProtocolError) as error:
                invalid.append("{}: {}".format(name or "?", error))
                content.append(body)
                continue
            calls.append({"id": "call_" + str(len(calls)), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(call["args"], ensure_ascii=False,
                                                                              allow_nan=False)}})
        else:
            invalid.append("message {} has an unknown kind".format(index))
            content.append(body)
    if invalid:
        # One bad call spoils the turn: nothing in it is executed.
        calls = []
    return {"content": "".join(content), "reasoning_content": "".join(reasoning), "tool_calls": calls,
            "finish_reason": "tool_calls" if calls else "stop", "invalid_calls": invalid}
