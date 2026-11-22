"""CPU-only tests for Inkling's tool-call parser and the adapter's prompt splice (no model, GPU or subprocess)."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling_adapter  # noqa: E402
import inkling_tool_protocol as protocol  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "shell", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
    "additionalProperties": False}}}]


def call(name, args):
    return "<|message_model|>{}<|content_invoke_tool_json|>{}<|end_message|>".format(
        name, json.dumps({"name": name, "args": args}, separators=(",", ":")))


THINK = "<|message_model|><|content_thinking|>look first<|end_message|>"


class ParserTests(unittest.TestCase):
    def test_thinking_and_one_call(self):
        out = protocol.parse_completion(THINK + call("shell", {"command": "ls"}), TOOLS)
        self.assertEqual(out["finish_reason"], "tool_calls")
        self.assertEqual(out["reasoning_content"], "look first")
        self.assertEqual(out["content"], "")
        [c] = out["tool_calls"]
        self.assertEqual(c["function"]["name"], "shell")
        self.assertEqual(json.loads(c["function"]["arguments"]), {"command": "ls"})

    def test_text_answer_and_sampling_end_marker(self):
        text = THINK + "<|message_model|><|content_text|>All green.<|end_message|><|content_model_end_sampling|>"
        out = protocol.parse_completion(text, TOOLS)
        self.assertEqual((out["content"], out["finish_reason"], out["tool_calls"]), ("All green.", "stop", []))

    def test_two_calls_keep_their_order(self):
        out = protocol.parse_completion(call("shell", {"command": "a"}) + call("shell", {"command": "b"}), TOOLS)
        self.assertEqual([json.loads(c["function"]["arguments"])["command"] for c in out["tool_calls"]], ["a", "b"])
        self.assertEqual([c["id"] for c in out["tool_calls"]], ["call_0", "call_1"])

    def test_truncated_output_never_yields_a_call(self):
        out = protocol.parse_completion(call("shell", {"command": "ls"}), TOOLS, stopped=False)
        self.assertEqual((out["tool_calls"], out["finish_reason"]), ([], "length"))

    def test_bad_calls_come_back_as_text_and_execute_nothing(self):
        for bad in (call("rm_rf", {"command": "x"}),                     # not offered
                    call("shell", {"cmd": "x"}),                          # schema rejects
                    call("shell", {"command": 3}),                        # wrong type
                    "<|message_model|>shell<|content_invoke_tool_json|>{not json}<|end_message|>",
                    "<|message_model|>shell<|content_invoke_tool_json|>{\"name\":\"other\",\"args\":{}}<|end_message|>",
                    "<|message_model|>shell<|content_invoke_tool_json|>{\"name\":\"shell\",\"args\":{\"command\":\"x\"}}"):  # unclosed
            out = protocol.parse_completion(call("shell", {"command": "ok"}) + bad, TOOLS)
            self.assertEqual(out["tool_calls"], [], bad)
            self.assertEqual(out["finish_reason"], "stop")
            self.assertTrue(out["invalid_calls"], bad)

    def test_duplicate_argument_keys_are_rejected(self):
        raw = "<|message_model|>shell<|content_invoke_tool_json|>{\"name\":\"shell\",\"args\":{\"command\":\"a\",\"command\":\"b\"}}<|end_message|>"
        self.assertEqual(protocol.parse_completion(raw, TOOLS)["tool_calls"], [])


class CharTokenizer:
    """One character per token, the Inkling markers as single characters would be too coarse: use text ids."""
    def __init__(self):
        self.template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, tools, reasoning_effort):
        assert tokenize is False and reasoning_effort == "low"
        text = "[tools:{}]".format(len(tools))
        for m in messages:
            if m["role"] == "assistant":
                text += "<|message_model|><|content_text|>{}<|end_message|><|content_model_end_sampling|>".format(m["content"])
            else:
                text += "<{}>{}".format(m["role"], m["content"])
        return text


class FakeWorker:
    instances = []
    replies = []

    def __init__(self, command, timeout=0):
        self.requests = []
        FakeWorker.instances.append(self)

    def __enter__(self):
        return self

    def request(self, payload):
        self.requests.append(payload)
        if payload["id"] == 0:
            return dict(tokens=[1], eos=False)
        reply = FakeWorker.replies.pop(0)
        before = self.position if not payload["reset"] else 0
        self.position = before + len(payload["tokens"]) + len(reply)
        return dict(id=payload["id"], tokens=reply, eos=reply[-1] == 3, cache_before=before, cache_after=self.position,
                    prefill_tokens=len(payload["tokens"]), decode_forward_tokens=len(reply), reused_prefix_tokens=before,
                    prefill_seconds=0.5, decode_seconds=1.0, kv_allocated_bytes=10, kv_valid_bytes=5)

    def close(self, graceful=False):
        pass


class SpliceTests(unittest.TestCase):
    def test_a_followup_turn_reuses_the_exact_generated_ids(self):
        # Generated bytes the template would never write (extra spaces), then EOS (\x03).
        raw_turn = "<|message_model|><|content_thinking|>hm<|end_message|><|message_model|>shell<|content_invoke_tool_json|>" \
                   "{ \"name\":\"shell\", \"args\":{\"command\":\"ls\"} }<|end_message|>"
        generated = [ord(c) for c in raw_turn] + [3]
        FakeWorker.instances = []
        backend = inkling_adapter.InklingBackend(CharTokenizer(), ["x"], 100000, [3], 1 << 20, 4096, 5,
                                                 worker_factory=FakeWorker)
        user = {"role": "user", "content": "fix it"}
        FakeWorker.replies = [generated, [ord("o"), ord("k"), 3]]
        first = backend.complete({"model": "flint", "messages": [user], "tools": TOOLS})
        worker = FakeWorker.instances[0]
        self.assertEqual(worker.requests[0]["id"], 0)  # the warm-up came first
        message = first["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "shell")
        self.assertTrue(worker.requests[1]["reset"])
        # the client echoes the turn (arguments as a JSON string) and the tool's output
        history = [user, {"role": "assistant", "content": message["content"], "tool_calls": message["tool_calls"],
                          "reasoning_content": message["reasoning_content"]},
                   {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "a.py"}]
        second = backend.complete({"model": "flint", "messages": history, "tools": TOOLS})
        self.assertEqual(second["choices"][0]["message"]["content"], "ok")
        again = worker.requests[2]
        self.assertFalse(again["reset"])
        self.assertEqual(again["tokens"][:0], [])
        suffix = "".join(chr(i) for i in again["tokens"])
        self.assertEqual(suffix, "<tool>a.py")  # only what is new: the tool message, none of the old turn
        self.assertEqual(second["usage"]["prompt_tokens_details"]["cached_tokens"], first["usage"]["prompt_tokens"] + len(generated))


if __name__ == "__main__":
    unittest.main()
