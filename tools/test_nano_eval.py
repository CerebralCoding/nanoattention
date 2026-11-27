#!/usr/bin/env python3
"""CPU-only unit tests for tools/nano_eval.py.

Run via:
    python3 -B -m unittest discover -s tools -p 'test_*.py'
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import nano_eval  # noqa: E402


class GraderTests(unittest.TestCase):
    def test_merge_intervals_pass_and_failures(self):
        good = """Here is the O(n log n) solution:
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted((iv[0], iv[1]) for iv in intervals)
    out = [[ordered[0][0], ordered[0][1]]]
    for s, e in ordered[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out
```"""
        res = nano_eval.grade_merge_intervals(good)
        self.assertTrue(res.passed, res.reason)

        wrong_touching = """```python
def merge_intervals(intervals):
    ordered = sorted(intervals)
    out = []
    for s, e in ordered:
        if not out or s >= out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out
```"""
        res_bad = nano_eval.grade_merge_intervals(wrong_touching)
        self.assertFalse(res_bad.passed)
        self.assertIn("touching", res_bad.reason)

        self.assertFalse(nano_eval.grade_merge_intervals("No code here.").passed)

    def test_binary_search_rejects_linear_scan(self):
        good = """```python
def first_occurrence(nums, target):
    lo, hi = 0, len(nums) - 1
    ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = nums[mid]
        if val >= target:
            if val == target:
                ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans
```"""
        res = nano_eval.grade_binary_search(good)
        self.assertTrue(res.passed, res.reason)
        self.assertLessEqual(res.detail["reads_100k"], nano_eval.MAX_ARRAY_ACCESSES_BINARY_SEARCH)

        linear = """```python
def first_occurrence(nums, target):
    for i in range(len(nums)):
        if nums[i] == target:
            return i
    return -1
```"""
        res_lin = nano_eval.grade_binary_search(linear)
        self.assertFalse(res_lin.passed)
        self.assertIn("excessive_reads", res_lin.reason)

    def test_math_graders(self):
        self.assertTrue(nano_eval.grade_math_mod7("Table...\n**FINAL_ANSWER: 2, 9, 16, 23**").passed)
        self.assertFalse(nano_eval.grade_math_mod7("FINAL_ANSWER: 2, 9, 16").passed)
        self.assertFalse(nano_eval.grade_math_mod7("The answers are 2, 9, 16, 23.").passed)

        self.assertTrue(nano_eval.grade_math_batches("Steps...\nFINAL_ANSWER: A=20, B=44, C=34").passed)
        self.assertFalse(nano_eval.grade_math_batches("FINAL_ANSWER: A=19, B=42, C=32").passed)
        self.assertFalse(nano_eval.grade_math_batches("A=20, B=44, C=34").passed)

    def test_json_extract_strict_and_fenced(self):
        raw = '{"release": "v0.3.1", "decode_tok_s": 133.26, "exact_parity": true, "modes": ["hidden", "logits", "logits-adaptive"]}'
        r1 = nano_eval.grade_json_extract(raw)
        self.assertTrue(r1.passed)
        self.assertTrue(r1.detail["raw_json"])

        fenced = f"```json\n{raw}\n```"
        r2 = nano_eval.grade_json_extract(fenced)
        self.assertTrue(r2.passed)
        self.assertFalse(r2.detail["raw_json"])

        bool_as_int = '{"release": "v0.3.1", "decode_tok_s": 133.26, "exact_parity": 1, "modes": ["hidden", "logits", "logits-adaptive"]}'
        self.assertFalse(nano_eval.grade_json_extract(bool_as_int).passed)

        wrong_mode = '{"release": "v0.3.1", "decode_tok_s": 133.26, "exact_parity": true, "modes": ["hidden"]}'
        self.assertFalse(nano_eval.grade_json_extract(wrong_mode).passed)

    def test_code_4k_grader(self):
        good = (
            "Clearing lsa_slot_valid before resetting ring_head ensures that recycled ring buffer slots "
            "cannot serve stale speculative entries on the next verification batch.\n"
            "FINAL_ANSWER: RING_GUARD_SLOTS=64, FUNCTION=rollbackRejectedSuffix"
        )
        r = nano_eval.grade_code_4k(good)
        self.assertTrue(r.passed, r.reason)
        self.assertFalse(r.detail["used_fallback"])

        no_expl = "FINAL_ANSWER: RING_GUARD_SLOTS=64, FUNCTION=rollbackRejectedSuffix"
        self.assertFalse(nano_eval.grade_code_4k(no_expl).passed)
        wrong_const = (
            "Detailed explanation of rollback ordering in the ring state machine.\n"
            "FINAL_ANSWER: RING_GUARD_SLOTS=32, FUNCTION=rollbackRejectedSuffix"
        )
        self.assertFalse(nano_eval.grade_code_4k(wrong_const).passed)


class CoherenceAndTextTests(unittest.TestCase):
    def test_extract_visible_answer_with_and_without_thinking(self):
        self.assertEqual(nano_eval.extract_visible_answer("plain ok", "<|content_text|>plain ok<|end_message|>"), "plain ok")
        special = (
            "<|content_thinking|>Wait let me think.<|end_message|>"
            "<|content_text|>FINAL_ANSWER: 2, 9, 16, 23<|end_message|><|content_model_end_sampling|>"
        )
        self.assertEqual(nano_eval.extract_visible_answer("Wait let me think.FINAL_ANSWER: 2, 9, 16, 23", special), "FINAL_ANSWER: 2, 9, 16, 23")
        unclosed = "<|content_thinking|>still thinking without answer"
        self.assertEqual(nano_eval.extract_visible_answer("still thinking without answer", unclosed), "")

    def test_check_coherence_flags_loops_and_corruption(self):
        self.assertEqual(nano_eval.check_coherence(list(range(10, 40)), "Clean normal explanation.")[0], True)
        self.assertEqual(nano_eval.check_coherence([], "nonempty"), (False, "empty_tokens"))
        self.assertEqual(nano_eval.check_coherence([1, 2], "bad\x00byte"), (False, "nul_byte"))
        self.assertEqual(nano_eval.check_coherence([1, 2], "\ufffd" * 5), (False, "replacement_chars:5"))

        loop_tokens = [10, 20, 30, 40] + [1, 2, 3, 4] * 5
        ok, reason = nano_eval.check_coherence(loop_tokens, "repeat loop")
        self.assertFalse(ok)
        self.assertIn("repeated_4gram", reason)

        long_low_div = list(range(50)) + [77] * 80
        ok_div, reason_div = nano_eval.check_coherence(long_low_div, "stuck tail")
        self.assertFalse(ok_div)

        rep_line = "This exact identical sentence repeats again and again."
        ok_line, reason_line = nano_eval.check_coherence(list(range(20, 60)), "\n".join([rep_line] * 4))
        self.assertFalse(ok_line)
        self.assertIn("repeated_line", reason_line)

    def test_leak_and_parity_checks(self):
        self.assertEqual(nano_eval.check_stderr_leaks("nano: inkling2_loop, 204 ops"), (True, []))
        ok, reasons = nano_eval.check_stderr_leaks("error(gpa): memory leaked at 0x1000 SafeAllocator")
        self.assertFalse(ok)
        self.assertIn("SafeAllocator", reasons)
        self.assertIn("leaked", reasons)

        self.assertEqual(nano_eval.compare_token_parity([1, 2, 3], [1, 2, 3]), (True, None))
        eq, diff = nano_eval.compare_token_parity([1, 2, 3], [1, 9, 3])
        self.assertFalse(eq)
        self.assertEqual(diff, {"index": 1, "base": 2, "draft": 9})


class BridgeAndChunkingTests(unittest.TestCase):
    def test_make_chunked_requests_under_line_limit(self):
        ids = list(range(60000))
        reqs = nano_eval.make_chunked_requests(ids, turn1_max=64, chunk_size=24576, reset_first=True)
        self.assertEqual(len(reqs), 3)
        self.assertTrue(reqs[0]["reset"])
        self.assertEqual(reqs[0]["max_tokens"], 0)
        self.assertEqual(reqs[2]["max_tokens"], 64)
        for r in reqs:
            self.assertLess(len(json.dumps(r, separators=(",", ":")).encode("utf-8")), nano_eval.MAX_SESSION_LINE_BYTES)

    def test_nano_tokenizer_batch_protocol(self):
        seen_bytes: list[bytes] = []

        def fake_run(cmd, capture_output=True, timeout=120, check=False):
            batch_path = Path(cmd[cmd.index("--batch") + 1])
            data = batch_path.read_bytes()
            seen_bytes.append(data)
            # Parse binary records to verify exact wire format.
            off = 0
            out_lines: list[str] = []
            while off < len(data):
                kind = chr(data[off])
                length = struct.unpack_from("<I", data, off + 1)[0]
                off += 5
                size = length * 4 if kind == "D" else length
                payload = data[off : off + size]
                off += size
                if kind == "E":
                    out_lines.append(json.dumps({"ids": [101, 102]}))
                elif kind == "D":
                    out_lines.append(json.dumps({"text": "ok", "special": "ok", "stream": "ok"}))
                elif kind == "C":
                    out_lines.append(json.dumps({"text": "prompt", "ids": [1, 2, 3]}))
                elif kind == "N":
                    out_lines.append(json.dumps({"text": f"rendered:{payload.decode()}"}))
            return subprocess.CompletedProcess(cmd, 0, stdout=("\n".join(out_lines) + "\n").encode("utf-8"), stderr=b"")

        tok = nano_eval.NanoTokenizer(Path("/fake/nano"), Path("/fake/ckpt"))
        with mock.patch("subprocess.run", side_effect=fake_run):
            chat_ids = tok.chat_many(["hello"], thinking=False)
            self.assertEqual(chat_ids, [[1, 2, 3]])
            pair = tok.next_turn_ids_pair("turn2", thinking=False)
            self.assertEqual(pair[True], [101, 102])
            self.assertEqual(pair[False], [101, 102])
            dec = tok.decode_many([[1, 2]])
            self.assertEqual(dec[0]["text"], "ok")
        self.assertGreaterEqual(len(seen_bytes), 3)


class OrchestrationAndCompareTests(unittest.TestCase):
    def test_end_to_end_core_orchestration_and_compare_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nano_bin = root / "nano"
            nano_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            ckpt = root / "Inkling-2-Mini"
            ckpt.mkdir()
            (ckpt / "config.json").write_text('{"model_type": "inkling2_loop"}', encoding="utf-8")
            draft = root / "inkling-2-mini-dflash2"
            draft.mkdir()

            answers = {
                1: "```python\ndef merge_intervals(intervals):\n    if not intervals: return []\n    s = sorted(intervals)\n    out = [[s[0][0], s[0][1]]]\n    for a, b in s[1:]:\n        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)\n        else: out.append([a, b])\n    return out\n```",
                2: "```python\ndef first_occurrence(nums, target):\n    lo, hi, ans = 0, len(nums) - 1, -1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        v = nums[m]\n        if v >= target:\n            if v == target: ans = m\n            hi = m - 1\n        else:\n            lo = m + 1\n    return ans\n```",
                3: "Table done.\nFINAL_ANSWER: 2, 9, 16, 23",
                4: "Algebra done.\nFINAL_ANSWER: A=20, B=44, C=34",
                5: '{"release": "v0.3.1", "decode_tok_s": 133.26, "exact_parity": true, "modes": ["hidden", "logits", "logits-adaptive"]}',
                6: "Clearing lsa_slot_valid before resetting ring_head guarantees no stale entries are read.\nFINAL_ANSWER: RING_GUARD_SLOTS=64, FUNCTION=rollbackRejectedSuffix",
            }
            tok_counter = 1000
            text_for_toks: dict[tuple[int, ...], str] = {}
            toks_for_task: dict[int, list[int]] = {}
            for tid, ans_text in answers.items():
                toks = [tok_counter + tid * 10 + j for j in range(1, 8)]
                toks_for_task[tid] = toks
                text_for_toks[tuple(toks)] = ans_text

            def fake_batch(self, records):
                out = []
                for rec in records:
                    kind = chr(rec[0])
                    length = struct.unpack_from("<I", rec, 1)[0]
                    payload = rec[5:]
                    if kind == "C":
                        obj = json.loads(payload.decode("utf-8"))
                        user = obj["messages"][0]["content"]
                        # Make code_4k probe hit ~4050 tokens when repeated == 14 times.
                        n_exc = user.count("### Excerpt")
                        ids = list(range(1, (4050 if n_exc == 14 else max(16, n_exc * 280)) + 1))
                        out.append({"text": user, "ids": ids})
                    elif kind == "D":
                        ids = tuple(struct.unpack(f"<{length}I", payload))
                        txt = text_for_toks.get(ids, "ok")
                        out.append({"text": txt, "special": txt, "stream": txt})
                    else:
                        out.append({"ids": [1, 2, 3]})
                return out

            def fake_run_session_batch(cmd, requests, timeout=900):
                is_draft = "--draft" in cmd
                resps = []
                for r in requests:
                    rid = r["id"]
                    toks = [99] if rid == 0 else list(toks_for_task[rid])
                    # Make draft fail parity on one config if NANO_TEST_BREAK_PARITY=1.
                    if is_draft and os.environ.get("NANO_TEST_BREAK_PARITY") == "1" and rid == 5:
                        toks[-1] += 1
                    dec_s = (len(toks) / 110.0) if is_draft else (len(toks) / 65.0)
                    resps.append({
                        "id": rid,
                        "tokens": toks,
                        "eos": True,
                        "cache_before": 0,
                        "cache_after": len(r["tokens"]) + len(toks),
                        "prefill_tokens": len(r["tokens"]),
                        "decode_forward_tokens": len(toks),
                        "reused_prefix_tokens": 0,
                        "prefill_seconds": max(0.001, len(r["tokens"]) / 1500.0),
                        "decode_seconds": dec_s,
                        "kv_allocated_bytes": 1024,
                        "kv_valid_bytes": 512,
                        "speculative_batches": 2 if is_draft else 0,
                        "speculative_proposed": 10 if is_draft else 0,
                        "speculative_accepted": 6 if is_draft else 0,
                        "dispatches": 100,
                    })
                return resps, "nano: inkling2_loop, 204 ops, 615 states"

            out_base = root / "base.json"
            out_cand = root / "cand.json"
            md_path = root / "report.md"

            with (
                mock.patch.object(nano_eval.NanoTokenizer, "batch", fake_batch),
                mock.patch.object(nano_eval, "run_session_batch", side_effect=fake_run_session_batch),
                mock.patch.object(nano_eval, "read_nano_version", return_value="nano 0.3.1"),
            ):
                rc = nano_eval.main([
                    "run",
                    str(ckpt),
                    "--binary",
                    str(nano_bin),
                    "--draft",
                    str(draft),
                    "--config",
                    "3p_hidden",
                    "--repeats",
                    "2",
                    "--strict",
                    "--out",
                    str(out_base),
                    "--md",
                    str(md_path),
                ])
                self.assertEqual(rc, 0)

                base_doc = json.loads(out_base.read_text(encoding="utf-8"))
                self.assertEqual(base_doc["meta"]["checkpoint_name"], "Inkling-2-Mini")
                self.assertNotIn(str(root), out_base.read_text(encoding="utf-8"))
                self.assertEqual(base_doc["configs"][0]["summary"]["tasks_passed"], 6)
                self.assertEqual(base_doc["configs"][0]["summary"]["parity_passed"], 6)
                self.assertIn("Config Summary", md_path.read_text(encoding="utf-8"))

                # Self-compare passes --gate.
                self.assertEqual(nano_eval.main(["compare", str(out_base), str(out_base), "--gate"]), 0)

                # Candidate with parity failure fails both --strict and compare --gate.
                with mock.patch.dict(os.environ, {"NANO_TEST_BREAK_PARITY": "1"}):
                    rc_strict = nano_eval.main([
                        "run",
                        str(ckpt),
                        "--binary",
                        str(nano_bin),
                        "--draft",
                        str(draft),
                        "--config",
                        "3p_hidden",
                        "--strict",
                        "--out",
                        str(out_cand),
                    ])
                    self.assertEqual(rc_strict, 1)
                self.assertEqual(nano_eval.main(["compare", str(out_base), str(out_cand), "--gate"]), 1)

    def test_coherent_decode_excludes_incoherent_tasks(self):
        tasks = [
            {
                "id": "a",
                "suite": "core",
                "prompt_tokens": 100,
                "generated_tokens": 100,
                "prefill_seconds": 0.1,
                "decode_seconds": 1.0,  # 100 tok/s
                "passed": True,
                "quality_passed": True,
                "coherent": True,
                "exact_parity": True,
                "leak_free": True,
                "mean_step": 4.0,
                "arms": {},
            },
            {
                "id": "b",
                "suite": "core",
                "prompt_tokens": 100,
                "generated_tokens": 200,
                "prefill_seconds": 0.1,
                "decode_seconds": 1.0,  # 200 tok/s loop-inflated incoherent
                "passed": False,
                "quality_passed": False,
                "coherent": False,
                "exact_parity": True,
                "leak_free": True,
                "mean_step": 6.0,
                "arms": {},
            },
        ]
        summary = nano_eval.rollup_config(tasks)
        self.assertAlmostEqual(summary["weighted_decode_tok_s"], 150.0, places=3)
        self.assertAlmostEqual(summary["coherent_decode_tok_s"], 100.0, places=3)


if __name__ == "__main__":
    unittest.main()
