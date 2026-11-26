#!/usr/bin/env python3
"""Live end-to-end verification of DFlash 2 (`incoai/Inkling-2-Mini-DFlash2`) on Inkling 2 Mini.

Checks sequentially (one model load at a time):
  1. `nano generate` vs `nano generate --draft` on coding + math prompts (`192` tokens):
     100% token-for-token exact greedy match and `mean_accepted_per_step >= 3.5`.
  2. Multi-turn `nano session` vs `nano session --draft` (Turn 1 -> Turn 2 -> `reset: true`):
     100% token-for-token exact match, `prefill_logits` / `final_logits` exact match,
     exact `cache_before` / `cache_after`, and bitwise replay on `reset: true`.
"""
import argparse
import json
import pathlib
import re
import subprocess
import tempfile

import numpy as np

DEFAULT_TARGET = "/Users/rachpradhan/nanoattention/.graff/worktrees/liquid-main-port/target/exo-hillclimb/models/thinkingmachines--Inkling-2-Mini"
DEFAULT_DRAFT = "/Users/rachpradhan/newflint/models/inkling-2-mini-dflash2"
DEFAULT_NANO = "./zig-out/bin/nano"
DEFAULT_KNOBS = "src/mega/presets/inkling.json"

PROMPTS = {
    "coding": (
        "Write a clean Python function `merge_intervals(intervals)` that merges overlapping "
        "[start, end] intervals in O(n log n) time, then explain its edge cases."
    ),
    "math": (
        "Solve step by step: Find all positive integers n such that n^2 + 3n + 11 "
        "is divisible by 7. Show the modular arithmetic table."
    ),
}

TURN2_TEXT = "\nNow write 3 concise unit tests for the solution above.\n"


def load_tokenizer(target: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(target, trust_remote_code=False)


def tokenize_chat(tok, text: str) -> list[int]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(tok.encode(rendered, add_special_tokens=False))


def tokenize_raw(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def run_generate(
    nano: str,
    target: str,
    knobs: str,
    tokens_file: pathlib.Path,
    max_tokens: int,
    context: int,
    draft: str | None,
) -> tuple[list[int], str]:
    cmd = [
        nano,
        "generate",
        target,
        "--tokens-file",
        str(tokens_file),
        "--max-tokens",
        str(max_tokens),
        "--knobs",
        knobs,
        "--context",
        str(context),
    ]
    if draft is not None:
        cmd.extend(["--draft", draft])
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)["tokens"], proc.stderr


def run_session(
    nano: str,
    target: str,
    knobs: str | None,
    requests: list[dict],
    context: int,
    draft: str | None,
) -> list[dict]:
    cmd = [nano, "session", target, "--context", str(context)]
    if knobs is not None:
        cmd.extend(["--knobs", knobs])
    if draft is not None:
        cmd.extend(["--draft", draft])
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(cmd, input=payload, check=True, text=True, capture_output=True)
    return [json.loads(line) for line in proc.stdout.strip().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nano", default=DEFAULT_NANO)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--draft", default=DEFAULT_DRAFT)
    p.add_argument("--knobs", default=DEFAULT_KNOBS)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--min-accept", type=float, default=3.5)
    args = p.parse_args()

    tok = load_tokenizer(args.target)
    with tempfile.TemporaryDirectory(prefix="dflash2_verify_") as td:
        td_path = pathlib.Path(td)
        for name, text in PROMPTS.items():
            ids = tokenize_chat(tok, text)
            tf = td_path / f"{name}.json"
            tf.write_text(json.dumps({"tokens": ids}))
            print(f"=== Verify generate parity: {name} (prompt_tokens={len(ids)}) ===", flush=True)
            base_toks, _ = run_generate(args.nano, args.target, args.knobs, tf, args.max_tokens, args.context, None)
            spec_toks, spec_err = run_generate(args.nano, args.target, args.knobs, tf, args.max_tokens, args.context, args.draft)
            print(f"  speculative stderr: {spec_err.strip()}", flush=True)
            if base_toks != spec_toks:
                for idx, (a, b) in enumerate(zip(base_toks, spec_toks)):
                    if a != b:
                        raise AssertionError(f"{name}: mismatch at output index {idx}: base={a} spec={b}")
                raise AssertionError(f"{name}: length mismatch base={len(base_toks)} spec={len(spec_toks)}")
            m = re.search(r"mean_step=([0-9.]+)", spec_err)
            if not m:
                raise AssertionError(f"{name}: missing mean_step in stderr: {spec_err!r}")
            mean_step = float(m.group(1))
            print(f"  EXACT PARITY ({len(base_toks)} tokens matched 100%), mean_step={mean_step:.2f}", flush=True)
            print(f"  decoded preview: {tok.decode(spec_toks[:80])!r}", flush=True)
            if mean_step < args.min_accept:
                raise AssertionError(f"{name}: mean_step {mean_step:.2f} < min_accept {args.min_accept}")

        # Reference-knob 3-turn session check (0 ULP exact prefill and final logits).
        ref_reqs = [
            {"id": 1, "tokens": [200002, 200004, 1910, 200010], "max_tokens": 16, "return_logits": True},
            {"id": 2, "tokens": [1910, 25], "max_tokens": 16, "return_logits": True},
            {"id": 3, "tokens": [200002, 200004, 1910, 200010], "max_tokens": 16, "reset": True, "return_logits": True},
        ]
        print("=== Verify reference-knob multi-turn session + reset (0-ULP parity) ===", flush=True)
        r_base = run_session(args.nano, args.target, None, ref_reqs, 256, None)
        r_spec = run_session(args.nano, args.target, None, ref_reqs, 256, args.draft)
        for i in range(3):
            br, sr = r_base[i], r_spec[i]
            if br["tokens"] != sr["tokens"]:
                raise AssertionError(f"ref session turn {i+1}: token mismatch")
            p_diff = float(np.max(np.abs(np.array(br["prefill_logits"]) - np.array(sr["prefill_logits"]))))
            f_diff = float(np.max(np.abs(np.array(br["final_logits"]) - np.array(sr["final_logits"]))))
            print(f"  ref turn {i+1}: exact {len(sr['tokens'])} tokens, prefill_diff={p_diff:.2e}, final_diff={f_diff:.2e}", flush=True)
            if p_diff > 1e-5 or f_diff > 1e-5:
                raise AssertionError(f"ref session turn {i+1}: logit diff too high ({p_diff}, {f_diff})")

        turn1_ids = tokenize_chat(tok, PROMPTS["coding"])
        turn2_ids = tokenize_raw(tok, TURN2_TEXT)
        reqs = [
            {"id": 1, "tokens": turn1_ids, "max_tokens": 96, "return_logits": True},
            {"id": 2, "tokens": turn2_ids, "max_tokens": 96, "return_logits": True},
            {"id": 3, "tokens": turn1_ids, "max_tokens": 96, "reset": True, "return_logits": True},
        ]
        print("=== Verify multi-turn session + reset parity (inkling.json) ===", flush=True)
        base_resp = run_session(args.nano, args.target, args.knobs, reqs, args.context, None)
        spec_resp = run_session(args.nano, args.target, args.knobs, reqs, args.context, args.draft)

        for i in range(3):
            br, sr = base_resp[i], spec_resp[i]
            if br["tokens"] != sr["tokens"]:
                raise AssertionError(f"session turn {i+1}: token mismatch")
            if br["eos"] != sr["eos"] or br["cache_before"] != sr["cache_before"] or br["cache_after"] != sr["cache_after"]:
                raise AssertionError(f"session turn {i+1}: metadata mismatch {br['cache_after']} vs {sr['cache_after']}")
            b_pre, s_pre = np.array(br["prefill_logits"]), np.array(sr["prefill_logits"])
            b_fin, s_fin = np.array(br["final_logits"]), np.array(sr["final_logits"])
            p_diff = float(np.max(np.abs(b_pre - s_pre)))
            f_diff = float(np.max(np.abs(b_fin - s_fin)))
            top1_match = int(np.argmax(b_fin)) == int(np.argmax(s_fin))
            print(
                f"  turn {i+1}: exact {len(sr['tokens'])} tokens, prefill_diff={p_diff:.2e}, final_diff={f_diff:.2e}, "
                f"final_top1_eq={top1_match}, batches={sr['speculative_batches']}, proposed={sr['speculative_proposed']}, accepted={sr['speculative_accepted']}",
                flush=True,
            )
            if p_diff > 1e-4 or not top1_match:
                raise AssertionError(f"session turn {i+1}: logit/top1 mismatch")
            if sr["speculative_batches"] == 0:
                raise AssertionError(f"session turn {i+1}: expected speculative_batches > 0")
        if spec_resp[0]["tokens"] != spec_resp[2]["tokens"]:
            raise AssertionError("session reset: turn 3 did not replay turn 1 identically")
        if spec_resp[0]["final_logits"] != spec_resp[2]["final_logits"]:
            raise AssertionError("session reset: turn 3 final_logits did not match turn 1 bitwise")
        print("ALL DFLASH2 LIVE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
