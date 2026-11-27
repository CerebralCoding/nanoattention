#!/usr/bin/env python3
"""Canonical zero-dependency evaluation and performance harness for `nano`.

Measures BOTH task quality/coherence and native Metal throughput (prefill,
decode, speculative DFlash 2 parity/speedup, LoopCD, LSA) using only the native
`nano` binary (`tokenize --batch`, `session`, `bench`, `--version`) and Python
3.9+ standard library modules.

Subcommands:
    list                                 List built-in suites, tasks, and configs
    run CHECKPOINT [options]             Run eval suites and optionally save JSON/MD
    compare BASELINE.json CANDIDATE.json Diff two saved runs (optional CI --gate)
    report REPORT.json [--md OUT.md]     Render Markdown report from saved JSON
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Optional

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "zig-out" / "bin" / "nano"
MAX_SESSION_LINE_BYTES = 950_000
MAX_ARRAY_ACCESSES_BINARY_SEARCH = 64

BUILTIN_CONFIGS: dict[str, list[str]] = {
    "default": [],
    "3p_off": ["--loop-passes", "3", "--loopcd", "off"],
    "3p_hidden": ["--loop-passes", "3", "--loopcd", "hidden", "--loopcd-omega", "0.5"],
    "3p_logits": ["--loop-passes", "3", "--loopcd", "logits", "--loopcd-omega", "0.5"],
    "3p_adaptive": ["--loop-passes", "3", "--loopcd", "logits-adaptive", "--loopcd-omega", "0.7"],
    "2p_off": ["--loop-passes", "2", "--loopcd", "off"],
    "2p_hidden": ["--loop-passes", "2", "--loopcd", "hidden", "--loopcd-omega", "0.25"],
    "2p_adaptive": [
        "--loop-passes",
        "2",
        "--loopcd",
        "logits-adaptive",
        "--loopcd-omega",
        "0.7",
        "--loopcd-weak-pass",
        "1",
    ],
}

CORE_TASK_SPECS: list[dict[str, Any]] = [
    {
        "id": "coding_merge_intervals",
        "suite": "core",
        "category": "coding",
        "max_tokens": 1024,
        "description": "Write O(n log n) Python merge_intervals(intervals) and execute 7 edge-case tests",
    },
    {
        "id": "coding_binary_search",
        "suite": "core",
        "category": "coding",
        "max_tokens": 1024,
        "description": "Write O(log n) Python first_occurrence(nums, target) with instrumented read-count check",
    },
    {
        "id": "math_mod7",
        "suite": "core",
        "category": "math",
        "max_tokens": 1280,
        "description": "Solve n^2 + 3n + 11 == 0 (mod 7) for 1 <= n <= 25 via FINAL_ANSWER",
    },
    {
        "id": "math_batches",
        "suite": "core",
        "category": "math",
        "max_tokens": 768,
        "description": "Solve exact integer batch sizes A, B, C with FINAL_ANSWER",
    },
    {
        "id": "json_extract",
        "suite": "core",
        "category": "structured",
        "max_tokens": 512,
        "description": "Extract exact release metadata fields into strict JSON",
    },
    {
        "id": "code_4k",
        "suite": "core",
        "category": "code_4k",
        "max_tokens": 1536,
        "description": "Inspect ~4,000-token Zig/Metal excerpt bundle for constant, function, and ordering reason",
    },
]

LONG_SPECS: list[dict[str, Any]] = [
    {
        "id": "needle_32k",
        "suite": "long",
        "category": "long_context",
        "context": 33792,
        "haystack_tokens": 32000,
        "passkey": "739482",
        "turn1_max": 512,
        "turn2_max": 512,
        "description": "32K multi-turn needle retrieval + technical summary (auto LSA)",
    },
    {
        "id": "needle_64k",
        "suite": "long",
        "category": "long_context",
        "context": 66560,
        "haystack_tokens": 64500,
        "passkey": "481926",
        "turn1_max": 512,
        "turn2_max": 512,
        "description": "64K multi-turn needle retrieval + technical summary (auto LSA)",
    },
    {
        "id": "needle_131k",
        "suite": "long",
        "category": "long_context",
        "context": 131072,
        "haystack_tokens": 129500,
        "passkey": "952614",
        "turn1_max": 512,
        "turn2_max": 512,
        "description": "131K multi-turn needle retrieval + technical summary (auto LSA)",
    },
]

SCALE_SPECS: list[dict[str, Any]] = [
    {
        "id": "needle_256k",
        "suite": "scale",
        "category": "scale_context",
        "context": 262144,
        "haystack_tokens": 250000,
        "passkey": "614829",
        "turn1_max": 512,
        "turn2_max": 512,
        "description": "256K multi-turn needle retrieval + technical summary (auto LSA)",
    },
    {
        "id": "smoke_512k",
        "suite": "scale",
        "category": "scale_smoke",
        "context": 524288,
        "haystack_tokens": 32768,
        "passkey": None,
        "turn1_max": 32,
        "turn2_max": 0,
        "description": "512K KV allocation & 64-bit addressing smoke test (32K prompt, auto LSA)",
    },
    {
        "id": "smoke_1m",
        "suite": "scale",
        "category": "scale_smoke",
        "context": 1048576,
        "haystack_tokens": 32768,
        "passkey": None,
        "turn1_max": 32,
        "turn2_max": 0,
        "description": "1M KV allocation & 64-bit addressing smoke test (32K prompt, auto LSA)",
    },
]

BENCH_SPECS: list[dict[str, Any]] = [
    {
        "id": "bench_4k",
        "suite": "bench",
        "category": "synthetic_bench",
        "prompt_tokens": 4096,
        "context": 4200,
        "decode_tokens": 64,
        "repeats": 3,
        "description": "Raw synthetic nano bench at 4,096 prompt tokens (context 4,200)",
    },
    {
        "id": "bench_16k",
        "suite": "bench",
        "category": "synthetic_bench",
        "prompt_tokens": 16384,
        "context": 16512,
        "decode_tokens": 64,
        "repeats": 3,
        "description": "Raw synthetic nano bench at 16,384 prompt tokens (context 16,512)",
    },
    {
        "id": "bench_32k",
        "suite": "bench",
        "category": "synthetic_bench",
        "prompt_tokens": 32768,
        "context": 32900,
        "decode_tokens": 64,
        "repeats": 3,
        "description": "Raw synthetic nano bench at 32,768 prompt tokens (context 32,900)",
    },
]

ALL_SUITES: dict[str, list[dict[str, Any]]] = {
    "core": CORE_TASK_SPECS,
    "long": LONG_SPECS,
    "scale": SCALE_SPECS,
    "bench": BENCH_SPECS,
}


# ---- Self-contained prompts & deterministic text corpora ----

PROMPT_MERGE_INTERVALS = """Write a Python function `merge_intervals(intervals)` that merges all overlapping or touching `[start, end]` intervals and returns a new list of merged `[start, end]` intervals sorted by start in O(n log n) time.

Requirements:
- Input elements are 2-element lists `[start, end]` with `start <= end`, possibly unsorted, empty, nested, negative, or touching (`[1, 4]` and `[4, 5]` merge to `[1, 5]`).
- Do not mutate the caller's nested interval lists.
- Put your complete Python implementation in a fenced ```python block."""

PROMPT_BINARY_SEARCH = """Write a Python function `first_occurrence(nums, target)` that finds the smallest index `i` where `nums[i] == target` in a sorted non-decreasing sequence `nums`, or returns `-1` if `target` is absent.

Requirements:
- Must run in O(log n) element reads using binary search; assume `nums` only supports `len(nums)` and indexing `nums[i]` (do not convert to list or linearly scan).
- Handle empty sequences, duplicates (`[2, 2, 2, 3]` with target `2` -> `0`), missing targets below/between/above values, and negative numbers.
- Put your complete Python implementation in a fenced ```python block."""

PROMPT_MOD7 = """Find all positive integers n with 1 <= n <= 25 such that:
    n^2 + 3n + 11
is divisible by 7.

Show the modular arithmetic residue table for n mod 7 = 0, 1, 2, 3, 4, 5, 6, then list all matching positive integers n <= 25 in ascending order.
End your reply on a new line with the exact format:
FINAL_ANSWER: n1, n2, ..."""

PROMPT_BATCHES = """A warehouse ships three batches of accelerators totaling 98 units across Batch A, Batch B, and Batch C.
- Batch B has 4 more than twice as many units as Batch A.
- Batch C has 10 fewer units than Batch B.

Solve step by step for the exact integer counts A, B, and C, and verify the sum equals 98.
End your reply on a new line with the exact format:
FINAL_ANSWER: A=<int>, B=<int>, C=<int>"""

PROMPT_JSON_EXTRACT = """Extract the facts from the release snippet below into a single valid JSON object.

Snippet:
In v0.3.1, single-weight-read BF16 MMA dual-unembed kernels preserve exact greedy parity (exact_parity is true), reaching 133.26 decode tok/s on 3p_hidden math while accelerating the three contrasted LoopCD modes: hidden, logits, and logits-adaptive.

Return ONLY valid JSON (no markdown fences, no commentary) with exact keys:
- "release": string
- "decode_tok_s": number
- "exact_parity": boolean
- "modes": array of strings"""

CODE_4K_EXCERPTS: list[tuple[str, str]] = [
    (
        "src/mega/ring_state.zig",
        """pub const SpecRingConfig = struct {
    capacity: u32,
    window: u32,
    conv_taps: u32,
    head_align: u32 = 8,
};

pub fn advanceRingHead(pos: u32, delta: u32, cap: u32) u32 {
    std.debug.assert(cap > 0);
    return (pos + delta) % cap;
}

pub fn slotDistance(front: u32, back: u32, cap: u32) u32 {
    return if (front >= back) front - back else (cap - back) + front;
}""",
    ),
    (
        "src/mega/lsa_quant.zig",
        """pub fn quantizeKeySlotQ4(src: []const f32, dst_q: []u8, scale_out: *f32) void {
    var absmax: f32 = 0.0;
    for (src) |v| absmax = @max(absmax, @abs(v));
    const scale = if (absmax > 0.0) absmax / 7.0 else 1.0;
    scale_out.* = scale;
    const inv = 1.0 / scale;
    var i: usize = 0;
    while (i + 1 < src.len) : (i += 2) {
        const q0: i8 = @intFromFloat(@round(std.math.clamp(src[i] * inv, -7.0, 7.0)));
        const q1: i8 = @intFromFloat(@round(std.math.clamp(src[i + 1] * inv, -7.0, 7.0)));
        const lo: u8 = @as(u8, @intCast(q0 + 8)) & 0x0F;
        const hi: u8 = (@as(u8, @intCast(q1 + 8)) & 0x0F) << 4;
        dst_q[i / 2] = lo | hi;
    }
}""",
    ),
    (
        "src/mega/moe_router.zig",
        """pub fn selectTopKExperts(logits: []const f32, top_k: usize, out_ids: []u16, out_weights: []f32) void {
    std.debug.assert(out_ids.len >= top_k and out_weights.len >= top_k);
    var sum: f32 = 0.0;
    for (0..top_k) |slot| {
        var best_idx: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (logits, 0..) |val, idx| {
            var seen = false;
            for (out_ids[0..slot]) |prev| {
                if (prev == idx) seen = true;
            }
            if (!seen and val > best_val) {
                best_val = val;
                best_idx = idx;
            }
        }
        out_ids[slot] = @intCast(best_idx);
        out_weights[slot] = @exp(best_val);
        sum += out_weights[slot];
    }
    const norm = if (sum > 0.0) 1.0 / sum else 1.0;
    for (out_weights[0..top_k]) |*w| w.* *= norm;
}""",
    ),
    (
        "src/mega/loopcd_head.zig",
        """pub fn contrastHiddenStates(strong: []const f32, weak: []const f32, out: []f32, omega: f32) void {
    std.debug.assert(strong.len == weak.len and weak.len == out.len);
    for (strong, weak, out) |s, w, *dst| {
        const diff = s - w;
        const mixed = s + omega * diff;
        dst.* = roundToBf16F32(mixed);
    }
}

pub fn softcapValue(x: f32, cap: f32) f32 {
    if (cap <= 0.0) return x;
    return cap * std.math.tanh(x / cap);
}""",
    ),
    (
        "src/mega/prefill_sched.zig",
        """pub const PrefillBatchPlan = struct {
    chunk_tokens: u32,
    max_context: u32,
    use_bf16_mma: bool,
    dcut_enabled: bool,

    pub fn numChunks(self: PrefillBatchPlan, prompt_len: u32) u32 {
        return (prompt_len + self.chunk_tokens - 1) / self.chunk_tokens;
    }
};""",
    ),
    (
        "src/mega/dflash_selector.zig",
        """pub fn chooseVerifyDepth(confidence: []const f32, verify_ms: []const f32, draft_ms: f32) usize {
    var best_r: usize = 1;
    var best_rate: f64 = 0.0;
    var expected_prefix: f64 = 1.0;
    var tail_prob: f64 = 1.0;
    for (1..confidence.len + 1) |r| {
        if (r > 1) {
            tail_prob *= @floatCast(confidence[r - 2]);
            expected_prefix += tail_prob;
        }
        const total_ms = @as(f64, @floatCast(draft_ms)) + @as(f64, @floatCast(verify_ms[r - 1]));
        const rate = expected_prefix / @max(total_ms, 1e-6);
        if (rate > best_rate) {
            best_rate = rate;
            best_r = r;
        }
    }
    return best_r;
}""",
    ),
]

CODE_4K_ANCHOR_SNIPPET = """pub const RING_GUARD_SLOTS: u32 = 64;

pub const SpeculativeState = struct {
    ring_head: u32,
    committed_pos: u32,
    lsa_slot_valid: []u8,

    pub fn clearLsaSlotValid(self: *SpeculativeState, from_pos: u32, to_pos: u32) void {
        var pos = from_pos;
        while (pos < to_pos) : (pos += 1) {
            self.lsa_slot_valid[pos % self.lsa_slot_valid.len] = 0;
        }
    }

    pub fn rollbackRejectedSuffix(self: *SpeculativeState, accepted_end: u32) void {
        // Must clear LSA validity flags before rewinding ring_head so subsequent
        // attention queries never read stale uninitialized or overwritten slots.
        self.clearLsaSlotValid(accepted_end, self.ring_head);
        self.ring_head = accepted_end;
        self.committed_pos = accepted_end;
    }
};"""

CODE_4K_QUESTION = """Answer these three questions based strictly on the Zig excerpts above:
1. What is the exact numeric value of `RING_GUARD_SLOTS`?
2. Which function invokes `clearLsaSlotValid` before resetting `ring_head`?
3. Explain in 2-3 sentences why calling `clearLsaSlotValid` before resetting `ring_head` matters when rolling back rejected speculative tokens.

End your reply on a new line with the exact format:
FINAL_ANSWER: RING_GUARD_SLOTS=<int>, FUNCTION=<name>"""

ARCHIVE_MODULES: list[tuple[str, str]] = [
    (
        "src/mega/exec.zig",
        "Executes lowered Metal IR graphs over resident GPU buffers, tracks committed KV position,\n"
        "schedules BF16 simdgroup_matrix decode kernels, and guards d2 gate settlements.\n"
        "Attention dispatches check valid_slots <= context before touching ring storage.",
    ),
    (
        "src/mega/dflash2.zig",
        "Parallel block-diffusion speculative drafter (`DFlash2DraftModel`) proposing up to 7 tokens\n"
        "per pass from captured target residual layers, followed by AngelSpec D-cut tail pruning\n"
        "and exact greedy token verification against the Inkling MoE verifier.",
    ),
    (
        "src/mega/inkling2_loop.zig",
        "Lowers thinkingmachines/Inkling-2-Mini loop schedules (prelude, repeated core passes, coda)\n"
        "including depth attention, short causal convolution rings, 128 routed experts, and\n"
        "Loop Contrastive Decoding (hidden, logits, logits-adaptive) across weak and strong passes.",
    ),
    (
        "src/mega/kv_cartridge.zig",
        "Serializes and restores persistent `.nkv` KV cache cartridges (FP32 or BF16 payload)\n"
        "along with metadata hashes and exact position markers so prefills stay resumable.",
    ),
    (
        "src/mega/knobs.zig",
        "Validates runtime optimization knobs loaded from preset JSON files (`inkling.json`,\n"
        "`exact.json`, `mlx-prefill.json`) covering `bf16_verify_mma`, `dflash_dcut`, `lsa`,\n"
        "`lsa_cross_loop_reuse`, and grouped MoE kernel dispatch flags.",
    ),
    (
        "src/mega/frontend.zig",
        "Selects the model loader and chat template (`inkling2_loop`, `lfm2`, `lfm3`, `qwen3_5`)\n"
        "from `config.json` without duplicating execution machinery.",
    ),
    (
        "src/mega/tokenizer.zig",
        "Native zero-Python BPE tokenizer supporting Llama 3 and Unicode 16.0.0 `o200k_base`\n"
        "pre-tokenization regexes with streaming UTF-8 output and batch binary IPC.",
    ),
    (
        "src/mega/session.zig",
        "JSONL persistent session loop over `Engine` and `DFlash2DraftModel`. Commits every\n"
        "generated token (including EOS), validates preflight budget, and tracks KV bytes.",
    ),
    (
        "src/mega/inkling_fast.metal",
        "Hardware BF16 `simdgroup_matrix` Metal kernels including single-weight-read dual-unembed\n"
        "LoopCD kernels (`mk_bf16_mma_m8n16_softcap_loopcd_direct`, m8n32), grouped MoE\n"
        "NVFP4/MXFP8 decode, top-k group tile routing, and conv ring batch updates.",
    ),
    (
        "src/mega/dflash2.metal",
        "Draft network Metal shaders: fused multi-projection Q+K+V MMA kernels, SwiGLU MMA,\n"
        "grouped dynamic convolution, GQA4 attention, and GPU top-16 confidence selection.",
    ),
    (
        "src/mega/lfm3.zig",
        "Looped/hybrid Liquid AI LFM3 graph builder combining gated short convolutions and GQA\n"
        "attention layers with unified preset tuning via `exact.json`.",
    ),
    (
        "src/mega/cli.zig",
        "Standalone user CLI (`run`, `chat`, `tokenize`, `detokenize`, `--version`) auto-picking\n"
        "`presets/inkling.json` for `inkling2_loop` and streaming greedy `--draft` replies.",
    ),
    (
        "src/mega/sample.zig",
        "Deterministic host-side top-k, top-p, temperature, and repetition penalty sampler\n"
        "with explicit PCG state for non-greedy runs.",
    ),
    (
        "src/mega/loader.zig",
        "Memory-maps safetensors weight shards, validates tensor dtypes/shapes, and builds\n"
        "resident GPU weight tables with SafeAllocator leak accounting.",
    ),
    (
        "src/mega/plan.zig",
        "Compile-time and runtime execution plan emitter connecting IR buffers, barrier hazards,\n"
        "and Metal command encoder lifecycles cleanly.",
    ),
    (
        "src/mega/bench.zig",
        "Warmup-excluded decode and prefill microbenchmark harness verifying deterministic\n"
        "greedy/teacher-forced prediction checksums across repeated runs.",
    ),
    (
        "src/mega/ir.zig",
        "Minimal typed IR defining linear projections, norms, softcap heads, LoopCD contrast ops,\n"
        "relative/depth attention, and quantized MoE routing primitives.",
    ),
    (
        "src/mega/gpu.zig",
        "Apple Silicon Metal command buffer, pipeline state cache, shared memory mapping,\n"
        "and barrier synchronization wrapper with explicit resource ownership.",
    ),
    (
        "src/resident_plan.zig",
        "Core resident KV & slot mapping layout shared with Lean 4 formal verification bindings\n"
        "to guarantee non-overlapping ring buffer indices and stripe reconstruction bounds.",
    ),
    (
        "proofs/Inkling.lean",
        "Lean 4 specification containing 65 machine-checked theorems covering Inkling loop\n"
        "schedules, softcap contrast bounds, and speculative ring-slot invariants.",
    ),
    (
        "proofs/Speculative.lean",
        "Lean 4 proofs for DFlash 2 acceptance prefixes, D-cut concave utility monotonicity,\n"
        "and exact greedy parity preservation under tail truncation.",
    ),
    (
        "tools/nano_eval.py",
        "Canonical zero-dependency evaluation and throughput regression suite validating\n"
        "functional code/math/JSON/long-context tasks alongside Metal token/s metrics.",
    ),
]

TURN2_TEXT = (
    "Now write a concise technical summary of the main modules, KV cache mechanisms, "
    "and execution kernels described in the archive above."
)

ARCHIVE_KEYWORDS = (
    "ring",
    "cache",
    "quant",
    "router",
    "scheduler",
    "metal",
    "barrier",
    "attention",
    "speculative",
    "cartridge",
    "kv",
    "moe",
    "prefill",
    "decode",
    "dflash",
    "loopcd",
    "lsa",
)

LEAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SafeAllocator", re.compile(r"\bsafeallocator\b", re.IGNORECASE)),
    ("leaked", re.compile(r"\bleak(?:ed|s|ing)?\b", re.IGNORECASE)),
    ("panic", re.compile(r"\bpanic\b", re.IGNORECASE)),
    ("reached_unreachable", re.compile(r"reached unreachable", re.IGNORECASE)),
    ("error_colon", re.compile(r"(?:^|\s)(?:error(?:\([^)]*\))?|session failed)\s*:", re.IGNORECASE)),
]


# ---- Utility helpers ----


def parse_csv_list(values: Optional[list[str]]) -> list[str]:
    out: list[str] = []
    for item in values or []:
        for part in item.split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def detect_model_type(checkpoint: Path) -> str:
    cfg = checkpoint / "config.json"
    if not cfg.is_file():
        return "unknown"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        mt = data.get("model_type")
        return str(mt) if isinstance(mt, str) and mt else "unknown"
    except Exception:
        return "unknown"


def resolve_knobs_path(checkpoint: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit is not None:
        if explicit.lower() == "none":
            return None
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"knobs file not found: {explicit}")
        return p
    model_type = detect_model_type(checkpoint)
    if model_type == "inkling2_loop":
        p = REPO_ROOT / "src" / "mega" / "presets" / "inkling.json"
        return p if p.is_file() else None
    if model_type in ("lfm2", "lfm3"):
        p = REPO_ROOT / "src" / "mega" / "presets" / "exact.json"
        return p if p.is_file() else None
    return None


def safe_path_label(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
        return str(rel)
    except Exception:
        return path.name


def read_nano_version(binary: Path) -> str:
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        out = (proc.stdout or proc.stderr).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


@contextlib.contextmanager
def inference_lock(lock_path: Optional[Path]):
    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---- Native Tokenizer Bridge (`nano tokenize MODEL --batch FILE`) ----


class NanoTokenizer:
    """Zero-dependency Python bridge over `nano tokenize MODEL --batch`."""

    def __init__(self, binary: Path, checkpoint: Path, timeout: int = 120):
        self.binary = Path(binary)
        self.checkpoint = Path(checkpoint)
        self.timeout = timeout
        self._chat_cache: dict[tuple[str, bool], list[int]] = {}
        self._encode_cache: dict[str, list[int]] = {}
        self._next_turn_ids_cache: dict[tuple[bool, str, bool], list[int]] = {}

    @staticmethod
    def rec_encode(text: str) -> bytes:
        raw = text.encode("utf-8")
        return b"E" + struct.pack("<I", len(raw)) + raw

    @staticmethod
    def rec_decode(ids: Iterable[int]) -> bytes:
        id_list = [int(x) for x in ids]
        return b"D" + struct.pack("<I", len(id_list)) + struct.pack(f"<{len(id_list)}I", *id_list)

    @staticmethod
    def _rec_json(kind: bytes, payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return kind + struct.pack("<I", len(raw)) + raw

    @classmethod
    def rec_chat(cls, messages: list[dict[str, str]], *, thinking: bool = False, generate: bool = True) -> bytes:
        return cls._rec_json(b"C", {"messages": messages, "thinking": thinking, "generate": generate})

    @classmethod
    def rec_next_turn(cls, *, closed: bool, user: str, thinking: bool = False) -> bytes:
        return cls._rec_json(b"N", {"closed": closed, "user": user, "thinking": thinking})

    def batch(self, records: list[bytes]) -> list[dict[str, Any]]:
        if not records:
            return []
        with tempfile.TemporaryDirectory(prefix="nano_eval_tok_") as tmpdir:
            batch_path = Path(tmpdir) / "batch.bin"
            batch_path.write_bytes(b"".join(records))
            proc = subprocess.run(
                [str(self.binary), "tokenize", str(self.checkpoint), "--batch", str(batch_path)],
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"nano tokenize --batch failed ({proc.returncode}): {stderr}")
        lines = [ln for ln in proc.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if len(lines) != len(records):
            raise RuntimeError(f"nano tokenize --batch returned {len(lines)} lines for {len(records)} records")
        return [json.loads(ln) for ln in lines]

    def encode_many(self, texts: list[str]) -> list[list[int]]:
        missing_idx = [i for i, t in enumerate(texts) if t not in self._encode_cache]
        if missing_idx:
            recs = [self.rec_encode(texts[i]) for i in missing_idx]
            resps = self.batch(recs)
            for idx, r in zip(missing_idx, resps):
                self._encode_cache[texts[idx]] = [int(x) for x in r["ids"]]
        return [list(self._encode_cache[t]) for t in texts]

    def decode_many(self, token_lists: list[list[int]]) -> list[dict[str, str]]:
        if not token_lists:
            return []
        resps = self.batch([self.rec_decode(toks) for toks in token_lists])
        return [
            {
                "text": str(r.get("text", "")),
                "special": str(r.get("special", "")),
                "stream": str(r.get("stream", "")),
            }
            for r in resps
        ]

    def chat_many(self, user_prompts: list[str], *, thinking: bool = False) -> list[list[int]]:
        missing_idx = [i for i, p in enumerate(user_prompts) if (p, thinking) not in self._chat_cache]
        if missing_idx:
            recs = [
                self.rec_chat([{"role": "user", "content": user_prompts[i]}], thinking=thinking, generate=True)
                for i in missing_idx
            ]
            resps = self.batch(recs)
            for idx, r in zip(missing_idx, resps):
                self._chat_cache[(user_prompts[idx], thinking)] = [int(x) for x in r["ids"]]
        return [list(self._chat_cache[(p, thinking)]) for p in user_prompts]

    def next_turn_ids_pair(self, user: str, *, thinking: bool = False) -> dict[bool, list[int]]:
        need = [closed for closed in (True, False) if (closed, user, thinking) not in self._next_turn_ids_cache]
        if need:
            rendered = self.batch([self.rec_next_turn(closed=c, user=user, thinking=thinking) for c in need])
            encoded = self.encode_many([str(r["text"]) for r in rendered])
            for c, ids in zip(need, encoded):
                self._next_turn_ids_cache[(c, user, thinking)] = ids
        return {c: list(self._next_turn_ids_cache[(c, user, thinking)]) for c in (True, False)}


# ---- Output Extraction, Coherence, Parity, and Leak Checks ----


def extract_visible_answer(plain_text: str, special_text: str = "") -> str:
    """Extract user-visible text, stripping Inkling thinking trace when present."""
    if "<|content_thinking|>" in special_text:
        idx = special_text.rfind("<|content_text|>")
        if idx < 0:
            return ""
        tail = special_text[idx + len("<|content_text|>") :]
        for stop_tag in ("<|end_message|>", "<|content_model_end_sampling|>"):
            end_idx = tail.find(stop_tag)
            if end_idx >= 0:
                tail = tail[:end_idx]
        return tail
    return plain_text


def check_coherence(tokens: list[int], text: str) -> tuple[bool, str]:
    """Detect degenerate repetition loops, empty replies, or binary corruption."""
    if not tokens:
        return False, "empty_tokens"
    if not text or not text.strip():
        return False, "empty_text"
    if "\x00" in text:
        return False, "nul_byte"
    repl_count = text.count("\ufffd")
    if repl_count >= 5:
        return False, f"replacement_chars:{repl_count}"

    n_tok = len(tokens)
    for gram in (4, 6, 8, 12, 16):
        if n_tok < gram * 4:
            continue
        i = 0
        while i + gram <= n_tok:
            block = tokens[i : i + gram]
            run = 1
            while i + (run + 1) * gram <= n_tok and tokens[i + run * gram : i + (run + 1) * gram] == block:
                run += 1
            if run >= 4:
                return False, f"repeated_{gram}gram_x{run}"
            i += 1

    if n_tok >= 80:
        tail_len = max(48, (n_tok * 2) // 5)
        tail = tokens[-tail_len:]
        total_4grams = max(1, len(tail) - 3)
        unique_4grams = len({tuple(tail[j : j + 4]) for j in range(total_4grams)})
        ratio = unique_4grams / total_4grams
        if ratio < 0.30:
            return False, f"low_tail_diversity:{ratio:.3f}"

    line_counts: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 20:
            continue
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] >= 4:
            return False, f"repeated_line_x{line_counts[line]}"

    return True, "clean"


def check_stderr_leaks(stderr: str) -> tuple[bool, list[str]]:
    """Return (True, []) if clean, else (False, matched_reasons)."""
    if not stderr:
        return True, []
    reasons: list[str] = []
    for name, pat in LEAK_PATTERNS:
        if pat.search(stderr):
            reasons.append(name)
    return len(reasons) == 0, reasons


def compare_token_parity(base_tokens: list[int], draft_tokens: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
    if base_tokens == draft_tokens:
        return True, None
    for idx, (bt, dt) in enumerate(zip(base_tokens, draft_tokens)):
        if bt != dt:
            return False, {"index": idx, "base": bt, "draft": dt}
    return False, {"length_mismatch": [len(base_tokens), len(draft_tokens)]}


# ---- Task Graders ----


@dataclass
class GradeResult:
    passed: bool
    reason: str
    detail: dict[str, Any]


def _extract_python_candidates(text: str, fn_name: str) -> list[str]:
    candidates: list[str] = []
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        if f"def {fn_name}" in block:
            candidates.append(block.strip())
    for block in fenced:
        s = block.strip()
        if s and s not in candidates:
            candidates.append(s)
    m = re.search(rf"(def\s+{re.escape(fn_name)}\s*\(.*)", text, flags=re.DOTALL)
    if m:
        snippet = m.group(1).strip()
        if "```" in snippet:
            snippet = snippet.split("```", 1)[0].strip()
        if snippet and snippet not in candidates:
            candidates.append(snippet)
    return candidates


def _run_python_sandbox(code: str, harness: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    script = f"from __future__ import annotations\nimport math\n\n{code}\n\n{harness}\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0:
        return True, (proc.stdout or "ok").strip()
    err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()
    return False, err[-1] if err else f"exit {proc.returncode}"


def grade_merge_intervals(text: str) -> GradeResult:
    candidates = _extract_python_candidates(text, "merge_intervals")
    if not candidates:
        return GradeResult(False, "no_python_code_found", {})
    harness = """
orig = [[1, 3], [2, 6]]
orig_copy = [x[:] for x in orig]
assert merge_intervals([]) == [], "empty"
assert merge_intervals([[5, 7]]) == [[5, 7]], "single"
assert merge_intervals(orig) == [[1, 6], [8, 10], [15, 18]] if False else True
assert merge_intervals(orig) == [[1, 6]], "basic_overlap"
assert orig == orig_copy, "mutated_input"
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]], "standard"
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]], "touching"
assert merge_intervals([[1, 10], [2, 3], [4, 8]]) == [[1, 10]], "nested"
assert merge_intervals([[-5, -2], [-3, 0], [2, 4]]) == [[-5, 0], [2, 4]], "negative"
assert merge_intervals([[8, 10], [1, 3], [2, 6], [15, 18]]) == [[1, 6], [8, 10], [15, 18]], "unsorted"
print("PASS_MERGE_7")
"""
    last_err = "no_candidate_passed"
    for idx, code in enumerate(candidates):
        ok, msg = _run_python_sandbox(code, harness)
        if ok and "PASS_MERGE_7" in msg:
            return GradeResult(True, "all_7_tests_passed", {"candidate_index": idx})
        last_err = msg
    return GradeResult(False, f"python_test_failed:{last_err}", {"candidates_tried": len(candidates)})


def grade_binary_search(text: str) -> GradeResult:
    candidates = _extract_python_candidates(text, "first_occurrence")
    if not candidates:
        return GradeResult(False, "no_python_code_found", {})
    harness = f"""
class ReadCountingSeq:
    def __init__(self, n):
        self._n = n
        self.reads = 0
    def __len__(self):
        return self._n
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise TypeError("slicing forbidden in O(log n) check")
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        self.reads += 1
        if self.reads > 200:
            raise RuntimeError(f"excessive_reads:{{self.reads}}")
        if idx < 70000:
            return idx // 2
        if idx < 70500:
            return 35000
        return idx

assert first_occurrence([], 5) == -1, "empty"
assert first_occurrence([5], 5) == 0, "single_hit"
assert first_occurrence([5], 3) == -1, "single_miss"
assert first_occurrence([2, 2, 2, 3], 2) == 0, "dup_start"
assert first_occurrence([1, 2, 2, 2, 3], 2) == 1, "dup_mid"
assert first_occurrence([1, 2, 3, 3, 3], 3) == 2, "dup_end"
assert first_occurrence([-10, -5, -5, 0, 4], -5) == 1, "negative_dup"
assert first_occurrence([1, 3, 5, 7], 4) == -1, "missing_gap"
assert first_occurrence([1, 3, 5], 0) == -1, "missing_low"
assert first_occurrence([1, 3, 5], 9) == -1, "missing_high"

seq = ReadCountingSeq(100000)
pos = first_occurrence(seq, 35000)
assert pos == 70000, f"wrong_index:{{pos}}"
assert seq.reads <= {MAX_ARRAY_ACCESSES_BINARY_SEARCH}, f"too_many_reads:{{seq.reads}}"
miss_seq = ReadCountingSeq(100000)
assert first_occurrence(miss_seq, 70123) == -1, "wrong_miss_large"
assert miss_seq.reads <= {MAX_ARRAY_ACCESSES_BINARY_SEARCH}, f"too_many_miss_reads:{{miss_seq.reads}}"
print(f"PASS_BSEARCH:{{seq.reads}}")
"""
    last_err = "no_candidate_passed"
    for idx, code in enumerate(candidates):
        ok, msg = _run_python_sandbox(code, harness)
        if ok and msg.startswith("PASS_BSEARCH:"):
            reads = int(msg.split(":", 1)[1])
            return GradeResult(True, "all_tests_and_log_reads_passed", {"candidate_index": idx, "reads_100k": reads})
        last_err = msg
    return GradeResult(False, f"python_test_failed:{last_err}", {"candidates_tried": len(candidates)})


def _extract_final_answer_line(text: str) -> Optional[str]:
    matches = re.findall(r"(?im)^[*`\s]*FINAL_ANSWER\s*[:：]\s*(.+?)[*`\s]*$", text)
    if not matches:
        return None
    return matches[-1].strip()


def grade_math_mod7(text: str) -> GradeResult:
    expected = [2, 9, 16, 23]
    final_line = _extract_final_answer_line(text)
    if final_line is None:
        return GradeResult(False, "missing_FINAL_ANSWER_line", {})
    cleaned = re.sub(r"[`*]", "", final_line)
    nums = [int(x) for x in re.findall(r"-?\d+", cleaned)]
    if sorted(nums) == expected and len(nums) == len(expected):
        return GradeResult(True, "exact_mod7_roots", {"answers": nums})
    return GradeResult(False, f"wrong_mod7_answers:{nums}", {"final_line": final_line})


def grade_math_batches(text: str) -> GradeResult:
    final_line = _extract_final_answer_line(text)
    if final_line is None:
        return GradeResult(False, "missing_FINAL_ANSWER_line", {})
    cleaned = re.sub(r"[`*]", "", final_line)
    ma = re.search(r"\bA\s*=\s*(-?\d+)\b", cleaned, flags=re.IGNORECASE)
    mb = re.search(r"\bB\s*=\s*(-?\d+)\b", cleaned, flags=re.IGNORECASE)
    mc = re.search(r"\bC\s*=\s*(-?\d+)\b", cleaned, flags=re.IGNORECASE)
    if not (ma and mb and mc):
        return GradeResult(False, f"unparsed_ABC:{final_line}", {})
    got = (int(ma.group(1)), int(mb.group(1)), int(mc.group(1)))
    if got == (20, 44, 34):
        return GradeResult(True, "exact_batches_20_44_34", {"A": 20, "B": 44, "C": 34})
    return GradeResult(False, f"wrong_batches:{got}", {"got": list(got)})


def grade_json_extract(text: str) -> GradeResult:
    raw = text.strip()
    raw_json_ok = False
    parsed: Any = None
    try:
        parsed = json.loads(raw)
        raw_json_ok = True
    except Exception:
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1).strip())
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        return GradeResult(False, "invalid_or_non_object_json", {"raw_json": raw_json_ok})
    expected_keys = {"release", "decode_tok_s", "exact_parity", "modes"}
    if set(parsed.keys()) != expected_keys:
        return GradeResult(False, f"key_mismatch:{sorted(parsed.keys())}", {"raw_json": raw_json_ok})
    if parsed["release"] != "v0.3.1":
        return GradeResult(False, f"wrong_release:{parsed['release']!r}", {"raw_json": raw_json_ok})
    tok_s = parsed["decode_tok_s"]
    if isinstance(tok_s, bool) or not isinstance(tok_s, (int, float)) or not math.isclose(float(tok_s), 133.26, rel_tol=0, abs_tol=1e-6):
        return GradeResult(False, f"wrong_decode_tok_s:{tok_s!r}", {"raw_json": raw_json_ok})
    if not isinstance(parsed["exact_parity"], bool) or parsed["exact_parity"] is not True:
        return GradeResult(False, f"wrong_exact_parity:{parsed['exact_parity']!r}", {"raw_json": raw_json_ok})
    if parsed["modes"] != ["hidden", "logits", "logits-adaptive"]:
        return GradeResult(False, f"wrong_modes:{parsed['modes']!r}", {"raw_json": raw_json_ok})
    return GradeResult(
        True,
        "exact_json_match" if raw_json_ok else "exact_json_match_fenced_fallback",
        {"raw_json": raw_json_ok},
    )


def grade_code_4k(text: str) -> GradeResult:
    final_line = _extract_final_answer_line(text)
    used_fallback = False
    slots_val: Optional[int] = None
    fn_val: Optional[str] = None
    if final_line is not None:
        cleaned = re.sub(r"[`*'\"]", "", final_line)
        ms = re.search(r"RING_GUARD_SLOTS\s*=\s*(\d+)", cleaned, flags=re.IGNORECASE)
        mf = re.search(r"FUNCTION\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", cleaned, flags=re.IGNORECASE)
        if ms:
            slots_val = int(ms.group(1))
        if mf:
            fn_val = mf.group(1)
    if slots_val is None or fn_val is None:
        used_fallback = True
        if slots_val is None and re.search(r"\bRING_GUARD_SLOTS\b.{0,24}\b64\b", text, flags=re.DOTALL):
            slots_val = 64
        if fn_val is None and "rollbackRejectedSuffix" in text:
            fn_val = "rollbackRejectedSuffix"

    if slots_val != 64:
        return GradeResult(False, f"wrong_or_missing_RING_GUARD_SLOTS:{slots_val}", {"used_fallback": used_fallback})
    if fn_val != "rollbackRejectedSuffix":
        return GradeResult(False, f"wrong_or_missing_FUNCTION:{fn_val}", {"used_fallback": used_fallback})

    body = re.sub(r"(?im)^[*`\s]*FINAL_ANSWER\s*[:：].*$", "", text).strip()
    if len(body) < 40:
        return GradeResult(False, "missing_ordering_explanation", {"used_fallback": used_fallback})
    return GradeResult(
        True,
        "exact_code_4k_answer",
        {"slots": slots_val, "function": fn_val, "used_fallback": used_fallback},
    )


def grade_long_multiturn(passkey: Optional[str], t1_text: str, t2_text: Optional[str], t1_coherent: bool, t2_coherent: bool) -> GradeResult:
    if not t1_coherent:
        return GradeResult(False, "turn1_incoherent", {})
    if passkey is not None and passkey not in t1_text:
        return GradeResult(False, f"needle_missing:{passkey}", {"needle_pass": False})
    if t2_text is not None:
        if not t2_coherent:
            return GradeResult(False, "turn2_incoherent", {"needle_pass": True if passkey else None})
        lower = t2_text.lower()
        hits = [kw for kw in ARCHIVE_KEYWORDS if kw in lower]
        if len(hits) < 2:
            return GradeResult(False, f"turn2_insufficient_keywords:{hits}", {"needle_pass": True if passkey else None, "keyword_hits": hits})
        return GradeResult(
            True,
            "needle_and_summary_passed" if passkey else "smoke_and_summary_passed",
            {"needle_pass": True if passkey else None, "keyword_hits": hits},
        )
    return GradeResult(True, "smoke_passed", {"needle_pass": True if passkey else None})


CORE_GRADERS: dict[str, Callable[[str], GradeResult]] = {
    "coding_merge_intervals": grade_merge_intervals,
    "coding_binary_search": grade_binary_search,
    "math_mod7": grade_math_mod7,
    "math_batches": grade_math_batches,
    "json_extract": grade_json_extract,
    "code_4k": grade_code_4k,
}


# ---- Prompt Builders (100% self-contained) ----


def build_code_4k_prompt(tok: NanoTokenizer, *, thinking: bool = False, target_tokens: int = 4050) -> tuple[str, list[int]]:
    def render_n(n_excerpts: int) -> str:
        anchor_pos = n_excerpts // 2
        blocks: list[str] = []
        idx = 0
        for i in range(n_excerpts):
            if i == anchor_pos:
                blocks.append(f"### Excerpt {len(blocks) + 1} (src/mega/spec_rollback.zig)\n```zig\n{CODE_4K_ANCHOR_SNIPPET}\n```")
            mod, body = CODE_4K_EXCERPTS[idx % len(CODE_4K_EXCERPTS)]
            blocks.append(f"### Excerpt {len(blocks) + 1} ({mod})\n```zig\n{body}\n```")
            idx += 1
        return "Analyze the following Zig code excerpts:\n\n" + "\n\n".join(blocks) + "\n\n" + CODE_4K_QUESTION

    probe_counts = [10, 14, 18, 22]
    probe_texts = [render_n(c) for c in probe_counts]
    probe_ids = tok.chat_many(probe_texts, thinking=thinking)
    counts = {c: (t, ids) for c, t, ids in zip(probe_counts, probe_texts, probe_ids)}

    for c in probe_counts:
        if 3980 <= len(counts[c][1]) <= 4140:
            return counts[c]

    # Estimate linear slope tokens/excerpt and search tightly in one follow-up batch.
    lo_c, hi_c = probe_counts[0], probe_counts[-1]
    lo_n, hi_n = len(counts[lo_c][1]), len(counts[hi_c][1])
    per_excerpt = max(1.0, (hi_n - lo_n) / max(1, hi_c - lo_c))
    est = max(4, int(round(lo_c + (target_tokens - lo_n) / per_excerpt)))
    refine_counts = [c for c in range(max(4, est - 3), est + 4) if c not in counts]
    if refine_counts:
        ref_texts = [render_n(c) for c in refine_counts]
        ref_ids = tok.chat_many(ref_texts, thinking=thinking)
        for c, t, ids in zip(refine_counts, ref_texts, ref_ids):
            counts[c] = (t, ids)
            if 3980 <= len(ids) <= 4140:
                return t, ids

    best_c = min(counts, key=lambda c: abs(len(counts[c][1]) - target_tokens))
    return counts[best_c]


WARMUP_PROMPT = (
    "System calibration check: verify KV ring alignment, barrier phase tracking, "
    "quantized weight page residency, and Metal simdgroup execution readiness.\n" * 10
    + "Reply with the single word OK."
)


def build_core_tasks(tok: NanoTokenizer, *, thinking: bool = False) -> tuple[list[int], list[dict[str, Any]]]:
    static_prompts = {
        "coding_merge_intervals": PROMPT_MERGE_INTERVALS,
        "coding_binary_search": PROMPT_BINARY_SEARCH,
        "math_mod7": PROMPT_MOD7,
        "math_batches": PROMPT_BATCHES,
        "json_extract": PROMPT_JSON_EXTRACT,
    }
    ordered_keys = ["__warmup__", *static_prompts.keys()]
    ordered_texts = [WARMUP_PROMPT, *static_prompts.values()]
    encoded = tok.chat_many(ordered_texts, thinking=thinking)
    by_id = dict(zip(ordered_keys, encoded))

    code_4k_text, code_4k_ids = build_code_4k_prompt(tok, thinking=thinking)
    prompts_map = {**static_prompts, "code_4k": code_4k_text}
    ids_map = {**by_id, "code_4k": code_4k_ids}

    tasks: list[dict[str, Any]] = []
    for spec in CORE_TASK_SPECS:
        tasks.append({
            **spec,
            "prompt_text": prompts_map[spec["id"]],
            "prompt_ids": ids_map[spec["id"]],
        })
    return by_id["__warmup__"], tasks


def build_archive_prompt(
    tok: NanoTokenizer,
    *,
    target_tokens: int,
    passkey: Optional[str],
    max_allowed_tokens: int,
    thinking: bool = False,
) -> list[int]:
    sample_entries = [
        f"=== Archive Entry #{i + 1:04d} ({mod}) ===\n{body}\n"
        for i, (mod, body) in enumerate(ARCHIVE_MODULES[:12])
    ]
    sample_ids = tok.encode_many(["\n".join(sample_entries)])[0]
    avg_toks = max(35.0, len(sample_ids) / len(sample_entries))
    n = max(6, int(round(target_tokens / avg_toks)))

    def render(n_entries: int) -> str:
        mid = n_entries // 2
        parts = ["Review the following numbered archive of systems engineering modules carefully.\n"]
        needle = (
            f"\nIMPORTANT RECORD: The reference calibration code for Project Hyperion is {passkey}. "
            "Record this exact 6-digit identifier.\n"
        )
        for i in range(n_entries):
            mod, body = ARCHIVE_MODULES[i % len(ARCHIVE_MODULES)]
            parts.append(f"=== Archive Entry #{i + 1:04d} ({mod}) ===\n{body}\n")
            if passkey is not None and i + 1 == mid:
                parts.append(needle)
        if passkey is not None:
            parts.append(
                "\nQuestion: Based strictly on the Archive Entries above, what is the reference "
                "calibration code for Project Hyperion? State ONLY the exact 6-digit number."
            )
        else:
            parts.append(
                "\nQuestion: Name three core subsystems described in the Archive Entries above in one short sentence."
            )
        return "\n".join(parts)

    lo_tol = target_tokens - 600
    hi_tol = min(target_tokens + 600, max_allowed_tokens)
    seen: dict[int, list[int]] = {}
    for _ in range(4):
        if n not in seen:
            seen[n] = tok.chat_many([render(n)], thinking=thinking)[0]
        ids = seen[n]
        if lo_tol <= len(ids) <= hi_tol:
            break
        ratio = min(target_tokens, max_allowed_tokens) / max(len(ids), 1)
        new_n = max(6, int(round(n * ratio)))
        if new_n == n:
            new_n += 1 if len(ids) < target_tokens else -1
        n = max(6, new_n)

    best_n = min(seen, key=lambda k: abs(len(seen[k]) - min(target_tokens, max_allowed_tokens)))
    ids = seen[best_n]
    if len(ids) > max_allowed_tokens:
        excess = len(ids) - max_allowed_tokens
        keep_prefix = min(128, max_allowed_tokens // 8)
        ids = ids[:keep_prefix] + ids[keep_prefix + excess :]
    if len(ids) > max_allowed_tokens:
        raise ValueError(f"built prompt len {len(ids)} > max {max_allowed_tokens}")
    return ids


# ---- Chunking & `nano session` Runners ----


def make_chunked_requests(
    turn1_ids: list[int],
    *,
    turn1_max: int,
    chunk_size: int = 24576,
    reset_first: bool = True,
    start_id: int = 1,
) -> list[dict[str, Any]]:
    if not turn1_ids:
        raise ValueError("turn1_ids must be non-empty")
    reqs: list[dict[str, Any]] = []
    req_id = start_id
    offset = 0
    while offset < len(turn1_ids):
        chunk = turn1_ids[offset : offset + chunk_size]
        is_first = offset == 0
        is_last = offset + len(chunk) >= len(turn1_ids)
        req: dict[str, Any] = {
            "id": req_id,
            "tokens": chunk,
            "max_tokens": turn1_max if is_last else 0,
        }
        if is_first and reset_first:
            req["reset"] = True
        line = json.dumps(req, separators=(",", ":"))
        if len(line.encode("utf-8")) >= MAX_SESSION_LINE_BYTES:
            raise ValueError(f"session request line exceeded {MAX_SESSION_LINE_BYTES} bytes")
        reqs.append(req)
        req_id += 1
        offset += len(chunk)
    return reqs


def build_session_cmd(
    binary: Path,
    checkpoint: Path,
    *,
    context: int,
    knobs_path: Optional[Path],
    config_flags: list[str],
    knob_overrides: list[str],
    extra_flags: list[str],
    draft: Optional[Path],
) -> list[str]:
    cmd = [str(binary), "session", str(checkpoint), "--context", str(context)]
    if knobs_path is not None:
        cmd.extend(["--knobs", str(knobs_path)])
    cmd.extend(config_flags)
    for kv in knob_overrides:
        cmd.extend(["--knob", kv])
    cmd.extend(extra_flags)
    if draft is not None:
        cmd.extend(["--draft", str(draft)])
    return cmd


def run_session_batch(cmd: list[str], requests: list[dict[str, Any]], *, timeout: int = 900) -> tuple[list[dict[str, Any]], str]:
    payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in requests) + "\n"
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nano session failed (exit {proc.returncode}): {(proc.stderr or '')[-2000:]}"
        )
    resps = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    if len(resps) != len(requests):
        raise RuntimeError(f"expected {len(requests)} session responses, got {len(resps)}")
    return resps, proc.stderr


def run_session_long_task(
    cmd: list[str],
    turn1_ids: list[int],
    *,
    turn1_max: int,
    turn2_ids_by_closed: Optional[dict[bool, list[int]]],
    turn2_max: int,
    warmup_ids: Optional[list[int]] = None,
    timeout: int = 900,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], int, str]:
    """Run a single-task chunked session; dynamically send Turn 2 matching Turn 1's `eos`."""
    reqs: list[dict[str, Any]] = []
    next_id = 0
    if warmup_ids:
        reqs.append({"id": next_id, "tokens": warmup_ids, "max_tokens": 32, "reset": True})
        next_id += 1
    chunk_reqs = make_chunked_requests(
        turn1_ids,
        turn1_max=turn1_max,
        reset_first=True,
        start_id=next_id,
    )
    reqs.extend(chunk_reqs)
    turn2_req_id = next_id + len(chunk_reqs)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    deadline = time.monotonic() + timeout
    t1_resps: list[dict[str, Any]] = []
    t2_resp: Optional[dict[str, Any]] = None
    turn2_prompt_len = 0
    try:
        for req in reqs:
            proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        proc.stdin.flush()

        for _ in range(len(reqs)):
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(cmd, timeout)
            line = proc.stdout.readline()
            if not line:
                stderr_tail = proc.stderr.read()
                raise RuntimeError(f"nano session ended early reading Turn 1: {stderr_tail[-1500:]}")
            t1_resps.append(json.loads(line))

        t1_final = t1_resps[-1]
        if turn2_ids_by_closed is not None and turn2_max > 0:
            closed = bool(t1_final.get("eos", False))
            t2_tokens = turn2_ids_by_closed[closed]
            turn2_prompt_len = len(t2_tokens)
            t2_req = {"id": turn2_req_id, "tokens": t2_tokens, "max_tokens": turn2_max}
            proc.stdin.write(json.dumps(t2_req, separators=(",", ":")) + "\n")
            proc.stdin.flush()

        proc.stdin.close()
        if turn2_ids_by_closed is not None and turn2_max > 0:
            line = proc.stdout.readline()
            if not line:
                stderr_tail = proc.stderr.read()
                raise RuntimeError(f"nano session ended early reading Turn 2: {stderr_tail[-1500:]}")
            t2_resp = json.loads(line)

        rem_timeout = max(5.0, deadline - time.monotonic())
        proc.wait(timeout=rem_timeout)
        stderr = proc.stderr.read()
        if proc.returncode != 0:
            raise RuntimeError(f"nano session failed (exit {proc.returncode}): {stderr[-2000:]}")
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if warmup_ids:
        t1_resps = t1_resps[1:]
    return t1_resps, t2_resp, turn2_prompt_len, stderr


# ---- Execution & Aggregation ----


def _summarize_arm_metrics(
    *,
    prompt_tokens: int,
    gen_tokens: int,
    prefill_seconds: float,
    decode_seconds: float,
    spec_batches: int,
    spec_proposed: int,
    spec_accepted: int,
    dispatches: int,
    eos: bool,
    kv_allocated_bytes: int,
    kv_valid_bytes: int,
) -> dict[str, Any]:
    prefill_tok_s = prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
    decode_tok_s = gen_tokens / decode_seconds if decode_seconds > 0 else 0.0
    mean_step = (gen_tokens / spec_batches) if spec_batches > 0 else (1.0 if gen_tokens > 0 else 0.0)
    acceptance_rate = (spec_accepted / spec_proposed) if spec_proposed > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "mean_step": mean_step,
        "acceptance_rate": acceptance_rate,
        "speculative_batches": spec_batches,
        "speculative_proposed": spec_proposed,
        "speculative_accepted": spec_accepted,
        "dispatches": dispatches,
        "eos": eos,
        "kv_allocated_bytes": kv_allocated_bytes,
        "kv_valid_bytes": kv_valid_bytes,
    }


def _aggregate_arm_repeats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) == 1:
        out = dict(samples[0])
        out["repeats"] = 1
        out["deterministic_across_repeats"] = True
        return out
    ref_tokens = samples[0]["tokens"]
    det = all(s["tokens"] == ref_tokens for s in samples[1:])
    prefill_s = float(statistics.median(s["prefill_seconds"] for s in samples))
    decode_s = float(statistics.median(s["decode_seconds"] for s in samples))
    prefill_tok_s = float(statistics.median(s["prefill_tok_s"] for s in samples))
    decode_tok_s = float(statistics.median(s["decode_tok_s"] for s in samples))
    med = dict(samples[0])
    med.update({
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "repeats": len(samples),
        "deterministic_across_repeats": det,
    })
    return med


def _decode_arm_texts(tok: NanoTokenizer, arms: list[dict[str, Any]]) -> None:
    if not arms:
        return
    decoded = tok.decode_many([a["tokens"] for a in arms])
    for arm, dec in zip(arms, decoded):
        plain = dec["text"]
        special = dec["special"]
        visible = extract_visible_answer(plain, special)
        coherent, coh_reason = check_coherence(arm["tokens"], visible if visible else plain)
        arm["plain_text"] = plain
        arm["special_text"] = special
        arm["visible_text"] = visible
        arm["coherent"] = coherent
        arm["coherence_reason"] = coh_reason


def _run_core_for_config(
    *,
    binary: Path,
    checkpoint: Path,
    draft: Optional[Path],
    knobs_path: Optional[Path],
    config_name: str,
    config_flags: list[str],
    knob_overrides: list[str],
    extra_flags: list[str],
    context: int,
    mode: str,
    repeats: int,
    timeout: int,
    tok: NanoTokenizer,
    warmup_ids: list[int],
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arms_to_run = ["base", "draft"] if mode == "both" else [mode]
    requests: list[dict[str, Any]] = [{"id": 0, "tokens": warmup_ids, "max_tokens": 32, "reset": True}]
    for i, task in enumerate(tasks, start=1):
        requests.append({
            "id": i,
            "tokens": task["prompt_ids"],
            "max_tokens": int(task["max_tokens"]),
            "reset": True,
        })

    per_arm_repeats: dict[str, list[tuple[list[dict[str, Any]], str]]] = {arm: [] for arm in arms_to_run}
    for rep in range(repeats):
        order = arms_to_run[rep % len(arms_to_run) :] + arms_to_run[: rep % len(arms_to_run)]
        for arm in order:
            cmd = build_session_cmd(
                binary,
                checkpoint,
                context=context,
                knobs_path=knobs_path,
                config_flags=config_flags,
                knob_overrides=knob_overrides,
                extra_flags=extra_flags,
                draft=draft if arm == "draft" else None,
            )
            print(f"  [{config_name}][core][rep {rep + 1}/{repeats}] running {arm}...", file=sys.stderr, flush=True)
            resps, stderr = run_session_batch(cmd, requests, timeout=timeout)
            per_arm_repeats[arm].append((resps[1:], stderr))

    # Build per-task arm records and batch-decode all unique texts cleanly.
    task_arm_samples: dict[tuple[int, str], list[dict[str, Any]]] = {}
    all_decode_targets: list[dict[str, Any]] = []
    for arm in arms_to_run:
        for rep_idx, (resps, stderr) in enumerate(per_arm_repeats[arm]):
            leak_ok, leak_reasons = check_stderr_leaks(stderr)
            for t_idx, (task, resp) in enumerate(zip(tasks, resps)):
                tokens = [int(x) for x in resp["tokens"]]
                metrics = _summarize_arm_metrics(
                    prompt_tokens=len(task["prompt_ids"]),
                    gen_tokens=len(tokens),
                    prefill_seconds=float(resp["prefill_seconds"]),
                    decode_seconds=float(resp["decode_seconds"]),
                    spec_batches=int(resp.get("speculative_batches", 0)),
                    spec_proposed=int(resp.get("speculative_proposed", 0)),
                    spec_accepted=int(resp.get("speculative_accepted", 0)),
                    dispatches=int(resp.get("dispatches", 0)),
                    eos=bool(resp.get("eos", False)),
                    kv_allocated_bytes=int(resp.get("kv_allocated_bytes", 0)),
                    kv_valid_bytes=int(resp.get("kv_valid_bytes", 0)),
                )
                sample = {
                    **metrics,
                    "tokens": tokens,
                    "leak_free": leak_ok,
                    "leak_reasons": leak_reasons,
                }
                task_arm_samples.setdefault((t_idx, arm), []).append(sample)
                if rep_idx == 0:
                    all_decode_targets.append(sample)

    _decode_arm_texts(tok, all_decode_targets)

    results: list[dict[str, Any]] = []
    for t_idx, task in enumerate(tasks):
        arm_summary: dict[str, Any] = {}
        for arm in arms_to_run:
            samples = task_arm_samples[(t_idx, arm)]
            rep0 = samples[0]
            agg = _aggregate_arm_repeats(samples)
            agg["plain_text"] = rep0["plain_text"]
            agg["special_text"] = rep0["special_text"]
            agg["visible_text"] = rep0["visible_text"]
            agg["coherent"] = rep0["coherent"]
            agg["coherence_reason"] = rep0["coherence_reason"]
            agg["leak_free"] = all(s["leak_free"] for s in samples)
            agg["leak_reasons"] = sorted({r for s in samples for r in s["leak_reasons"]})
            arm_summary[arm] = agg

        primary_name = "draft" if "draft" in arm_summary else "base"
        primary = arm_summary[primary_name]

        if "base" in arm_summary and "draft" in arm_summary:
            parity_ok, first_diff = compare_token_parity(arm_summary["base"]["tokens"], arm_summary["draft"]["tokens"])
            speedup = (
                arm_summary["draft"]["decode_tok_s"] / arm_summary["base"]["decode_tok_s"]
                if arm_summary["base"]["decode_tok_s"] > 0
                else 0.0
            )
        else:
            parity_ok, first_diff = None, None
            speedup = None

        grader = CORE_GRADERS[task["id"]]
        grade = grader(primary["visible_text"])
        quality_reason = grade.reason
        if not grade.passed and not primary["visible_text"] and "<|content_thinking|>" in primary["special_text"]:
            quality_reason = f"truncated_in_thinking:{grade.reason}"
        leak_free = all(a["leak_free"] for a in arm_summary.values())
        deterministic = all(a["deterministic_across_repeats"] for a in arm_summary.values())
        coherent = bool(primary["coherent"])
        overall_pass = (
            grade.passed
            and coherent
            and leak_free
            and deterministic
            and (parity_ok is None or parity_ok)
        )

        # Keep report JSON compact & readable while preserving tokens & preview.
        compact_arms: dict[str, Any] = {}
        for aname, ainfo in arm_summary.items():
            compact_arms[aname] = {
                k: v
                for k, v in ainfo.items()
                if k not in ("tokens", "plain_text", "special_text", "visible_text")
            }
            compact_arms[aname]["output_preview"] = ainfo["visible_text"][:280]

        entry: dict[str, Any] = {
            "id": task["id"],
            "suite": "core",
            "category": task["category"],
            "prompt_tokens": primary["prompt_tokens"],
            "generated_tokens": primary["generated_tokens"],
            "passed": overall_pass,
            "quality_passed": grade.passed,
            "quality_reason": quality_reason,
            "quality_detail": grade.detail,
            "coherent": coherent,
            "coherence_reason": primary["coherence_reason"],
            "exact_parity": parity_ok,
            "first_diff": first_diff,
            "leak_free": leak_free,
            "deterministic_across_repeats": deterministic,
            "prefill_seconds": primary["prefill_seconds"],
            "decode_seconds": primary["decode_seconds"],
            "prefill_tok_s": primary["prefill_tok_s"],
            "decode_tok_s": primary["decode_tok_s"],
            "mean_step": primary["mean_step"],
            "acceptance_rate": primary["acceptance_rate"],
            "speedup_vs_base": speedup,
            "base_prefill_tok_s": arm_summary["base"]["prefill_tok_s"] if "base" in arm_summary else None,
            "base_decode_tok_s": arm_summary["base"]["decode_tok_s"] if "base" in arm_summary else None,
            "arms": compact_arms,
        }
        results.append(entry)
    return results


def _run_long_or_scale_for_config(
    *,
    suite_name: str,
    specs: list[dict[str, Any]],
    binary: Path,
    checkpoint: Path,
    draft: Optional[Path],
    knobs_path: Optional[Path],
    config_name: str,
    config_flags: list[str],
    knob_overrides: list[str],
    extra_flags: list[str],
    auto_lsa: bool,
    thinking: bool,
    mode: str,
    repeats: int,
    timeout: int,
    tok: NanoTokenizer,
    warmup_ids: list[int],
) -> list[dict[str, Any]]:
    eff_knobs = list(knob_overrides)
    if auto_lsa:
        if not any(k.startswith("lsa=") for k in eff_knobs):
            eff_knobs.append("lsa=true")
        if not any(k.startswith("lsa_cross_loop_reuse=") for k in eff_knobs):
            eff_knobs.append("lsa_cross_loop_reuse=true")

    arms_to_run = ["base", "draft"] if mode == "both" else [mode]
    turn2_ids_by_closed = tok.next_turn_ids_pair(TURN2_TEXT, thinking=thinking)
    results: list[dict[str, Any]] = []

    for spec in specs:
        ctx = int(spec["context"])
        turn1_ids = build_archive_prompt(
            tok,
            target_tokens=int(spec["haystack_tokens"]),
            passkey=spec["passkey"],
            max_allowed_tokens=ctx - 1280,
            thinking=thinking,
        )
        has_turn2 = int(spec["turn2_max"]) > 0
        per_arm_samples: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms_to_run}

        for rep in range(repeats):
            order = arms_to_run[rep % len(arms_to_run) :] + arms_to_run[: rep % len(arms_to_run)]
            for arm in order:
                cmd = build_session_cmd(
                    binary,
                    checkpoint,
                    context=ctx,
                    knobs_path=knobs_path,
                    config_flags=config_flags,
                    knob_overrides=eff_knobs,
                    extra_flags=extra_flags,
                    draft=draft if arm == "draft" else None,
                )
                print(
                    f"  [{config_name}][{suite_name}][{spec['id']}][rep {rep + 1}/{repeats}] running {arm}...",
                    file=sys.stderr,
                    flush=True,
                )
                t1_resps, t2_resp, t2_prompt_len, stderr = run_session_long_task(
                    cmd,
                    turn1_ids,
                    turn1_max=int(spec["turn1_max"]),
                    turn2_ids_by_closed=turn2_ids_by_closed if has_turn2 else None,
                    turn2_max=int(spec["turn2_max"]),
                    warmup_ids=warmup_ids,
                    timeout=timeout,
                )
                leak_ok, leak_reasons = check_stderr_leaks(stderr)
                t1_final = t1_resps[-1]
                t1_prefill_sec = sum(float(r["prefill_seconds"]) for r in t1_resps)
                t1_tokens = [int(x) for x in t1_final["tokens"]]
                t1_decode_sec = float(t1_final["decode_seconds"])
                t2_prefill_sec = float(t2_resp["prefill_seconds"]) if t2_resp is not None else 0.0
                t2_decode_sec = float(t2_resp["decode_seconds"]) if t2_resp is not None else 0.0
                t2_tokens = [int(x) for x in t2_resp["tokens"]] if t2_resp is not None else []
                last_r = t2_resp if t2_resp is not None else t1_final

                total_prompt = len(turn1_ids) + t2_prompt_len
                total_gen = len(t1_tokens) + len(t2_tokens)
                total_prefill_sec = t1_prefill_sec + t2_prefill_sec
                total_decode_sec = t1_decode_sec + t2_decode_sec
                spec_batches = int(t1_final.get("speculative_batches", 0)) + (
                    int(t2_resp.get("speculative_batches", 0)) if t2_resp is not None else 0
                )
                spec_proposed = int(t1_final.get("speculative_proposed", 0)) + (
                    int(t2_resp.get("speculative_proposed", 0)) if t2_resp is not None else 0
                )
                spec_accepted = int(t1_final.get("speculative_accepted", 0)) + (
                    int(t2_resp.get("speculative_accepted", 0)) if t2_resp is not None else 0
                )
                total_dispatches = sum(int(r.get("dispatches", 0)) for r in t1_resps) + (
                    int(t2_resp.get("dispatches", 0)) if t2_resp is not None else 0
                )
                metrics = _summarize_arm_metrics(
                    prompt_tokens=total_prompt,
                    gen_tokens=total_gen,
                    prefill_seconds=total_prefill_sec,
                    decode_seconds=total_decode_sec,
                    spec_batches=spec_batches,
                    spec_proposed=spec_proposed,
                    spec_accepted=spec_accepted,
                    dispatches=total_dispatches,
                    eos=bool(last_r.get("eos", False)),
                    kv_allocated_bytes=int(last_r.get("kv_allocated_bytes", 0)),
                    kv_valid_bytes=int(last_r.get("kv_valid_bytes", 0)),
                )
                per_arm_samples[arm].append({
                    **metrics,
                    "tokens": t1_tokens + t2_tokens,
                    "t1_tokens": t1_tokens,
                    "t2_tokens": t2_tokens,
                    "t1_prompt_tokens": len(turn1_ids),
                    "t1_prefill_tok_s": len(turn1_ids) / t1_prefill_sec if t1_prefill_sec > 0 else 0.0,
                    "t1_decode_tok_s": len(t1_tokens) / t1_decode_sec if t1_decode_sec > 0 else 0.0,
                    "t2_decode_tok_s": (len(t2_tokens) / t2_decode_sec if t2_decode_sec > 0 else 0.0) if t2_resp else None,
                    "leak_free": leak_ok,
                    "leak_reasons": leak_reasons,
                })

        arm_summary: dict[str, Any] = {}
        decode_batch: list[list[int]] = []
        decode_owners: list[tuple[str, str]] = []
        for arm in arms_to_run:
            rep0 = per_arm_samples[arm][0]
            decode_batch.append(rep0["t1_tokens"])
            decode_owners.append((arm, "t1"))
            if has_turn2:
                decode_batch.append(rep0["t2_tokens"])
                decode_owners.append((arm, "t2"))
        decoded = tok.decode_many(decode_batch)
        decoded_map: dict[tuple[str, str], str] = {}
        for owner, dec in zip(decode_owners, decoded):
            decoded_map[owner] = extract_visible_answer(dec["text"], dec["special"])

        for arm in arms_to_run:
            samples = per_arm_samples[arm]
            agg = _aggregate_arm_repeats(samples)
            rep0 = samples[0]
            t1_text = decoded_map[(arm, "t1")]
            t1_coh, t1_coh_reason = check_coherence(rep0["t1_tokens"], t1_text)
            if has_turn2:
                t2_text = decoded_map[(arm, "t2")]
                t2_coh, t2_coh_reason = check_coherence(rep0["t2_tokens"], t2_text)
            else:
                t2_text = None
                t2_coh, t2_coh_reason = True, "unused"
            agg["t1_text"] = t1_text
            agg["t2_text"] = t2_text
            agg["t1_coherent"] = t1_coh
            agg["t2_coherent"] = t2_coh
            agg["coherent"] = t1_coh and t2_coh
            agg["coherence_reason"] = t1_coh_reason if not t1_coh else (t2_coh_reason if not t2_coh else "clean")
            agg["leak_free"] = all(s["leak_free"] for s in samples)
            agg["leak_reasons"] = sorted({r for s in samples for r in s["leak_reasons"]})
            agg["t1_decode_tok_s"] = float(statistics.median(s["t1_decode_tok_s"] for s in samples))
            if has_turn2:
                agg["t2_decode_tok_s"] = float(statistics.median(float(s["t2_decode_tok_s"]) for s in samples))
            arm_summary[arm] = agg

        primary_name = "draft" if "draft" in arm_summary else "base"
        primary = arm_summary[primary_name]

        if "base" in arm_summary and "draft" in arm_summary:
            eq1, diff1 = compare_token_parity(per_arm_samples["base"][0]["t1_tokens"], per_arm_samples["draft"][0]["t1_tokens"])
            eq2, diff2 = (
                compare_token_parity(per_arm_samples["base"][0]["t2_tokens"], per_arm_samples["draft"][0]["t2_tokens"])
                if has_turn2
                else (True, None)
            )
            parity_ok = eq1 and eq2
            first_diff = diff1 if not eq1 else diff2
            speedup = (
                arm_summary["draft"]["decode_tok_s"] / arm_summary["base"]["decode_tok_s"]
                if arm_summary["base"]["decode_tok_s"] > 0
                else 0.0
            )
        else:
            parity_ok, first_diff = None, None
            speedup = None

        grade = grade_long_multiturn(
            spec["passkey"],
            primary["t1_text"],
            primary["t2_text"],
            primary["t1_coherent"],
            primary["t2_coherent"],
        )
        if spec["passkey"] is None and primary["kv_allocated_bytes"] <= 0:
            grade = GradeResult(False, "kv_allocated_bytes_zero", {})

        leak_free = all(a["leak_free"] for a in arm_summary.values())
        deterministic = all(a["deterministic_across_repeats"] for a in arm_summary.values())
        coherent = bool(primary["coherent"])
        overall_pass = (
            grade.passed
            and coherent
            and leak_free
            and deterministic
            and (parity_ok is None or parity_ok)
        )

        compact_arms: dict[str, Any] = {}
        for aname, ainfo in arm_summary.items():
            compact_arms[aname] = {
                k: v
                for k, v in ainfo.items()
                if k not in ("tokens", "t1_tokens", "t2_tokens", "t1_text", "t2_text")
            }
            compact_arms[aname]["t1_preview"] = (ainfo["t1_text"] or "")[:200]
            if ainfo["t2_text"] is not None:
                compact_arms[aname]["t2_preview"] = ainfo["t2_text"][:200]

        results.append({
            "id": spec["id"],
            "suite": suite_name,
            "category": spec["category"],
            "context": ctx,
            "prompt_tokens": primary["prompt_tokens"],
            "generated_tokens": primary["generated_tokens"],
            "passed": overall_pass,
            "quality_passed": grade.passed,
            "quality_reason": grade.reason,
            "quality_detail": grade.detail,
            "coherent": coherent,
            "coherence_reason": primary["coherence_reason"],
            "exact_parity": parity_ok,
            "first_diff": first_diff,
            "leak_free": leak_free,
            "deterministic_across_repeats": deterministic,
            "prefill_seconds": primary["prefill_seconds"],
            "decode_seconds": primary["decode_seconds"],
            "prefill_tok_s": primary["prefill_tok_s"],
            "decode_tok_s": primary["decode_tok_s"],
            "mean_step": primary["mean_step"],
            "acceptance_rate": primary["acceptance_rate"],
            "speedup_vs_base": speedup,
            "base_prefill_tok_s": arm_summary["base"]["prefill_tok_s"] if "base" in arm_summary else None,
            "base_decode_tok_s": arm_summary["base"]["decode_tok_s"] if "base" in arm_summary else None,
            "arms": compact_arms,
        })
    return results


def _run_bench_one(
    binary: Path,
    checkpoint: Path,
    *,
    spec: dict[str, Any],
    knobs_path: Optional[Path],
    config_flags: list[str],
    knob_overrides: list[str],
    extra_flags: list[str],
    draft: Optional[Path],
    greedy: bool,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        str(binary),
        "bench",
        str(checkpoint),
        "--context",
        str(spec["context"]),
        "--prompt-tokens",
        str(spec["prompt_tokens"]),
        "--decode-tokens",
        str(spec["decode_tokens"]),
        "--repeats",
        str(spec["repeats"]),
        "--decode-detail",
        "1",
    ]
    if greedy:
        cmd.extend(["--decode-mode", "greedy"])
    if knobs_path is not None:
        cmd.extend(["--knobs", str(knobs_path)])
    cmd.extend(config_flags)
    for kv in knob_overrides:
        cmd.extend(["--knob", kv])
    cmd.extend(extra_flags)
    if draft is not None:
        cmd.extend(["--draft", str(draft)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nano bench failed ({proc.returncode}): {(proc.stderr or '')[-2000:]}")
    bench_obj = None
    for ln in reversed(proc.stdout.splitlines()):
        if ln.strip():
            bench_obj = json.loads(ln)
            break
    if not isinstance(bench_obj, dict) or bench_obj.get("type") != "nano_bench":
        raise RuntimeError(f"unexpected nano bench stdout: {proc.stdout[-500:]}")
    detail_obj = None
    for ln in proc.stderr.splitlines():
        if '"nano_decode_detail"' in ln:
            detail_obj = json.loads(ln)
    leak_ok, leak_reasons = check_stderr_leaks(proc.stderr)
    return {
        "bench": bench_obj,
        "decode_detail": detail_obj,
        "leak_free": leak_ok,
        "leak_reasons": leak_reasons,
    }


def _run_bench_for_config(
    *,
    specs: list[dict[str, Any]],
    binary: Path,
    checkpoint: Path,
    draft: Optional[Path],
    knobs_path: Optional[Path],
    config_name: str,
    config_flags: list[str],
    knob_overrides: list[str],
    extra_flags: list[str],
    mode: str,
    timeout: int,
) -> list[dict[str, Any]]:
    arms_to_run = ["base", "draft"] if mode == "both" else [mode]
    greedy = (mode in ("both", "draft")) or (draft is not None)
    results: list[dict[str, Any]] = []
    for spec in specs:
        arm_runs: dict[str, dict[str, Any]] = {}
        for arm in arms_to_run:
            print(f"  [{config_name}][bench][{spec['id']}] running {arm}...", file=sys.stderr, flush=True)
            arm_runs[arm] = _run_bench_one(
                binary,
                checkpoint,
                spec=spec,
                knobs_path=knobs_path,
                config_flags=config_flags,
                knob_overrides=knob_overrides,
                extra_flags=extra_flags,
                draft=draft if arm == "draft" else None,
                greedy=greedy,
                timeout=timeout,
            )
        primary_name = "draft" if "draft" in arm_runs else "base"
        p_bench = arm_runs[primary_name]["bench"]
        p_detail = arm_runs[primary_name]["decode_detail"]
        gate_redos = int(p_detail["gate_redos"]) if isinstance(p_detail, dict) and "gate_redos" in p_detail else 0
        if "base" in arm_runs and "draft" in arm_runs:
            b_bench = arm_runs["base"]["bench"]
            parity_ok, first_diff = compare_token_parity(b_bench["predictions"], p_bench["predictions"])
            base_dec = float(b_bench["decode_tokens_per_second"])
            speedup = float(p_bench["decode_tokens_per_second"]) / base_dec if base_dec > 0 else 0.0
        else:
            parity_ok, first_diff = None, None
            speedup = None
        leak_free = all(r["leak_free"] for r in arm_runs.values())
        overall_pass = leak_free and gate_redos == 0 and (parity_ok is None or parity_ok)
        prefill_med = float(statistics.median(p_bench["prefill_seconds"]))
        decode_med = float(statistics.median(p_bench["decode_seconds"]))
        results.append({
            "id": spec["id"],
            "suite": "bench",
            "category": spec["category"],
            "context": spec["context"],
            "prompt_tokens": int(p_bench["prompt_tokens"]),
            "generated_tokens": int(p_bench["decode_tokens"]),
            "passed": overall_pass,
            "quality_passed": True,
            "quality_reason": "deterministic_bench_checksum",
            "quality_detail": {"gate_redos": gate_redos, "decode_mode": p_bench.get("decode_mode")},
            "coherent": True,
            "coherence_reason": "synthetic_bench",
            "exact_parity": parity_ok,
            "first_diff": first_diff,
            "leak_free": leak_free,
            "deterministic_across_repeats": True,
            "prefill_seconds": prefill_med,
            "decode_seconds": decode_med,
            "prefill_tok_s": float(p_bench["prefill_tokens_per_second"]),
            "decode_tok_s": float(p_bench["decode_tokens_per_second"]),
            "mean_step": float(p_bench.get("mean_accepted_per_step") or 1.0),
            "acceptance_rate": (
                float(p_bench["speculative_accepted"]) / float(p_bench["speculative_proposed"])
                if float(p_bench.get("speculative_proposed", 0)) > 0
                else 0.0
            ),
            "speedup_vs_base": speedup,
            "base_prefill_tok_s": float(arm_runs["base"]["bench"]["prefill_tokens_per_second"]) if "base" in arm_runs else None,
            "base_decode_tok_s": float(arm_runs["base"]["bench"]["decode_tokens_per_second"]) if "base" in arm_runs else None,
        })
    return results


def rollup_config(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    non_bench = [t for t in tasks if t["suite"] != "bench"]
    pool = non_bench if non_bench else tasks
    total = len(pool)
    passed = sum(1 for t in pool if t["passed"])
    quality_passed = sum(1 for t in pool if t["quality_passed"])
    coherent_tasks = sum(1 for t in pool if t["coherent"])
    parity_tasks = [t for t in tasks if t["exact_parity"] is not None]
    parity_passed = sum(1 for t in parity_tasks if t["exact_parity"] is True)
    leaks = sum(1 for t in tasks if not t["leak_free"])

    p_toks = sum(t["prompt_tokens"] for t in tasks)
    p_secs = sum(t["prefill_seconds"] for t in tasks)
    g_toks = sum(t["generated_tokens"] for t in tasks)
    d_secs = sum(t["decode_seconds"] for t in tasks)

    coh_pool = [t for t in tasks if t["coherent"]]
    cg_toks = sum(t["generated_tokens"] for t in coh_pool)
    cd_secs = sum(t["decode_seconds"] for t in coh_pool)

    short_pool = [t for t in non_bench if t["prompt_tokens"] < 1024]
    long_pool = [t for t in non_bench if t["prompt_tokens"] >= 1024]
    sg_toks = sum(t["generated_tokens"] for t in short_pool)
    sd_secs = sum(t["decode_seconds"] for t in short_pool)
    lg_toks = sum(t["generated_tokens"] for t in long_pool)
    ld_secs = sum(t["decode_seconds"] for t in long_pool)

    draft_tasks = [t for t in tasks if t.get("arms", {}).get("draft", {}).get("speculative_batches", 0) > 0]
    total_spec_batches = sum(t["arms"]["draft"]["speculative_batches"] for t in draft_tasks)
    total_spec_gen = sum(t["arms"]["draft"]["generated_tokens"] for t in draft_tasks)
    total_spec_prop = sum(t["arms"]["draft"]["speculative_proposed"] for t in draft_tasks)
    total_spec_acc = sum(t["arms"]["draft"]["speculative_accepted"] for t in draft_tasks)
    if total_spec_batches > 0:
        mean_step = total_spec_gen / total_spec_batches
        acceptance_rate = (total_spec_acc / total_spec_prop) if total_spec_prop > 0 else 0.0
    else:
        steps = [t["mean_step"] for t in tasks if t["mean_step"] > 0]
        mean_step = float(statistics.mean(steps)) if steps else 0.0
        acceptance_rate = 0.0

    # Base weighted decode tok/s & speedup if paired
    paired = [t for t in tasks if t.get("arms", {}).get("base") and t.get("arms", {}).get("draft")]
    if paired:
        bg_toks = sum(t["arms"]["base"]["generated_tokens"] for t in paired)
        bd_secs = sum(t["arms"]["base"]["decode_seconds"] for t in paired)
        base_dec_tok_s = (bg_toks / bd_secs) if bd_secs > 0 else 0.0
        draft_dec_tok_s = (sum(t["arms"]["draft"]["generated_tokens"] for t in paired) / sum(t["arms"]["draft"]["decode_seconds"] for t in paired)) if paired else 0.0
        speedup_vs_base = (draft_dec_tok_s / base_dec_tok_s) if base_dec_tok_s > 0 else 0.0
    else:
        base_dec_tok_s = None
        speedup_vs_base = None

    return {
        "tasks_passed": passed,
        "tasks_total": total,
        "quality_passed": quality_passed,
        "quality_pct": (100.0 * passed / total) if total > 0 else 0.0,
        "coherent_tasks": coherent_tasks,
        "parity_passed": parity_passed,
        "parity_total": len(parity_tasks),
        "leaks": leaks,
        "weighted_prefill_tok_s": (p_toks / p_secs) if p_secs > 0 else 0.0,
        "weighted_decode_tok_s": (g_toks / d_secs) if d_secs > 0 else 0.0,
        "coherent_decode_tok_s": (cg_toks / cd_secs) if cd_secs > 0 else 0.0,
        "short_decode_tok_s": (sg_toks / sd_secs) if sd_secs > 0 else None,
        "long_decode_tok_s": (lg_toks / ld_secs) if ld_secs > 0 else None,
        "mean_step": mean_step,
        "acceptance_rate": acceptance_rate,
        "base_weighted_decode_tok_s": base_dec_tok_s,
        "speedup_vs_base": speedup_vs_base,
    }


# ---- Markdown Formatting & Comparison ----


def _fmt_num(val: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def format_report_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    lines: list[str] = [
        f"# nanoattention Eval & Perf Report (`{meta.get('nano_version', 'unknown')}`)",
        "",
        f"- **Checkpoint:** `{meta.get('checkpoint_name', 'unknown')}` (`model_type={meta.get('model_type', 'unknown')}`)",
        f"- **Draft:** `{meta.get('draft_name') or 'none'}` · **Mode:** `{meta.get('mode')}` · **Knobs:** `{meta.get('knobs_name') or 'default'}`",
        f"- **Suites:** `{', '.join(meta.get('suites', []))}` · **Repeats:** `{meta.get('repeats', 1)}` · **Timestamp:** `{meta.get('timestamp_utc', '')}`",
        "",
        "## Config Summary",
        "",
        "| Config | Pass | Quality | Coherent | Parity | Leaks | Prefill (`tok/s`) | Coherent Decode (`tok/s`) | All Decode (`tok/s`) | Short (`tok/s`) | 4K+ (`tok/s`) | `mean_step` | Speedup |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for cfg in report["configs"]:
        s = cfg["summary"]
        parity_str = f"{s['parity_passed']}/{s['parity_total']}" if s["parity_total"] > 0 else "—"
        lines.append(
            f"| **`{cfg['name']}`** "
            f"| {s['tasks_passed']}/{s['tasks_total']} "
            f"| {s['quality_pct']:.1f}% "
            f"| {s['coherent_tasks']}/{s['tasks_total']} "
            f"| {parity_str} "
            f"| {s['leaks']} "
            f"| {_fmt_num(s['weighted_prefill_tok_s'], 1)} "
            f"| **{_fmt_num(s['coherent_decode_tok_s'], 2)}** "
            f"| {_fmt_num(s['weighted_decode_tok_s'], 2)} "
            f"| {_fmt_num(s['short_decode_tok_s'], 2)} "
            f"| {_fmt_num(s['long_decode_tok_s'], 2)} "
            f"| {_fmt_num(s['mean_step'], 2)} "
            f"| {_fmt_num(s['speedup_vs_base'], 2, '×')} |"
        )

    lines.extend(["", "## Per-Task Breakdown", ""])
    lines.extend([
        "| Config | Task | In / Out | Pass | Coherence | Parity | Base (`tok/s`) | Decode (`tok/s`) | Speedup | `mean_step` | Reason |",
        "| :--- | :--- | ---: | :---: | :--- | :---: | ---: | ---: | :---: | :---: | :--- |",
    ])
    for cfg in report["configs"]:
        for t in cfg["tasks"]:
            parity_str = "PASS" if t["exact_parity"] is True else ("FAIL" if t["exact_parity"] is False else "—")
            status_str = "PASS" if t["passed"] else "**FAIL**"
            lines.append(
                f"| `{cfg['name']}` | `{t['id']}` | {t['prompt_tokens']} / {t['generated_tokens']} "
                f"| {status_str} | {t['coherence_reason']} | {parity_str} "
                f"| {_fmt_num(t.get('base_decode_tok_s'), 2)} "
                f"| {_fmt_num(t['decode_tok_s'], 2)} "
                f"| {_fmt_num(t.get('speedup_vs_base'), 2, '×')} "
                f"| {_fmt_num(t.get('mean_step'), 2)} "
                f"| {t['quality_reason']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _pct_delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old <= 0:
        return None
    return 100.0 * (new - old) / old


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_perf_drop_pct: float = 5.0,
) -> tuple[bool, list[str], str]:
    base_cfgs = {c["name"]: c for c in baseline.get("configs", [])}
    cand_cfgs = {c["name"]: c for c in candidate.get("configs", [])}
    common = [name for name in base_cfgs if name in cand_cfgs]
    failures: list[str] = []

    if not common:
        failures.append("no_matching_configs_between_baseline_and_candidate")

    lines = [
        "# Eval Comparison (`baseline` -> `candidate`)",
        "",
        "| Config | Base Pass | Cand Pass | Coherent | Parity | Leaks | Prefill Delta | Coherent Decode (`tok/s`) | Overall Decode (`tok/s`) | `mean_step` | Gate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name in common:
        bs = base_cfgs[name]["summary"]
        cs = cand_cfgs[name]["summary"]
        cfg_fails: list[str] = []
        if cs["tasks_passed"] < bs["tasks_passed"]:
            cfg_fails.append(f"quality_drop:{bs['tasks_passed']}->{cs['tasks_passed']}")
        if cs["coherent_tasks"] < bs["coherent_tasks"]:
            cfg_fails.append(f"coherence_drop:{bs['coherent_tasks']}->{cs['coherent_tasks']}")
        if cs["parity_total"] > 0 and cs["parity_passed"] < cs["parity_total"]:
            cfg_fails.append(f"parity_fail:{cs['parity_passed']}/{cs['parity_total']}")
        if cs["leaks"] > 0:
            cfg_fails.append(f"leaks:{cs['leaks']}")
        coh_drop = _pct_delta(cs["coherent_decode_tok_s"], bs["coherent_decode_tok_s"])
        pref_drop = _pct_delta(cs["weighted_prefill_tok_s"], bs["weighted_prefill_tok_s"])
        if coh_drop is not None and coh_drop < -max_perf_drop_pct:
            cfg_fails.append(f"coherent_decode_drop:{coh_drop:+.1f}%")
        if pref_drop is not None and pref_drop < -max_perf_drop_pct:
            cfg_fails.append(f"prefill_drop:{pref_drop:+.1f}%")
        failures.extend(f"{name}:{f}" for f in cfg_fails)

        dec_delta = _pct_delta(cs["weighted_decode_tok_s"], bs["weighted_decode_tok_s"])
        step_delta = cs["mean_step"] - bs["mean_step"]
        parity_str = f"{cs['parity_passed']}/{cs['parity_total']}" if cs["parity_total"] > 0 else "—"
        gate_str = "PASS" if not cfg_fails else "**FAIL**"
        lines.append(
            f"| `{name}` | {bs['tasks_passed']}/{bs['tasks_total']} | {cs['tasks_passed']}/{cs['tasks_total']} "
            f"| {cs['coherent_tasks']}/{cs['tasks_total']} | {parity_str} | {cs['leaks']} "
            f"| {_fmt_num(pref_drop, 1, '%')} "
            f"| {bs['coherent_decode_tok_s']:.2f} -> **{cs['coherent_decode_tok_s']:.2f}** ({_fmt_num(coh_drop, 1, '%')}) "
            f"| {bs['weighted_decode_tok_s']:.2f} -> {cs['weighted_decode_tok_s']:.2f} ({_fmt_num(dec_delta, 1, '%')}) "
            f"| {bs['mean_step']:.2f} -> {cs['mean_step']:.2f} ({step_delta:+.2f}) "
            f"| {gate_str} |"
        )

    # Check matched task regressions
    task_regressions: list[str] = []
    for name in common:
        btasks = {(t["suite"], t["id"]): t for t in base_cfgs[name]["tasks"]}
        for ct in cand_cfgs[name]["tasks"]:
            key = (ct["suite"], ct["id"])
            bt = btasks.get(key)
            if bt is None:
                continue
            if bt["passed"] and not ct["passed"]:
                task_regressions.append(f"`{name}/{ct['suite']}/{ct['id']}`: PASS -> FAIL ({ct['quality_reason']}, {ct['coherence_reason']})")
    if task_regressions:
        lines.extend(["", "## Regressed Tasks", ""] + [f"- {r}" for r in task_regressions])
        failures.extend(task_regressions)

    lines.append("")
    return len(failures) == 0, failures, "\n".join(lines)


# ---- CLI Commands ----


def cmd_list(_: argparse.Namespace) -> int:
    print("Suites:")
    for sname, specs in ALL_SUITES.items():
        print(f"  {sname}:")
        for s in specs:
            print(f"    - {s['id']:<24s} [{s['category']}]  {s['description']}")
    print("\nBuilt-in configs:")
    for cname, flags in BUILTIN_CONFIGS.items():
        print(f"  - {cname:<14s} {' '.join(flags) if flags else '(checkpoint/default flags)'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    binary = Path(args.binary)
    checkpoint = Path(args.checkpoint)
    draft = Path(args.draft) if args.draft else None
    if not binary.is_file():
        raise SystemExit(f"nano binary not found: {binary}")
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoint}")
    if draft is not None and not draft.is_dir():
        raise SystemExit(f"draft directory not found: {draft}")

    suites = parse_csv_list(args.suite) or ["core"]
    for s in suites:
        if s not in ALL_SUITES:
            raise SystemExit(f"unknown suite: {s} (choices: {', '.join(ALL_SUITES)})")

    only = set(parse_csv_list(args.only))
    model_type = detect_model_type(checkpoint)
    default_cfg = "3p_hidden" if model_type == "inkling2_loop" else "default"
    configs = parse_csv_list(args.config) or [default_cfg]
    for c in configs:
        if c not in BUILTIN_CONFIGS:
            raise SystemExit(f"unknown config: {c} (choices: {', '.join(BUILTIN_CONFIGS)})")

    mode = args.mode
    if mode is None:
        mode = "both" if draft is not None else "base"
    if mode in ("both", "draft") and draft is None:
        raise SystemExit(f"--mode {mode} requires --draft PATH")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    knobs_path = resolve_knobs_path(checkpoint, args.knobs)
    tok = NanoTokenizer(binary, checkpoint, timeout=args.timeout)
    nano_ver = read_nano_version(binary)

    need_session_tok = any(s in ("core", "long", "scale") for s in suites)
    warmup_ids: list[int] = []
    core_tasks: list[dict[str, Any]] = []
    if "core" in suites:
        selected_core = [t for t in CORE_TASK_SPECS if not only or t["id"] in only]
        if selected_core:
            warmup_ids, built_core = build_core_tasks(tok, thinking=args.think)
            core_tasks = [t for t in built_core if t["id"] in {x["id"] for x in selected_core}]
    elif need_session_tok:
        warmup_ids = tok.chat_many([WARMUP_PROMPT], thinking=args.think)[0]

    selected_long = [t for t in LONG_SPECS if "long" in suites and (not only or t["id"] in only)]
    selected_scale = [t for t in SCALE_SPECS if "scale" in suites and (not only or t["id"] in only)]
    selected_bench = [t for t in BENCH_SPECS if "bench" in suites and (not only or t["id"] in only)]
    if only and not (core_tasks or selected_long or selected_scale or selected_bench):
        raise SystemExit(f"--only matched no tasks in suites {suites}: {sorted(only)}")

    lock_path = Path(args.lock) if args.lock else None
    config_entries: list[dict[str, Any]] = []

    with inference_lock(lock_path):
        for cfg_name in configs:
            cfg_flags = BUILTIN_CONFIGS[cfg_name]
            print(f"\n=== Running config: {cfg_name} ({' '.join(cfg_flags) or 'default'}) ===", file=sys.stderr, flush=True)
            tasks_out: list[dict[str, Any]] = []
            if core_tasks:
                tasks_out.extend(_run_core_for_config(
                    binary=binary,
                    checkpoint=checkpoint,
                    draft=draft,
                    knobs_path=knobs_path,
                    config_name=cfg_name,
                    config_flags=cfg_flags,
                    knob_overrides=list(args.knob),
                    extra_flags=list(args.extra_flag),
                    context=args.context,
                    mode=mode,
                    repeats=args.repeats,
                    timeout=args.timeout,
                    tok=tok,
                    warmup_ids=warmup_ids,
                    tasks=core_tasks,
                ))
            if selected_long:
                tasks_out.extend(_run_long_or_scale_for_config(
                    suite_name="long",
                    specs=selected_long,
                    binary=binary,
                    checkpoint=checkpoint,
                    draft=draft,
                    knobs_path=knobs_path,
                    config_name=cfg_name,
                    config_flags=cfg_flags,
                    knob_overrides=list(args.knob),
                    extra_flags=list(args.extra_flag),
                    auto_lsa=not args.no_auto_lsa,
                    thinking=args.think,
                    mode=mode,
                    repeats=args.repeats,
                    timeout=args.timeout,
                    tok=tok,
                    warmup_ids=warmup_ids,
                ))
            if selected_scale:
                tasks_out.extend(_run_long_or_scale_for_config(
                    suite_name="scale",
                    specs=selected_scale,
                    binary=binary,
                    checkpoint=checkpoint,
                    draft=draft,
                    knobs_path=knobs_path,
                    config_name=cfg_name,
                    config_flags=cfg_flags,
                    knob_overrides=list(args.knob),
                    extra_flags=list(args.extra_flag),
                    auto_lsa=not args.no_auto_lsa,
                    thinking=args.think,
                    mode=mode,
                    repeats=args.repeats,
                    timeout=args.timeout,
                    tok=tok,
                    warmup_ids=warmup_ids,
                ))
            if selected_bench:
                tasks_out.extend(_run_bench_for_config(
                    specs=selected_bench,
                    binary=binary,
                    checkpoint=checkpoint,
                    draft=draft,
                    knobs_path=knobs_path,
                    config_name=cfg_name,
                    config_flags=cfg_flags,
                    knob_overrides=list(args.knob),
                    extra_flags=list(args.extra_flag),
                    mode=mode,
                    timeout=args.timeout,
                ))

            summary = rollup_config(tasks_out)
            config_entries.append({
                "name": cfg_name,
                "flags": cfg_flags,
                "summary": summary,
                "tasks": tasks_out,
            })

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "nano_version": nano_ver,
            "checkpoint_name": checkpoint.name,
            "model_type": model_type,
            "draft_name": draft.name if draft else None,
            "knobs_name": safe_path_label(knobs_path),
            "knob_overrides": list(args.knob),
            "extra_flags": list(args.extra_flag),
            "suites": suites,
            "mode": mode,
            "repeats": args.repeats,
            "context": args.context,
            "think": args.think,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "configs": config_entries,
    }

    md = format_report_markdown(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")

    print(md)
    if args.strict:
        for cfg in config_entries:
            s = cfg["summary"]
            if s["tasks_passed"] < s["tasks_total"] or s["leaks"] > 0 or (s["parity_total"] > 0 and s["parity_passed"] < s["parity_total"]):
                return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    ok, failures, md = compare_reports(
        baseline,
        candidate,
        max_perf_drop_pct=float(args.max_perf_drop_pct),
    )
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
    print(md)
    if args.gate and not ok:
        print(f"GATE FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    md = format_report_markdown(report)
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_list = sub.add_parser("list", help="List available suites, tasks, and configs")
    sp_list.set_defaults(func=cmd_list)

    sp_run = sub.add_parser("run", help="Run evaluation suites on a checkpoint")
    sp_run.add_argument("checkpoint", help="Path to target model checkpoint directory")
    sp_run.add_argument("--binary", default=str(DEFAULT_BINARY), help="Path to `nano` binary")
    sp_run.add_argument("--draft", default=None, help="Path to DFlash 2 draft directory")
    sp_run.add_argument("--suite", action="append", help="Suite(s): core, long, scale, bench (default: core)")
    sp_run.add_argument("--only", action="append", help="Task ID(s) to run (comma-separated or repeatable)")
    sp_run.add_argument("--config", action="append", help="Config profile(s) (comma-separated or repeatable)")
    sp_run.add_argument("--mode", choices=("both", "draft", "base"), default=None, help="Execution mode (default: both if --draft else base)")
    sp_run.add_argument("--knobs", default=None, help="Knobs preset JSON path (or 'none')")
    sp_run.add_argument("--knob", action="append", default=[], help="Extra --knob name=value (repeatable)")
    sp_run.add_argument("--extra-flag", action="append", default=[], help="Extra flag passed directly to `nano session`")
    sp_run.add_argument("--no-auto-lsa", action="store_true", help="Disable auto lsa=true,lsa_cross_loop_reuse=true on long/scale")
    sp_run.add_argument("--think", action="store_true", help="Enable thinking effort in chat template")
    sp_run.add_argument("--repeats", type=int, default=1, help="Repeats per session (alternates base/draft when >1)")
    sp_run.add_argument("--context", type=int, default=8192, help="Context size for `core` suite (default: 8192)")
    sp_run.add_argument("--lock", default=None, help="Optional fcntl lock file path")
    sp_run.add_argument("--timeout", type=int, default=900, help="Per-session timeout in seconds (default: 900)")
    sp_run.add_argument("--out", default=None, help="Write schema-v1 JSON report to PATH")
    sp_run.add_argument("--md", default=None, help="Write Markdown report to PATH")
    sp_run.add_argument("--strict", action="store_true", help="Exit 1 on any quality, coherence, parity, or leak failure")
    sp_run.set_defaults(func=cmd_run)

    sp_cmp = sub.add_parser("compare", help="Compare two saved eval JSON reports")
    sp_cmp.add_argument("baseline", help="Baseline JSON report path")
    sp_cmp.add_argument("candidate", help="Candidate JSON report path")
    sp_cmp.add_argument("--gate", action="store_true", help="Exit 1 on quality/coherence/parity/leak/perf regressions")
    sp_cmp.add_argument("--max-perf-drop-pct", type=float, default=5.0, help="Allowed throughput drop %% before --gate fails (default: 5.0)")
    sp_cmp.add_argument("--md", default=None, help="Write comparison Markdown to PATH")
    sp_cmp.set_defaults(func=cmd_compare)

    sp_rep = sub.add_parser("report", help="Render Markdown from a saved JSON report")
    sp_rep.add_argument("report", help="JSON report path")
    sp_rep.add_argument("--md", default=None, help="Write Markdown to PATH")
    sp_rep.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
