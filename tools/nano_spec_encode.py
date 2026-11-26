#!/usr/bin/env python3
"""2-Inkling Speculative Encoding suite for nanoattention (Zig + Metal + MLX).

Experiments (matching b10-tv/speculative-encoding patterns on RoPE-free Inkling 2 Mini):
  exp-crosspass  1-pass / 2-pass Inkling #1 -> full 3-pass Inkling #2 translation
                 (pass-copy, pass-extrap, closed-form Ridge, and MLX UnifiedKVTranslator)
  exp-concat     Parallel 2-Inkling document encoding + zero-RoPE prefix concatenation
  exp-kvzip      Zero-training query-aware KVzip compression (4x / 16x / 64x)
  all            Run all three experiments sequentially and save /tmp/nano_spec_encode/results.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np

DEFAULT_NANO = Path("./zig-out/bin/nano")
DEFAULT_CHECKPOINT = Path(
    "/Users/rachpradhan/nanoattention/.graff/worktrees/liquid-main-port"
    "/target/exo-hillclimb/models/thinkingmachines--Inkling-2-Mini"
)
DEFAULT_KNOBS = Path("src/mega/presets/inkling.json")
DEFAULT_WORK = Path("/tmp/nano_spec_encode")
REPO_ROOT = Path(__file__).resolve().parents[1]

SHARED_HEADER_TEXT = (
    "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
    "<|message_user|><|content_text|>"
)
QA_OPEN_TEXT = "<|end_message|><|message_model|><|content_text|>"

# Inkling-2-Mini structural constants (match src/mega/inkling2_loop.zig).
INKLING_QK_CLIP = 22.627416997969522
INKLING_RMS_EPS = 1e-6
INKLING_KV_HEADS = 8
INKLING_HEAD_DIM = 128
INKLING_GLOBAL_VISITS_3P = (5, 11, 17, 23, 29, 35, 41, 47)
INKLING_GLOBAL_VISITS_2P = (5, 11, 17, 23, 29, 35)
INKLING_GLOBAL_VISITS_1P = (5, 11, 17, 23)


def u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Exact truncation-compatible BF16 (u16) -> FP32 unpack matching Metal/Zig."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_u16(f32: np.ndarray) -> np.ndarray:
    """Exact truncation FP32 -> BF16 (u16) pack matching mk_store_pack_bf16."""
    return (np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_round_f32(f32: np.ndarray) -> np.ndarray:
    return u16_to_f32(f32_to_u16(f32))


def apply_inkling_invariants(name: str, arr: np.ndarray) -> np.ndarray:
    """Enforce Inkling-2-Mini key L2-clip, value RMSNorm, and BF16 rounding."""
    if arr.shape[0] == 0:
        return np.ascontiguousarray(arr, dtype=np.float32)
    x = np.asarray(arr, dtype=np.float32)
    if name.endswith(".values"):
        h = x.reshape(x.shape[0], INKLING_KV_HEADS, INKLING_HEAD_DIM)
        ms = np.mean(h * h, axis=-1, keepdims=True)
        normed = h / np.sqrt(ms + INKLING_RMS_EPS)
        return bf16_round_f32(normed.reshape(x.shape))
    if name.endswith(".keys"):
        h = x.reshape(x.shape[0], INKLING_KV_HEADS, INKLING_HEAD_DIM)
        norms = np.sqrt(np.sum(h * h, axis=-1, keepdims=True))
        scale = np.minimum(INKLING_QK_CLIP / (norms + 1e-6), 1.0)
        return bf16_round_f32((h * scale).reshape(x.shape))
    return bf16_round_f32(x)


def align8(v: int) -> int:
    return (v + 7) & ~7


@dataclass
class NkvEntry:
    name: str
    kind: int  # 2 = sequence, 3 = ring
    width: int
    horizon: int
    first_pos: int
    rows: int
    data: np.ndarray  # [rows, width] uint16 (bf16) or float32 (f32)

    def as_f32(self) -> np.ndarray:
        if self.data.dtype == np.uint16:
            return u16_to_f32(self.data)
        return np.asarray(self.data, dtype=np.float32)

    def set_f32(self, arr: np.ndarray, dtype: str, first_pos: int | None = None) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == self.width
        self.rows = int(arr.shape[0])
        if first_pos is not None:
            self.first_pos = int(first_pos)
        self.data = f32_to_u16(arr) if dtype == "bf16" else arr


class NkvCartridge:
    """Reader/writer/algebra container for NANOKV01 binary KV cartridges."""

    MAGIC = b"NANOKV01"
    HDR_FMT = "<8s7I28s"
    DIR_FMT = "<6I2Q"

    def __init__(
        self,
        *,
        dtype: str,
        position: int,
        model_type: str,
        meta: dict[str, Any],
        entries: list[NkvEntry],
    ) -> None:
        self.dtype = dtype
        self.position = int(position)
        self.model_type = model_type
        self.meta = meta
        self.entries = entries
        self.by_name: dict[str, NkvEntry] = {e.name: e for e in entries}

    def clone(self) -> "NkvCartridge":
        copied = [
            NkvEntry(
                name=e.name,
                kind=e.kind,
                width=e.width,
                horizon=e.horizon,
                first_pos=e.first_pos,
                rows=e.rows,
                data=e.data.copy(),
            )
            for e in self.entries
        ]
        return NkvCartridge(
            dtype=self.dtype,
            position=self.position,
            model_type=self.model_type,
            meta=json.loads(json.dumps(self.meta)),
            entries=copied,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NkvCartridge":
        raw = Path(path).read_bytes()
        magic, ver, dt_code, pos, count, mt_len, mj_len, _flags, _ = struct.unpack_from(
            cls.HDR_FMT, raw, 0
        )
        if magic != cls.MAGIC or ver != 1:
            raise ValueError(f"{path}: bad NANOKV01 header ({magic!r}, v{ver})")
        dtype = {0: "f32", 1: "bf16"}[dt_code]
        np_dt = np.float32 if dtype == "f32" else np.uint16
        cur = 64
        model_type = raw[cur : cur + mt_len].decode("utf-8")
        cur += align8(mt_len)
        meta_raw = raw[cur : cur + mj_len].decode("utf-8") if mj_len else "{}"
        meta = json.loads(meta_raw) if meta_raw.strip() else {}
        cur += align8(mj_len)
        entries: list[NkvEntry] = []
        for _ in range(count):
            name_len, kind, width, horizon, first_pos, rows, doff, dbytes = struct.unpack_from(
                cls.DIR_FMT, raw, cur
            )
            name = raw[cur + 40 : cur + 40 + name_len].decode("utf-8")
            cur += 40 + align8(name_len)
            arr = np.frombuffer(raw, dtype=np_dt, count=rows * width, offset=doff).reshape(rows, width).copy()
            assert arr.nbytes == dbytes
            entries.append(NkvEntry(name=name, kind=kind, width=width, horizon=horizon, first_pos=first_pos, rows=rows, data=arr))
        return cls(dtype=dtype, position=pos, model_type=model_type, meta=meta, entries=entries)

    def save(self, path: str | Path, dtype: str | None = None) -> int:
        out_dtype = dtype or self.dtype
        dt_code = {"f32": 0, "bf16": 1}[out_dtype]
        mt_bytes = self.model_type.encode("utf-8")
        mj_bytes = json.dumps(self.meta, separators=(",", ":")).encode("utf-8")
        hdr_size = 64 + align8(len(mt_bytes)) + align8(len(mj_bytes))
        for e in self.entries:
            hdr_size += 40 + align8(len(e.name.encode("utf-8")))
        cursor = align8(hdr_size)
        prepared: list[tuple[NkvEntry, bytes, int, int]] = []
        for e in self.entries:
            arr = e.data
            if out_dtype == "bf16" and arr.dtype != np.uint16:
                arr = f32_to_u16(arr)
            elif out_dtype == "f32" and arr.dtype != np.float32:
                arr = u16_to_f32(arr)
            arr = np.ascontiguousarray(arr)
            dbytes = arr.nbytes
            prepared.append((e, arr.tobytes(), cursor, dbytes))
            cursor += align8(dbytes)

        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(
                struct.pack(
                    self.HDR_FMT,
                    self.MAGIC,
                    1,
                    dt_code,
                    self.position,
                    len(self.entries),
                    len(mt_bytes),
                    len(mj_bytes),
                    0,
                    b"\x00" * 28,
                )
            )
            f.write(mt_bytes + b"\x00" * (align8(len(mt_bytes)) - len(mt_bytes)))
            f.write(mj_bytes + b"\x00" * (align8(len(mj_bytes)) - len(mj_bytes)))
            for e, _, doff, dbytes in prepared:
                nb = e.name.encode("utf-8")
                f.write(
                    struct.pack(
                        self.DIR_FMT,
                        len(nb),
                        e.kind,
                        e.width,
                        e.horizon,
                        e.first_pos,
                        e.rows,
                        doff,
                        dbytes,
                    )
                )
                f.write(nb + b"\x00" * (align8(len(nb)) - len(nb)))
            pos_now = f.tell()
            if prepared and pos_now < prepared[0][2]:
                f.write(b"\x00" * (prepared[0][2] - pos_now))
            for _, payload, _, dbytes in prepared:
                f.write(payload)
                pad = align8(dbytes) - dbytes
                if pad:
                    f.write(b"\x00" * pad)
        os.replace(tmp, path)
        return cursor


def prefix_concat(cart_a: NkvCartridge, cart_b: NkvCartridge, header_tokens: int, dtype: str = "bf16") -> NkvCartridge:
    """Zero-RoPE prefix concatenation: header + doc_A + doc_B without running prefill again."""
    assert cart_a.model_type == cart_b.model_type
    assert cart_a.position >= header_tokens and cart_b.position >= header_tokens
    doc_b_len = cart_b.position - header_tokens
    pos_out = cart_a.position + doc_b_len
    out_entries: list[NkvEntry] = []
    for ea in cart_a.entries:
        eb = cart_b.by_name[ea.name]
        assert ea.kind == eb.kind and ea.width == eb.width and ea.horizon == eb.horizon
        a_f = ea.as_f32()
        b_f = eb.as_f32()
        if ea.kind == 2:  # .sequence (global layers)
            cat = np.concatenate([a_f[: cart_a.position], b_f[header_tokens : cart_b.position]], axis=0)
            entry = NkvEntry(ea.name, ea.kind, ea.width, ea.horizon, 0, pos_out, np.empty((0, ea.width), dtype=np.float32))
            entry.set_f32(cat, dtype, first_pos=0)
        else:  # .ring (sliding KV 1023 or conv 3)
            need = min(pos_out, ea.horizon)
            take_b = min(need, doc_b_len)
            take_a = need - take_b
            parts = []
            if take_a > 0:
                parts.append(a_f[ea.rows - take_a : ea.rows])
            if take_b > 0:
                parts.append(b_f[eb.rows - take_b : eb.rows])
            tail = np.concatenate(parts, axis=0) if parts else np.zeros((0, ea.width), dtype=np.float32)
            first_pos = pos_out - tail.shape[0]
            entry = NkvEntry(ea.name, ea.kind, ea.width, ea.horizon, first_pos, tail.shape[0], np.empty((0, ea.width), dtype=np.float32))
            entry.set_f32(tail, dtype, first_pos=first_pos)
        out_entries.append(entry)
    return NkvCartridge(
        dtype=dtype,
        position=pos_out,
        model_type=cart_a.model_type,
        meta={"concat": True, "pos_a": cart_a.position, "doc_b_len": doc_b_len, "header_tokens": header_tokens},
        entries=out_entries,
    )


def parse_visit_name(name: str) -> tuple[int, str]:
    # "visit.<idx>.<suffix>"
    parts = name.split(".", 2)
    return int(parts[1]), parts[2]


class NanoRunner:
    """Thin subprocess wrapper around zig-out/bin/nano (always sequential on GPU)."""

    def __init__(self, nano: Path, checkpoint: Path, knobs: Path, context: int, work: Path) -> None:
        self.nano = nano
        self.checkpoint = checkpoint
        self.knobs = knobs
        self.context = context
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(self.checkpoint), local_files_only=True, trust_remote_code=False)
            self._tok = tok
        return self._tok

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_ids(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _run_json(self, cmd: list[str]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        res = subprocess.run(cmd, text=True, capture_output=True)
        dt = time.perf_counter() - t0
        if res.returncode != 0:
            raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip().startswith("{")]
        if not lines:
            raise RuntimeError(f"No JSON line in output of {' '.join(cmd)}:\n{res.stdout}\n{res.stderr}")
        return json.loads(lines[-1]), dt

    def encode_cartridge(
        self,
        tokens: list[int],
        kv_out: Path,
        *,
        loop_passes: int = 3,
        kv_dtype: str = "bf16",
        hidden_out: Path | None = None,
        context: int | None = None,
    ) -> dict[str, Any]:
        tok_path = self.work / "tmp_encode_tokens.json"
        tok_path.write_text(json.dumps(tokens))
        cmd = [
            str(self.nano),
            "encode",
            str(self.checkpoint),
            "--context",
            str(context or self.context),
            "--knobs",
            str(self.knobs),
            "--loop-passes",
            str(loop_passes),
            "--tokens-file",
            str(tok_path),
            "--kv-out",
            str(kv_out),
            "--kv-dtype",
            kv_dtype,
        ]
        if hidden_out is not None:
            cmd += ["--hidden-out", str(hidden_out)]
        out, wall = self._run_json(cmd)
        out["wall_seconds"] = wall
        return out

    def eval_logits_and_nll(
        self,
        prompt_tokens: list[int],
        target_tokens: list[int],
        *,
        kv_in: Path | None = None,
        loop_passes: int = 3,
        context: int | None = None,
    ) -> dict[str, Any]:
        """Compute exact Inkling-2-Mini NLL and top-1 predictions over target_tokens."""
        assert len(target_tokens) >= 1
        feed_tokens = prompt_tokens + target_tokens[:-1]
        n_prompt = len(prompt_tokens)
        n_tgt = len(target_tokens)
        positions = list(range(n_prompt - 1, n_prompt - 1 + n_tgt))
        tok_path = self.work / "tmp_eval_tokens.json"
        pos_path = self.work / "tmp_eval_pos.json"
        logits_path = self.work / "tmp_eval_logits.f32"
        tok_path.write_text(json.dumps(feed_tokens))
        pos_path.write_text(json.dumps({"positions": positions}))
        cmd = [
            str(self.nano),
            "eval",
            str(self.checkpoint),
            "--context",
            str(context or self.context),
            "--knobs",
            str(self.knobs),
            "--loop-passes",
            str(loop_passes),
            "--tokens-file",
            str(tok_path),
            "--positions-file",
            str(pos_path),
            "--feed",
            str(min(4096, len(feed_tokens))),
            "--logits-out",
            str(logits_path),
        ]
        if kv_in is not None:
            cmd += ["--kv-in", str(kv_in)]
        meta, _ = self._run_json(cmd)
        vocab = int(meta["vocab"])
        logits = np.fromfile(logits_path, dtype=np.float32).reshape(n_tgt, vocab)
        max_l = np.max(logits, axis=1, keepdims=True)
        lse = max_l[:, 0] + np.log(np.sum(np.exp(logits - max_l), axis=1))
        tgt_logits = logits[np.arange(n_tgt), np.array(target_tokens, dtype=np.int64)]
        nll_per_tok = lse - tgt_logits
        top1 = np.argmax(logits, axis=1)
        exact = (top1 == np.array(target_tokens, dtype=np.int64))
        return {
            "mean_nll": float(np.mean(nll_per_tok)),
            "first_tok_nll": float(nll_per_tok[0]),
            "top1_vs_gt": float(np.mean(exact)),
            "top1_ids": [int(x) for x in top1],
            "logits_first": logits[0].copy(),
        }

    def eval_attention(
        self,
        probe_tokens: list[int],
        *,
        kv_in: Path,
        context: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run nano eval with --heads-out --attention-out from kv_in and return [T, V, ctx] weights."""
        tok_path = self.work / "tmp_attn_tokens.json"
        heads_path = self.work / "tmp_heads.f32"
        attn_path = self.work / "tmp_attn.f32"
        tok_path.write_text(json.dumps(probe_tokens))
        cmd = [
            str(self.nano),
            "eval",
            str(self.checkpoint),
            "--context",
            str(context or self.context),
            "--knobs",
            str(self.knobs),
            "--kv-in",
            str(kv_in),
            "--tokens-file",
            str(tok_path),
            "--heads-out",
            str(heads_path),
            "--heads-from",
            "0",
            "--attention-out",
            str(attn_path),
        ]
        meta, wall = self._run_json(cmd)
        meta["wall_seconds"] = wall
        T = int(meta["tap_forwards"])
        V = len(meta["tap_windows"])
        C = int(meta["tap_context"])
        w = np.fromfile(attn_path, dtype=np.float32).reshape(T, V, C)
        return w, meta

    def session_run_batch(self, requests: list[dict[str, Any]], *, context: int | None = None) -> list[dict[str, Any]]:
        """Run requests sequentially in one persistent 3-pass Inkling-2-Mini session."""
        cmd = [
            str(self.nano),
            "session",
            str(self.checkpoint),
            "--context",
            str(context or self.context),
            "--knobs",
            str(self.knobs),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        responses = []
        assert proc.stdin and proc.stdout
        try:
            for req in requests:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    err = proc.stderr.read() if proc.stderr else ""
                    raise RuntimeError(f"nano session exited early while running req {req.get('id')}:\n{err}")
                responses.append(json.loads(line))
            proc.stdin.close()
            rc = proc.wait(timeout=30)
            if rc != 0:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"nano session failed rc={rc}:\n{err}")
        finally:
            if proc.poll() is None:
                proc.kill()
        return responses


def cosine_sim_rows(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.maximum(na * nb, 1e-12)
    return float(np.mean(np.sum(a * b, axis=-1) / denom))


# ---------------------------------------------------------------------------
# Translators (Pass-Copy, Extrap, Ridge, UnifiedKVTranslator in MLX)
# ---------------------------------------------------------------------------

def build_template_3p_from_2p(cart_2p: NkvCartridge, cart_3p_template: NkvCartridge) -> NkvCartridge:
    """Zero-training pass-copy mapping 2-pass Inkling #1 -> 3-pass Inkling #2."""
    out = cart_3p_template.clone()
    out.position = cart_2p.position
    for e in out.entries:
        v, suffix = parse_visit_name(e.name)
        if v < 30:
            src_name = e.name
        elif v < 42:
            src_name = f"visit.{v - 12}.{suffix}"
        else:
            src_name = f"visit.{v - 12}.{suffix}"
        src = cart_2p.by_name[src_name]
        e.first_pos = src.first_pos
        e.rows = src.rows
        e.data = src.data.copy()
    return out


def build_template_3p_from_1p(cart_1p: NkvCartridge, cart_3p_template: NkvCartridge) -> NkvCartridge:
    """Zero-training pass-copy mapping 1-pass Inkling #1 -> 3-pass Inkling #2."""
    out = cart_3p_template.clone()
    out.position = cart_1p.position
    for e in out.entries:
        v, suffix = parse_visit_name(e.name)
        if v < 18:
            src_name = e.name
        elif v < 42:
            layer_off = (v - 6) % 12
            src_name = f"visit.{6 + layer_off}.{suffix}"
        else:
            src_name = f"visit.{v - 24}.{suffix}"
        src = cart_1p.by_name[src_name]
        e.first_pos = src.first_pos
        e.rows = src.rows
        e.data = src.data.copy()
    return out


def apply_pass_extrap_2p(cart_2p: NkvCartridge, cart_3p_template: NkvCartridge, alpha: float = 0.25) -> NkvCartridge:
    """Zero-training pass extrapolation V[pass2] = V[pass1] + alpha * (V[pass1] - V[pass0])."""
    out = build_template_3p_from_2p(cart_2p, cart_3p_template)
    for e in out.entries:
        v, suffix = parse_visit_name(e.name)
        if 30 <= v < 42 and suffix in ("keys", "values"):
            p0 = cart_2p.by_name[f"visit.{v - 24}.{suffix}"].as_f32()
            p1 = cart_2p.by_name[f"visit.{v - 12}.{suffix}"].as_f32()
            pred = apply_inkling_invariants(e.name, p1 + alpha * (p1 - p0))
            e.set_f32(pred, out.dtype)
    return out


def fit_ridge_translator(
    train_pairs: list[tuple[NkvCartridge, NkvCartridge]],
    passes: int = 2,
    reg_frac: float = 0.05,
) -> dict[str, tuple[list[str], np.ndarray, str]]:
    """Solve closed-form Ridge residual predictors on M3 Ultra via MLX float32 matmul/solve."""
    import mlx.core as mx

    # Determine which target states to translate: Pass2+Coda KV (for 2p) or Pass1+2+Coda KV (for 1p).
    start_v = 30 if passes == 2 else 18
    sample_src, sample_tgt = train_pairs[0]
    models: dict[str, tuple[list[str], np.ndarray, str]] = {}

    # Wait: precompute X_global per train pair if needed, and per-visit features directly.
    global_end_p = f"visit.{17 if passes == 1 else 29}"
    coda_global_src = f"visit.{23 if passes == 1 else 35}"

    def extract_pair_features(src: NkvCartridge, tgt_entry: NkvEntry) -> tuple[list[str], np.ndarray, np.ndarray]:
        v, suffix = parse_visit_name(tgt_entry.name)
        is_seq = tgt_entry.kind == 2
        if v < 42:
            off = (v - 6) % 12
            if passes == 2:
                feat_names = [
                    f"visit.{6 + off}.keys",
                    f"visit.{6 + off}.values",
                    f"visit.{18 + off}.keys",
                    f"visit.{18 + off}.values",
                ]
                base_name = f"visit.{18 + off}.{suffix}"
            else:
                if 6 + off == 17:
                    feat_names = ["visit.17.keys", "visit.17.values", "visit.5.keys", "visit.5.values"]
                else:
                    feat_names = [f"visit.{6 + off}.keys", f"visit.{6 + off}.values", "visit.17.keys", "visit.17.values"]
                base_name = f"visit.{6 + off}.{suffix}"
        else:
            c_off = v - 42
            src_v = (18 if passes == 1 else 30) + c_off
            feat_names = [
                f"visit.{src_v}.keys",
                f"visit.{src_v}.values",
                f"{global_end_p}.keys",
                f"{global_end_p}.values",
            ]
            if passes == 2 and v != 47:
                feat_names += [f"{coda_global_src}.keys", f"{coda_global_src}.values"]
            base_name = f"visit.{src_v}.{suffix}"

        mats = []
        rows = tgt_entry.rows
        for fn in feat_names:
            arr = src.by_name[fn].as_f32()
            if arr.shape[0] != rows:
                arr = arr[arr.shape[0] - rows :]
            mats.append(arr)
        mats.append(np.ones((rows, 1), dtype=np.float32))
        X = np.concatenate(mats, axis=1)
        base = src.by_name[base_name].as_f32()
        return feat_names, X, base

    # Compute on MLX GPU in CPU-fallback-safe float32
    t0 = time.perf_counter()
    for te in sample_tgt.entries:
        v, suffix = parse_visit_name(te.name)
        if v < start_v or suffix not in ("keys", "values"):
            continue
        Xs, Deltas = [], []
        feat_names: list[str] = []
        for src, tgt in train_pairs:
            tgt_e = tgt.by_name[te.name]
            feat_names, X, base = extract_pair_features(src, tgt_e)
            Delta = tgt_e.as_f32() - base
            Xs.append(X)
            Deltas.append(Delta)
        X_all = np.concatenate(Xs, axis=0)
        D_all = np.concatenate(Deltas, axis=0)
        D = X_all.shape[1]
        X_mx = mx.array(X_all)
        D_mx = mx.array(D_all)
        XtX = X_mx.T @ X_mx
        XtD = X_mx.T @ D_mx
        mx.eval(XtX, XtD)
        XtX_np = np.array(XtX, dtype=np.float64)
        XtD_np = np.array(XtD, dtype=np.float64)
        lam = reg_frac * (np.trace(XtX_np) / max(1, D))
        XtX_np[np.diag_indices(D)] += lam
        W = np.linalg.solve(XtX_np, XtD_np).astype(np.float32)
        base_rule = (
            f"visit.{18 + (v - 6) % 12}.{suffix}"
            if (passes == 2 and v < 42)
            else f"visit.{v - 12}.{suffix}"
            if passes == 2
            else (f"visit.{6 + (v - 6) % 12}.{suffix}" if v < 42 else f"visit.{v - 24}.{suffix}")
        )
        models[te.name] = (feat_names, W, base_rule)
    dt = time.perf_counter() - t0
    print(f"[ridge passes={passes}] fit {len(models)} KV translators in {dt:.2f}s", flush=True)
    return models


def apply_ridge_translator(
    src_cart: NkvCartridge,
    cart_3p_template: NkvCartridge,
    models: dict[str, tuple[list[str], np.ndarray, str]],
    passes: int = 2,
) -> NkvCartridge:
    import mlx.core as mx

    out = build_template_3p_from_2p(src_cart, cart_3p_template) if passes == 2 else build_template_3p_from_1p(src_cart, cart_3p_template)
    for name, (feat_names, W, base_rule) in models.items():
        e = out.by_name[name]
        rows = e.rows
        mats = []
        for fn in feat_names:
            arr = src_cart.by_name[fn].as_f32()
            if arr.shape[0] != rows:
                arr = arr[arr.shape[0] - rows :]
            mats.append(arr)
        mats.append(np.ones((rows, 1), dtype=np.float32))
        X = np.concatenate(mats, axis=1)
        base = src_cart.by_name[base_rule].as_f32()
        pred_delta = np.array(mx.array(X) @ mx.array(W), dtype=np.float32)
        pred = apply_inkling_invariants(name, base + pred_delta)
        e.set_f32(pred, out.dtype)
    return out


# --- MLX UnifiedKVTranslator (inspired by b10-tv/speculative-encoding) ---

def train_mlx_unified_translator(
    train_pairs: list[tuple[NkvCartridge, NkvCartridge]],
    ridge_models: dict[str, tuple[list[str], np.ndarray, str]],
    *,
    steps: int = 360,
    lr: float = 1e-3,
    width: int = 256,
) -> tuple[Any, dict[str, Any]]:
    """Train a compact multi-stream UnifiedKVTranslator in MLX on M3 Ultra GPU.

    Reads the 12 full-sequence global KV streams already present in 2p.nkv
    (visits 5, 11, 17, 23, 29, 35 K+V) with zero extra forward cost on Inkling #1,
    runs 2 cross-stream + local-causal-conv mixer blocks, and predicts residuals for
    the 3 target global visits (35, 41, 47 K+V = 6 target streams).
    Combines with trained Ridge predictors on local sliding-window layers so every
    layer in the cartridge is translated cleanly at any sequence length T.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    SRC_NAMES = [f"visit.{v}.{kv}" for v in INKLING_GLOBAL_VISITS_2P for kv in ("keys", "values")]  # 12 streams
    TGT_NAMES = [f"visit.{v}.{kv}" for v in (35, 41, 47) for kv in ("keys", "values")]  # 6 target streams
    BASE_NAMES = [f"visit.{v}.{kv}" for v in (23, 29, 35) for kv in ("keys", "values")]  # pass-copy base

    class SwiGLU(nn.Module):
        def __init__(self, d: int, mult: int = 3):
            super().__init__()
            h = d * mult
            self.w1 = nn.Linear(d, h, bias=False)
            self.w3 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)
            self.w2.weight = mx.zeros_like(self.w2.weight)

        def __call__(self, x):
            return self.w2(nn.silu(self.w1(x)) * self.w3(x))

    class UnifiedStreamBlock(nn.Module):
        def __init__(self, n_streams: int, d: int):
            super().__init__()
            self.n_streams = n_streams
            self.d = d
            self.n_stream = nn.RMSNorm(d)
            self.qkv_s = nn.Linear(d, d * 3, bias=False)
            self.out_s = nn.Linear(d, d, bias=False)
            self.out_s.weight = mx.zeros_like(self.out_s.weight)
            # Causal depthwise 4-tap token mixer across adjacent positions (models conv state propagation!)
            self.n_tok = nn.RMSNorm(d)
            self.tok_w = mx.zeros((4, n_streams, d))
            self.tok_w = self.tok_w + mx.array([0.0, 0.0, 0.0, 0.1], dtype=mx.float32)[:, None, None]
            self.tok_proj = nn.Linear(d, d, bias=False)
            self.tok_proj.weight = mx.zeros_like(self.tok_proj.weight)
            # Feed-forward
            self.n_ffn = nn.RMSNorm(d)
            self.ffn = SwiGLU(d, mult=2)

        def __call__(self, x):
            # x: [T, S, D]
            T, S, D = x.shape
            h = self.n_stream(x)
            qkv = self.qkv_s(h).reshape(T, S, 3, 4, D // 4)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [T, S, 4, dh]
            q = q.transpose(0, 2, 1, 3)  # [T, 4, S, dh]
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D // 4)
            attn = mx.softmax(scores, axis=-1) @ v
            x = x + self.out_s(attn.transpose(0, 2, 1, 3).reshape(T, S, D))

            # Causal 4-tap local sequence mixing along T
            ht = self.n_tok(x)
            p = mx.pad(ht, [(3, 0), (0, 0), (0, 0)])
            win = (
                p[0:T] * self.tok_w[0][None]
                + p[1 : T + 1] * self.tok_w[1][None]
                + p[2 : T + 2] * self.tok_w[2][None]
                + p[3 : T + 3] * self.tok_w[3][None]
            )
            x = x + self.tok_proj(win)
            x = x + self.ffn(self.n_ffn(x))
            return x

    class UnifiedKVTranslator(nn.Module):
        def __init__(self, d: int = width):
            super().__init__()
            self.in_norms = [nn.RMSNorm(1024) for _ in SRC_NAMES]
            self.in_projs = [nn.Linear(1024, d, bias=False) for _ in SRC_NAMES]
            self.stream_emb = mx.random.normal((len(SRC_NAMES), d)) * 0.02
            self.blocks = [UnifiedStreamBlock(len(SRC_NAMES), d) for _ in range(2)]
            # Target query readout from the 12 source streams to 6 target streams
            self.read_q = mx.random.normal((len(TGT_NAMES), d)) * 0.02
            self.key_map = nn.Linear(d, d, bias=False)
            self.val_map = nn.Linear(d, d, bias=False)
            self.out_norms = [nn.RMSNorm(d) for _ in TGT_NAMES]
            self.out_projs = [nn.Linear(d, 1024, bias=True) for _ in TGT_NAMES]
            self.out_scales = [mx.ones((1,)) * 0.5 for _ in TGT_NAMES]
            for op in self.out_projs:
                op.weight = mx.zeros_like(op.weight)
                op.bias = mx.zeros_like(op.bias)

        def __call__(self, srcs: list[mx.array], bases: list[mx.array]) -> list[mx.array]:
            # srcs: 12 x [T, 1024], bases: 6 x [T, 1024]
            T = srcs[0].shape[0]
            cols = []
            for i in range(len(SRC_NAMES)):
                cols.append(self.in_projs[i](self.in_norms[i](srcs[i])) + self.stream_emb[i][None])
            x = mx.stack(cols, axis=1)  # [T, 12, D]
            for blk in self.blocks:
                x = blk(x)
            # Cross-attend 6 target queries over 12 encoded source streams per token
            k = self.key_map(x)  # [T, 12, D]
            v = self.val_map(x)
            scores = mx.einsum("gd,tsd->tgs", self.read_q, k) / math.sqrt(k.shape[-1])
            w = mx.softmax(scores, axis=-1)
            pooled = mx.einsum("tgs,tsd->tgd", w, v)  # [T, 6, D]
            preds = []
            for j in range(len(TGT_NAMES)):
                delta = self.out_projs[j](self.out_norms[j](pooled[:, j, :])) * self.out_scales[j]
                raw = bases[j] + delta
                name = TGT_NAMES[j]
                if name.endswith(".values"):
                    h = raw.reshape(T, INKLING_KV_HEADS, INKLING_HEAD_DIM)
                    h = h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + INKLING_RMS_EPS)
                    raw = h.reshape(T, 1024)
                else:
                    h = raw.reshape(T, INKLING_KV_HEADS, INKLING_HEAD_DIM)
                    nrm = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True))
                    scale = mx.minimum(INKLING_QK_CLIP / (nrm + 1e-6), 1.0)
                    raw = (h * scale).reshape(T, 1024)
                preds.append(raw)
            return preds

    # Prepare GPU arrays for each training document (~1536 x 1024 bf16 -> fp32 on the fly).
    docs_mx = []
    for src_cart, tgt_cart in train_pairs:
        srcs = [mx.array(src_cart.by_name[n].as_f32()) for n in SRC_NAMES]
        bases = [mx.array(src_cart.by_name[n].as_f32()) for n in BASE_NAMES]
        tgts = [mx.array(tgt_cart.by_name[n].as_f32()) for n in TGT_NAMES]
        docs_mx.append((srcs, bases, tgts))

    model = UnifiedKVTranslator(width)
    mx.eval(model.parameters())

    def loss_fn(model, srcs, bases, tgts):
        preds = model(srcs, bases)
        loss = mx.array(0.0)
        for p, t in zip(preds, tgts):
            dot = mx.sum(p * t, axis=-1)
            na = mx.sqrt(mx.sum(p * p, axis=-1) + 1e-8)
            nb = mx.sqrt(mx.sum(t * t, axis=-1) + 1e-8)
            cos_loss = mx.mean(1.0 - dot / (na * nb))
            mse_loss = mx.mean((p - t) ** 2)
            loss = loss + cos_loss + 0.15 * mse_loss
        return loss / len(preds)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    t0 = time.perf_counter()
    losses = []
    for step in range(steps):
        idx = step % len(docs_mx)
        srcs, bases, tgts = docs_mx[idx]
        # Random slice window of up to 768 tokens per step (still > causal conv radius, fast updates)
        T_doc = srcs[0].shape[0]
        if T_doc > 768 and step % 2 == 1:
            start = (step * 197) % (T_doc - 768)
            s_in = [x[start : start + 768] for x in srcs]
            b_in = [x[start : start + 768] for x in bases]
            t_in = [x[start : start + 768] for x in tgts]
        else:
            s_in, b_in, t_in = srcs, bases, tgts
        lval, grads = loss_and_grad(model, s_in, b_in, t_in)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, lval)
        losses.append(float(lval.item()))
    train_s = time.perf_counter() - t0
    info = {
        "steps": steps,
        "train_seconds": train_s,
        "initial_loss": float(np.mean(losses[: max(1, len(docs_mx))])),
        "final_loss": float(np.mean(losses[-len(docs_mx) :])),
        "src_names": SRC_NAMES,
        "tgt_names": TGT_NAMES,
        "base_names": BASE_NAMES,
    }
    print(f"[mlx-unified-kv] trained {steps} steps on {len(docs_mx)} docs in {train_s:.2f}s "
          f"(loss {info['initial_loss']:.4f} -> {info['final_loss']:.4f})", flush=True)
    return (model, ridge_models, SRC_NAMES, TGT_NAMES, BASE_NAMES), info


def apply_mlx_unified_translator(src_cart: NkvCartridge, cart_3p_template: NkvCartridge, bundle) -> tuple[NkvCartridge, float]:
    import mlx.core as mx

    model, ridge_models, SRC_NAMES, TGT_NAMES, BASE_NAMES = bundle
    t0 = time.perf_counter()
    # Start from ridge for local layers, then overwrite the 6 global streams with UnifiedKVTranslator.
    out = apply_ridge_translator(src_cart, cart_3p_template, ridge_models, passes=2)
    srcs = [mx.array(src_cart.by_name[n].as_f32()) for n in SRC_NAMES]
    bases = [mx.array(src_cart.by_name[n].as_f32()) for n in BASE_NAMES]
    preds = model(srcs, bases)
    mx.eval(*preds)
    for name, p in zip(TGT_NAMES, preds):
        out.by_name[name].set_f32(bf16_round_f32(np.array(p, dtype=np.float32)), out.dtype)
    dt = time.perf_counter() - t0
    return out, dt


def kv_similarity_metrics(cand: NkvCartridge, ref: NkvCartridge, min_visit: int = 30) -> dict[str, float]:
    g_k, g_v, l_k, l_v = [], [], [], []
    for e_ref in ref.entries:
        v, suf = parse_visit_name(e_ref.name)
        if v < min_visit or suf not in ("keys", "values"):
            continue
        c = cosine_sim_rows(cand.by_name[e_ref.name].as_f32(), e_ref.as_f32())
        if e_ref.kind == 2:
            (g_k if suf == "keys" else g_v).append(c)
        else:
            (l_k if suf == "keys" else l_v).append(c)
    return {
        "global_k_cos": float(np.mean(g_k)),
        "global_v_cos": float(np.mean(g_v)),
        "local_k_cos": float(np.mean(l_k)),
        "local_v_cos": float(np.mean(l_v)),
    }


# ---------------------------------------------------------------------------
# Disjoint Corpora Builders (Zero Data Leakage!)
# ---------------------------------------------------------------------------

def load_doc_text(rel: str, limit_chars: int = 12000) -> str:
    return (REPO_ROOT / rel).read_text(errors="replace")[:limit_chars]


def slice_doc_tokens(runner: NanoRunner, header_ids: list[int], text: str, total_len: int) -> list[int]:
    body_ids = runner.encode_text(text)
    room = total_len - len(header_ids)
    assert len(body_ids) >= room, f"Need >= {room} tokens, got {len(body_ids)}"
    return header_ids + body_ids[:room]


# ---------------------------------------------------------------------------
# Experiment A: 1p/2p Inkling -> 3p Inkling Cross-Pass Speculative Encoding
# ---------------------------------------------------------------------------

def run_exp_crosspass(runner: NanoRunner, work: Path) -> dict[str, Any]:
    print("\n================ EXPERIMENT A: 2-Inkling Cross-Pass Encoding (1p/2p -> 3p) ================", flush=True)
    exp_dir = work / "exp_a"
    exp_dir.mkdir(parents=True, exist_ok=True)
    hdr_ids = runner.encode_text(SHARED_HEADER_TEXT)

    # Train docs (disjoint files from all test files)
    train_sources = [
        load_doc_text("README.md", 18000),
        load_doc_text("docs/MEGAKERNEL.md", 22000),
        load_doc_text("docs/CODEGRAFF.md", 20000),
        load_doc_text("docs/CLI.md", 16000),
        load_doc_text("docs/LFM3.md", 20000),
        load_doc_text("docs/MLX-PREFILL.md", 20000),
    ]
    train_len = 1536
    train_pairs_2p: list[tuple[NkvCartridge, NkvCartridge]] = []
    train_pairs_1p: list[tuple[NkvCartridge, NkvCartridge]] = []

    for i, txt in enumerate(train_sources):
        ids = slice_doc_tokens(runner, hdr_ids, txt, train_len)
        p3 = exp_dir / f"train_{i}_3p.nkv"
        p2 = exp_dir / f"train_{i}_2p.nkv"
        p1 = exp_dir / f"train_{i}_1p.nkv"
        runner.encode_cartridge(ids, p3, loop_passes=3, kv_dtype="bf16")
        runner.encode_cartridge(ids, p2, loop_passes=2, kv_dtype="bf16")
        runner.encode_cartridge(ids, p1, loop_passes=1, kv_dtype="bf16")
        c3 = NkvCartridge.load(p3)
        c2 = NkvCartridge.load(p2)
        c1 = NkvCartridge.load(p1)
        train_pairs_2p.append((c2, c3))
        train_pairs_1p.append((c1, c3))
    print(f"[exp-a] prepared {len(train_pairs_2p)} training docs x {train_len} tokens", flush=True)

    ridge_2p = fit_ridge_translator(train_pairs_2p, passes=2)
    ridge_1p = fit_ridge_translator(train_pairs_1p, passes=1)
    mlx_bundle, train_info = train_mlx_unified_translator(train_pairs_2p, ridge_2p, steps=360, lr=1e-3, width=256)

    # Disjoint held-out test documents (~2048 tokens each):
    # Embed specific early facts (around pos ~250, strictly outside local window 1024!) + NLL continuation.
    qd_full = (
        load_doc_text("docs/QD.md", 12000)
        + "\n\n"
        + load_doc_text("docs/SPECULATIVE.md", 12000)
        + "\n\n"
        + load_doc_text("benchmarks/bf16-kv.md", 12000)
    )
    shc_full = (
        load_doc_text("benchmarks/speculative-hillclimb.md", 18000)
        + "\n\n"
        + load_doc_text("benchmarks/decode-hillclimb.md", 18000)
    )
    zig_full = (
        load_doc_text("src/mega/ir.zig", 16000)
        + "\n\n"
        + load_doc_text("src/mega/frontend.zig", 12000)
        + "\n\n"
        + load_doc_text("src/mega/cli.zig", 8000)
    )

    test_specs = [
        {
            "name": "test_qd_fact",
            "doc_text": "Note: Internal calibration code AZURE-FALCON-742 was verified on cluster omega-9.\n\n" + qd_full,
            "prompt_len": 2048,
            "question": "What internal calibration code and cluster name were recorded in the document?",
            "expected": ["AZURE-FALCON-742", "omega-9"],
        },
        {
            "name": "test_hillclimb",
            "doc_text": "Audit Tag: SPEC-RECORD-9183 achieved 166.83 tok/s in trial batch K2.\n\n" + shc_full,
            "prompt_len": 2048,
            "question": "Which audit tag and tok/s speed were recorded at the start of the report?",
            "expected": ["SPEC-RECORD-9183", "166.83"],
        },
        {
            "name": "test_zig_code",
            "doc_text": "// Anchor constant: NANO_SPEC_MAGIC = 0x5A17C0DE in IR verifier\n" + zig_full,
            "prompt_len": 2048,
            "question": "What hex value is assigned to the anchor constant NANO_SPEC_MAGIC?",
            "expected": ["5A17C0DE"],
        },
    ]

    method_names = ["native-3p", "pass2p-copy", "pass2p-extrap", "ridge-2p", "mlx-unified-kv", "pass1p-copy", "ridge-1p"]
    method_agg: dict[str, dict[str, list[float]]] = {
        m: {"mean_nll": [], "top1_vs_3p": [], "qa_hit": [], "global_k_cos": [], "global_v_cos": [], "local_k_cos": [], "local_v_cos": []}
        for m in method_names
    }
    per_doc_records = []

    for t_i, spec in enumerate(test_specs):
        full_ids = slice_doc_tokens(runner, hdr_ids, spec["doc_text"], spec["prompt_len"] + 33)
        ids = full_ids[: spec["prompt_len"]]
        p3 = exp_dir / f"test_{t_i}_3p.nkv"
        p2 = exp_dir / f"test_{t_i}_2p.nkv"
        p1 = exp_dir / f"test_{t_i}_1p.nkv"
        runner.encode_cartridge(ids, p3, loop_passes=3, kv_dtype="bf16")
        runner.encode_cartridge(ids, p2, loop_passes=2, kv_dtype="bf16")
        runner.encode_cartridge(ids, p1, loop_passes=1, kv_dtype="bf16")
        c3 = NkvCartridge.load(p3)
        c2 = NkvCartridge.load(p2)
        c1 = NkvCartridge.load(p1)

        carts: dict[str, tuple[NkvCartridge, Path]] = {"native-3p": (c3, p3)}
        c_copy2 = build_template_3p_from_2p(c2, c3)
        p_copy2 = exp_dir / f"test_{t_i}_pass2p_copy.nkv"
        c_copy2.save(p_copy2)
        carts["pass2p-copy"] = (c_copy2, p_copy2)

        c_ext = apply_pass_extrap_2p(c2, c3, alpha=0.25)
        p_ext = exp_dir / f"test_{t_i}_pass2p_extrap.nkv"
        c_ext.save(p_ext)
        carts["pass2p-extrap"] = (c_ext, p_ext)

        c_r2 = apply_ridge_translator(c2, c3, ridge_2p, passes=2)
        p_r2 = exp_dir / f"test_{t_i}_ridge2p.nkv"
        c_r2.save(p_r2)
        carts["ridge-2p"] = (c_r2, p_r2)

        c_mlx, _ = apply_mlx_unified_translator(c2, c3, mlx_bundle)
        p_mlx = exp_dir / f"test_{t_i}_mlx_unified.nkv"
        c_mlx.save(p_mlx)
        carts["mlx-unified-kv"] = (c_mlx, p_mlx)

        c_copy1 = build_template_3p_from_1p(c1, c3)
        p_copy1 = exp_dir / f"test_{t_i}_pass1p_copy.nkv"
        c_copy1.save(p_copy1)
        carts["pass1p-copy"] = (c_copy1, p_copy1)

        c_r1 = apply_ridge_translator(c1, c3, ridge_1p, passes=1)
        p_r1 = exp_dir / f"test_{t_i}_ridge1p.nkv"
        c_r1.save(p_r1)
        carts["ridge-1p"] = (c_r1, p_r1)

        # True held-out continuation IDs + NLL evaluation from suffix of same document
        nll_probe = [full_ids[spec["prompt_len"]]]
        cont_ids = full_ids[spec["prompt_len"] + 1 : spec["prompt_len"] + 33]
        ref_eval = runner.eval_logits_and_nll(nll_probe, cont_ids, kv_in=p3)
        ref_top1 = np.array(ref_eval["top1_ids"], dtype=np.int64)

        # Greedy QA session evaluation across all candidate cartridges
        q_ids = runner.encode_text(f"\n\nQuestion: {spec['question']}{QA_OPEN_TEXT}")
        reqs = [{"id": j, "tokens": q_ids, "max_tokens": 48, "kv_in": str(carts[m][1])} for j, m in enumerate(method_names)]
        qa_resps = runner.session_run_batch(reqs)

        doc_summary: dict[str, Any] = {"name": spec["name"], "methods": {}}
        for m, resp in zip(method_names, qa_resps):
            cart, cpath = carts[m]
            ev = ref_eval if m == "native-3p" else runner.eval_logits_and_nll(nll_probe, cont_ids, kv_in=cpath)
            sim = kv_similarity_metrics(cart, c3, min_visit=30) if m != "native-3p" else {
                "global_k_cos": 1.0, "global_v_cos": 1.0, "local_k_cos": 1.0, "local_v_cos": 1.0
            }
            agree = float(np.mean(np.array(ev["top1_ids"], dtype=np.int64) == ref_top1))
            ans = runner.decode_ids(resp["tokens"]).strip()
            hit = float(all(exp.lower() in ans.lower() for exp in spec["expected"]))
            method_agg[m]["mean_nll"].append(ev["mean_nll"])
            method_agg[m]["top1_vs_3p"].append(agree)
            method_agg[m]["qa_hit"].append(hit)
            for k_sim, v_sim in sim.items():
                method_agg[m][k_sim].append(v_sim)
            doc_summary["methods"][m] = {
                "nll": ev["mean_nll"],
                "top1_vs_3p": agree,
                "qa_hit": hit,
                "answer": ans[:160],
                **sim,
            }
        per_doc_records.append(doc_summary)
        print(f"[exp-a] {spec['name']}: native nll={ref_eval['mean_nll']:.3f} qa='{doc_summary['methods']['native-3p']['answer'][:80]}'", flush=True)

    # Benchmark end-to-end 16K prefill + translate + inject speed!
    long_txt = "\n\n".join(train_sources * 2)
    ids_16k = slice_doc_tokens(runner, hdr_ids, long_txt, 16384)
    ctx_16k = 20480
    p16_3p = exp_dir / "bench16k_3p.nkv"
    p16_2p = exp_dir / "bench16k_2p.nkv"
    p16_1p = exp_dir / "bench16k_1p.nkv"
    # Warm run then timed encode runs
    runner.encode_cartridge(ids_16k[:4096], exp_dir / "warm16.nkv", loop_passes=3, context=ctx_16k)
    m16_3p = runner.encode_cartridge(ids_16k, p16_3p, loop_passes=3, context=ctx_16k)
    m16_2p = runner.encode_cartridge(ids_16k, p16_2p, loop_passes=2, context=ctx_16k)
    m16_1p = runner.encode_cartridge(ids_16k, p16_1p, loop_passes=1, context=ctx_16k)
    c16_3p = NkvCartridge.load(p16_3p)
    c16_2p = NkvCartridge.load(p16_2p)
    t_ridge0 = time.perf_counter()
    c16_ridge = apply_ridge_translator(c16_2p, c16_3p, ridge_2p, passes=2)
    ridge_only_s = time.perf_counter() - t_ridge0
    t_trans0 = time.perf_counter()
    c16_mlx, mlx_only_s = apply_mlx_unified_translator(c16_2p, c16_3p, mlx_bundle)
    p16_trans = exp_dir / "bench16k_2p_mlx.nkv"
    c16_mlx.save(p16_trans)
    trans_and_save_s = time.perf_counter() - t_trans0
    # Measure cartridge inject + 32-token generate wall in session
    q_16k = runner.encode_text(f"\n\nSummarize the runtime architecture.{QA_OPEN_TEXT}")
    inj_resps = runner.session_run_batch(
        [
            {"id": 0, "tokens": q_16k[:4], "max_tokens": 1, "reset": True},
            {"id": 1, "tokens": q_16k, "max_tokens": 32, "kv_in": str(p16_trans)},
        ],
        context=ctx_16k,
    )
    inject_prefill_s = float(inj_resps[1]["prefill_seconds"])

    summary_table = {
        m: {k: float(np.mean(vs)) for k, vs in agg.items()} for m, agg in method_agg.items()
    }
    return {
        "train_info": train_info,
        "summary": summary_table,
        "docs": per_doc_records,
        "bench_16k": {
            "tokens": 16384,
            "native_3p_encode_s": m16_3p["encode_seconds"],
            "native_3p_tok_s": m16_3p["tokens_per_second"],
            "native_3p_save_s": m16_3p["save_seconds"],
            "ink1_2p_encode_s": m16_2p["encode_seconds"],
            "ink1_2p_tok_s": m16_2p["tokens_per_second"],
            "ink1_1p_encode_s": m16_1p["encode_seconds"],
            "ink1_1p_tok_s": m16_1p["tokens_per_second"],
            "ridge_translate_compute_s": ridge_only_s,
            "mlx_translate_compute_s": mlx_only_s,
            "mlx_translate_and_save_s": trans_and_save_s,
            "ink2_inject_query_prefill_s": inject_prefill_s,
            "ink1_2p_plus_ridge_total_s": m16_2p["encode_seconds"] + ridge_only_s,
            "ink1_2p_plus_translate_total_s": m16_2p["encode_seconds"] + mlx_only_s,
            "speedup_2p_encode_only": m16_3p["encode_seconds"] / m16_2p["encode_seconds"],
            "speedup_2p_plus_ridge": m16_3p["encode_seconds"] / (m16_2p["encode_seconds"] + ridge_only_s),
            "speedup_2p_plus_translate": m16_3p["encode_seconds"] / (m16_2p["encode_seconds"] + mlx_only_s),
        },
    }


# ---------------------------------------------------------------------------
# Experiment B: Parallel 2-Inkling Document Encoding + Zero-RoPE Prefix Concat
# ---------------------------------------------------------------------------

def run_exp_concat(runner: NanoRunner, work: Path) -> dict[str, Any]:
    print("\n================ EXPERIMENT B: Parallel 2-Inkling Zero-RoPE Concatenation ================", flush=True)
    exp_dir = work / "exp_b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    header_ids = runner.encode_text(SHARED_HEADER_TEXT + "Read both engineering specifications carefully and answer precisely.\n\n")
    doc1_body = (
        "## Document 1: Subsystem Alpha (CryoLink)\n"
        "Project Code: CRYO-4419.\nThe primary coolant manifold pressure limit is 412 kPa and its bypass valve opens at step 17.\n\n"
        + load_doc_text("docs/LFM3.md", 14000)
    )
    doc2_body = (
        "## Document 2: Subsystem Beta (IonBus)\n"
        "Project Code: ION-7782.\nThe nominal telemetry baud rate is 921600 bps with a CRC-32C trailer.\n\n"
        + load_doc_text("docs/CLI.md", 14000)
    )
    d1_end = runner.encode_text("\n\nNote at end of Document 1: CryoLink reserve tanks hold 85 liters.\n")
    d2_end = runner.encode_text("\n\nNote at end of Document 2: IonBus uses dual-rail +12V / +5V regulators.\n")
    d1_ids = runner.encode_text(doc1_body)[: max(0, 2000 - len(d1_end))] + d1_end
    d2_ids = runner.encode_text(doc2_body)[: max(0, 2000 - len(d2_end))] + d2_end

    full_d1 = header_ids + d1_ids
    full_d2 = header_ids + d2_ids
    full_combined = header_ids + d1_ids + d2_ids

    p_d1 = exp_dir / "doc1.nkv"
    p_d2 = exp_dir / "doc2.nkv"
    p_native = exp_dir / "combined_native.nkv"
    p_concat = exp_dir / "combined_concat.nkv"

    m1 = runner.encode_cartridge(full_d1, p_d1, loop_passes=3, kv_dtype="bf16")
    m2 = runner.encode_cartridge(full_d2, p_d2, loop_passes=3, kv_dtype="bf16")
    mn = runner.encode_cartridge(full_combined, p_native, loop_passes=3, kv_dtype="bf16")

    t_cat0 = time.perf_counter()
    c_concat = prefix_concat(NkvCartridge.load(p_d1), NkvCartridge.load(p_d2), len(header_ids), dtype="bf16")
    c_concat.save(p_concat)
    concat_s = time.perf_counter() - t_cat0

    qa_cases = [
        {
            "name": "doc1_early",
            "question": "In Document 1 (Subsystem Alpha), what are the project code and primary coolant manifold pressure limit?",
            "expected": ["CRYO-4419", "412"],
        },
        {
            "name": "doc2_early",
            "question": "In Document 2 (Subsystem Beta), what are the project code and nominal telemetry baud rate?",
            "expected": ["ION-7782", "921600"],
        },
        {
            "name": "cross_doc_both",
            "question": "List BOTH project codes (for Document 1 and Document 2) plus the CryoLink pressure limit and IonBus voltage rails.",
            "expected": ["CRYO-4419", "ION-7782", "412", "12V"],
        },
    ]

    results_cases = []
    for qc in qa_cases:
        q_ids = runner.encode_text(f"\n\nQuestion: {qc['question']}{QA_OPEN_TEXT}")
        resps = runner.session_run_batch([
            {"id": 0, "tokens": q_ids, "max_tokens": 64, "kv_in": str(p_native)},
            {"id": 1, "tokens": q_ids, "max_tokens": 64, "kv_in": str(p_concat)},
        ])
        ans_nat = runner.decode_ids(resps[0]["tokens"]).strip()
        ans_cat = runner.decode_ids(resps[1]["tokens"]).strip()
        # Compare NLL/agreement of concat continuation tokens generated by native
        ref_toks = resps[0]["tokens"][:24] or q_ids[-2:]
        ev_nat = runner.eval_logits_and_nll(q_ids, ref_toks, kv_in=p_native)
        ev_cat = runner.eval_logits_and_nll(q_ids, ref_toks, kv_in=p_concat)
        agree = float(np.mean(np.array(ev_nat["top1_ids"]) == np.array(ev_cat["top1_ids"])))
        hit_nat = all(e.lower() in ans_nat.lower() for e in qc["expected"])
        hit_cat = all(e.lower() in ans_cat.lower() for e in qc["expected"])
        print(f"[exp-b] {qc['name']}: native_hit={hit_nat} concat_hit={hit_cat} top1_agree={agree:.2f}", flush=True)
        results_cases.append({
            "name": qc["name"],
            "expected": qc["expected"],
            "native_hit": hit_nat,
            "concat_hit": hit_cat,
            "native_nll_on_own_gen": ev_nat["mean_nll"],
            "concat_nll_on_native_gen": ev_cat["mean_nll"],
            "top1_agree": agree,
            "native_answer": ans_nat[:220],
            "concat_answer": ans_cat[:220],
        })

    return {
        "doc1_tokens": len(d1_ids),
        "doc2_tokens": len(d2_ids),
        "combined_tokens": c_concat.position,
        "doc1_encode_s": m1["encode_seconds"],
        "doc2_encode_s": m2["encode_seconds"],
        "native_combined_encode_s": mn["encode_seconds"],
        "concat_stitch_s": concat_s,
        "theoretical_2node_encode_s": max(m1["encode_seconds"], m2["encode_seconds"]) + concat_s,
        "cases": results_cases,
    }


# ---------------------------------------------------------------------------
# Experiment C: Query-Aware KVzip Compression (4x / 16x / 64x, Zero Training)
# ---------------------------------------------------------------------------

def kvzip_compress(
    cart: NkvCartridge,
    attn_weights: np.ndarray,  # [T_probe, 48, tap_context]
    tap_windows: list[int],
    header_tokens: int,
    ratio: int,
    mode: str = "shared",
) -> tuple[NkvCartridge, np.ndarray, list[np.ndarray]]:
    """Compress global KV sequence states by `ratio` using query-probe max attention weights."""
    doc_len = cart.position - header_tokens
    global_visits = [i for i, w in enumerate(tap_windows) if w == 0]
    assert len(global_visits) == 8
    tail_keep = min(doc_len, 64 if ratio <= 4 else (32 if ratio <= 16 else 16))
    target_doc_keep = max(16, doc_len // ratio)
    k_score_budget = max(8, target_doc_keep - tail_keep)

    # Probe weights over [header_tokens .. cart.position)
    # Use last half of probe tokens (closer to question end, skip initial whitespace/tags) or all probe tokens.
    doc_attn = attn_weights[:, global_visits, header_tokens : cart.position]  # [T_probe, 8, doc_len]
    per_visit_max = np.max(doc_attn, axis=0)  # [8, doc_len]
    per_visit_norm = per_visit_max / np.maximum(np.sum(per_visit_max, axis=-1, keepdims=True), 1e-12)
    shared_scores = np.mean(per_visit_norm, axis=0)  # [doc_len]

    # Exclude tail region from top-k selection so kept count matches target_doc_keep tightly.
    score_end = max(1, doc_len - tail_keep)
    shared_top = np.argsort(-shared_scores[:score_end])[:k_score_budget]
    shared_kept_doc = np.sort(
        np.unique(np.concatenate([shared_top, np.arange(doc_len - tail_keep, doc_len)]))
    )
    shared_kept_abs = np.concatenate([np.arange(header_tokens), header_tokens + shared_kept_doc]).astype(np.int64)
    pos_new = int(shared_kept_abs.shape[0])

    global_visit_idx_map = {gv: idx for idx, gv in enumerate(global_visits)}
    kept_per_visit: dict[int, np.ndarray] = {}
    out_entries: list[NkvEntry] = []
    for e in cart.entries:
        v, _ = parse_visit_name(e.name)
        arr = e.as_f32()
        if e.kind == 2:  # .sequence (global)
            if mode == "per-visit":
                gi = global_visit_idx_map[v]
                v_scores = per_visit_norm[gi]
                v_top = np.argsort(-v_scores[:score_end])[:k_score_budget]
                v_kept_doc = np.sort(np.unique(np.concatenate([v_top, np.arange(doc_len - tail_keep, doc_len)])))
                kept_abs = np.concatenate([np.arange(header_tokens), header_tokens + v_kept_doc[: shared_kept_doc.shape[0]]])
            else:
                kept_abs = shared_kept_abs
            kept_per_visit[v] = kept_abs
            sel = arr[kept_abs]
            ne = NkvEntry(e.name, e.kind, e.width, e.horizon, 0, pos_new, np.empty((0, e.width), dtype=np.float32))
            ne.set_f32(sel, cart.dtype, first_pos=0)
        else:  # .ring (sliding window 1023 or conv 3) -> preserve contiguous tail up to min(pos_new, horizon)
            keep_rows = min(pos_new, e.rows, e.horizon)
            tail = arr[e.rows - keep_rows : e.rows]
            first_pos = pos_new - keep_rows
            ne = NkvEntry(e.name, e.kind, e.width, e.horizon, first_pos, keep_rows, np.empty((0, e.width), dtype=np.float32))
            ne.set_f32(tail, cart.dtype, first_pos=first_pos)
        out_entries.append(ne)

    comp = NkvCartridge(
        dtype=cart.dtype,
        position=pos_new,
        model_type=cart.model_type,
        meta={
            "kvzip": True,
            "mode": mode,
            "ratio_target": ratio,
            "orig_position": cart.position,
            "kept_position": pos_new,
            "kept_sample": [int(x) for x in shared_kept_abs[:24]],
        },
        entries=out_entries,
    )
    return comp, shared_kept_abs, [kept_per_visit.get(gv, shared_kept_abs) for gv in global_visits]


def run_exp_kvzip(runner: NanoRunner, work: Path) -> dict[str, Any]:
    print("\n================ EXPERIMENT C: Query-Aware KVzip Compression (4x / 16x / 64x) ================", flush=True)
    exp_dir = work / "exp_c"
    exp_dir.mkdir(parents=True, exist_ok=True)
    hdr_ids = runner.encode_text(SHARED_HEADER_TEXT)

    # Construct ~8,192-token haystack document with 3 distinct needles at ~12%, ~48%, ~82%
    # (all strictly outside the final 1,024-token sliding window!).
    haystack_chunks = [
        load_doc_text("README.md", 12000),
        load_doc_text("docs/MEGAKERNEL.md", 16000),
        load_doc_text("docs/MLX-PREFILL.md", 16000),
        load_doc_text("docs/LFM3.md", 16000),
    ]
    filler_ids = runner.encode_text("\n\n".join(haystack_chunks))

    needles = [
        {
            "name": "early_needle",
            "pos_frac": 0.12,
            "fact_text": "\n[Archive Record #E1] The vault key is COBALT-719 and the harbor gate code is 4092.\n",
            "question": "In Archive Record #E1, what is the vault key and harbor gate code?",
            "expected": ["COBALT-719", "4092"],
        },
        {
            "name": "middle_needle",
            "pos_frac": 0.48,
            "fact_text": "\n[Archive Record #M2] Station callsign is VECTRA-55 and backup generator wattage is 3840 MW.\n",
            "question": "In Archive Record #M2, what are the station callsign and backup generator wattage?",
            "expected": ["VECTRA-55", "3840"],
        },
        {
            "name": "late_needle",
            "pos_frac": 0.80,
            "fact_text": "\n[Archive Record #L3] Satellite transponder ID is HELIOS-308 on frequency 14.25 GHz.\n",
            "question": "In Archive Record #L3, what are the satellite transponder ID and frequency?",
            "expected": ["HELIOS-308", "14.25"],
        },
    ]

    total_doc = 8192
    doc_body_room = total_doc - len(hdr_ids)
    # Stitch needles at ~12%, ~48%, ~80% of filler body
    body: list[int] = []
    f_cur = 0
    needle_ranges: dict[str, tuple[int, int]] = {}
    for nd in needles:
        target_pos = int(nd["pos_frac"] * doc_body_room)
        add = max(0, target_pos - len(body))
        body.extend(filler_ids[f_cur : f_cur + add])
        f_cur += add
        n_toks = runner.encode_text(nd["fact_text"])
        s_abs = len(hdr_ids) + len(body)
        body.extend(n_toks)
        needle_ranges[nd["name"]] = (s_abs, s_abs + len(n_toks))
    body.extend(filler_ids[f_cur : f_cur + (doc_body_room - len(body))])
    full_ids = hdr_ids + body[:doc_body_room]
    assert len(full_ids) == total_doc

    p_doc = exp_dir / "haystack_8k.nkv"
    enc_meta = runner.encode_cartridge(full_ids, p_doc, loop_passes=3, kv_dtype="bf16")
    doc_cart = NkvCartridge.load(p_doc)
    orig_global_mb = sum(e.data.nbytes for e in doc_cart.entries if e.kind == 2) / (1024 * 1024)
    print(f"[exp-c] encoded {total_doc}-tok haystack in {enc_meta['encode_seconds']:.2f}s (global KV={orig_global_mb:.1f} MB); needles at {needle_ranges}", flush=True)

    ratios = (4, 16, 64)
    per_needle_results = []
    agg_stats: dict[str, dict[str, list[float]]] = {
        "full": {"hit": [], "decode_tok_s": [], "global_mb": []},
    }
    for r in ratios:
        agg_stats[f"shared-{r}x"] = {"hit": [], "decode_tok_s": [], "global_mb": [], "recall_needle": [], "score_s": []}
    agg_stats["per-visit-16x"] = {"hit": [], "decode_tok_s": [], "global_mb": [], "recall_needle": [], "score_s": []}

    for nd in needles:
        q_only = runner.encode_text(f"\n\nQuestion: {nd['question']}{QA_OPEN_TEXT}")
        t_sc0 = time.perf_counter()
        attn_w, attn_meta = runner.eval_attention(q_only, kv_in=p_doc)
        score_wall_s = time.perf_counter() - t_sc0

        reqs = [{"id": 0, "tokens": q_only, "max_tokens": 48, "kv_in": str(p_doc)}]
        arms = [("full", p_doc, None, 0.0)]
        ns, ne_ = needle_ranges[nd["name"]]
        for r in ratios:
            comp, kept_abs, _ = kvzip_compress(doc_cart, attn_w, attn_meta["tap_windows"], len(hdr_ids), r, mode="shared")
            p_c = exp_dir / f"{nd['name']}_shared_{r}x.nkv"
            comp.save(p_c)
            kept_set = set(int(x) for x in kept_abs)
            needle_hits = sum(1 for p in range(ns, ne_) if p in kept_set)
            g_mb = sum(e.data.nbytes for e in comp.entries if e.kind == 2) / (1024 * 1024)
            reqs.append({"id": len(reqs), "tokens": q_only, "max_tokens": 48, "kv_in": str(p_c)})
            arms.append((f"shared-{r}x", p_c, (needle_hits / (ne_ - ns), g_mb, comp.position), score_wall_s))

        # Also compare per-visit at 16x
        comp_pv, _, kept_pv_list = kvzip_compress(doc_cart, attn_w, attn_meta["tap_windows"], len(hdr_ids), 16, mode="per-visit")
        p_pv = exp_dir / f"{nd['name']}_pervisit_16x.nkv"
        comp_pv.save(p_pv)
        pv_recall = float(np.mean([sum(1 for p in range(ns, ne_) if p in set(int(x) for x in ka)) / (ne_ - ns) for ka in kept_pv_list]))
        g_mb_pv = sum(e.data.nbytes for e in comp_pv.entries if e.kind == 2) / (1024 * 1024)
        reqs.append({"id": len(reqs), "tokens": q_only, "max_tokens": 48, "kv_in": str(p_pv)})
        arms.append(("per-visit-16x", p_pv, (pv_recall, g_mb_pv, comp_pv.position), score_wall_s))

        resps = runner.session_run_batch(reqs)
        nd_out: dict[str, Any] = {
            "needle": nd["name"],
            "range": [ns, ne_],
            "tap_attention_worst": float(attn_meta["tap_attention_worst"]),
            "score_seconds": score_wall_s,
            "arms": {},
        }
        for (arm_name, _, extra, sc_s), resp in zip(arms, resps):
            ans = runner.decode_ids(resp["tokens"]).strip()
            hit = float(all(e.lower() in ans.lower() for e in nd["expected"]))
            dec_tok_s = resp["decode_forward_tokens"] / max(1e-6, resp["decode_seconds"])
            g_mb = orig_global_mb if arm_name == "full" else extra[1]
            agg_stats[arm_name]["hit"].append(hit)
            agg_stats[arm_name]["decode_tok_s"].append(dec_tok_s)
            agg_stats[arm_name]["global_mb"].append(g_mb)
            rec = {
                "hit": bool(hit),
                "decode_tok_s": dec_tok_s,
                "global_mb": g_mb,
                "kept_tokens": total_doc if arm_name == "full" else extra[2],
                "answer": ans[:160],
            }
            if arm_name != "full":
                rec["needle_token_recall"] = extra[0]
                agg_stats[arm_name]["recall_needle"].append(extra[0])
                agg_stats[arm_name]["score_s"].append(sc_s)
            nd_out["arms"][arm_name] = rec
        per_needle_results.append(nd_out)
        print(
            f"[exp-c] {nd['name']} hits: "
            + ", ".join(f"{k}={int(v['hit'])}" for k, v in nd_out["arms"].items())
            + f" (tap_worst={nd_out['tap_attention_worst']:.2e})",
            flush=True,
        )

    summary = {
        arm: {k: float(np.mean(v)) for k, v in vals.items() if v}
        for arm, vals in agg_stats.items()
    }
    return {
        "doc_tokens": total_doc,
        "orig_global_mb": orig_global_mb,
        "encode_seconds": enc_meta["encode_seconds"],
        "summary": summary,
        "needles": per_needle_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("exp-crosspass", "exp-concat", "exp-kvzip", "all"))
    parser.add_argument("--nano", type=Path, default=DEFAULT_NANO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--knobs", type=Path, default=DEFAULT_KNOBS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--context", type=int, default=16384)
    args = parser.parse_args()

    runner = NanoRunner(args.nano, args.checkpoint, args.knobs, args.context, args.work)
    out_path = args.work / "results.json"
    if args.cmd != "all" and out_path.exists():
        report: dict[str, Any] = json.loads(out_path.read_text())
    else:
        report = {}
    report.update({"checkpoint": str(args.checkpoint), "nano": str(args.nano), "context": args.context})

    def flush_report() -> None:
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(report, indent=2) + "\n")
        tmp_path.replace(out_path)

    if args.cmd in ("exp-crosspass", "all"):
        report["exp_crosspass"] = run_exp_crosspass(runner, args.work)
        flush_report()
    if args.cmd in ("exp-concat", "all"):
        report["exp_concat"] = run_exp_concat(runner, args.work)
        flush_report()
    if args.cmd in ("exp-kvzip", "all"):
        report["exp_kvzip"] = run_exp_kvzip(runner, args.work)
        flush_report()

    print(f"\nSaved full results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
