#!/usr/bin/env python3
"""Synthetic verification of Qwen3.5 / Qwen3.8 (`qwen3_5`) in `nano`.

Builds a 4-layer affine-INT4 checkpoint (`3 linear_attention + 1 full_attention`,
G=6, partial RoPE 0.25, output gate on attention, untied lm_head) in a temporary
directory and verifies:
  1. NumPy float64 whole-sequence vs 1-by-1 recurrent step agree to < 1e-9.
  2. `nano eval` feed=1 vs feed=4 are bit-identical across logits, traces, greedy.
  3. `nano eval` (reference and presets/qwen38.json) matches NumPy f64 within 1e-3,
     with 100% top-1 and greedy continuation match.
  4. `mlx_lm.utils.load_model` matches logits/top-1/greedy on the same weights.
  5. `nano session` turn1 + turn2 equals single-shot prefill, and `reset: true`
     zeroes recurrent/conv/KV state bit-for-bit.
"""
import argparse
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

PROMPT = [1, 17, 93, 4, 200, 8, 61, 130, 7, 222, 45, 3]
GREEDY_N = 16


def encode_bf16(values: np.ndarray):
    bits = values.astype(np.float32).view(np.uint32)
    bf = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    decoded = (bf.astype(np.uint32) << 16).view(np.float32)
    return decoded, bf.tobytes()


def quantize_affine4(values: np.ndarray, group_size: int):
    rows, cols = values.shape
    assert cols % group_size == 0 and (cols * 4) % 32 == 0
    g = values.reshape(rows, cols // group_size, group_size)
    lo, hi = g.min(-1), g.max(-1)
    scale = np.maximum((hi - lo) / 15.0, np.float32(1e-6))
    scale_v, scale_b = encode_bf16(scale)
    bias_v, bias_b = encode_bf16(lo)
    codes = np.clip(np.rint((g - bias_v[..., None]) / scale_v[..., None]), 0, 15).astype(np.uint8)
    packed = np.ascontiguousarray(codes.reshape(rows, cols))
    packed = packed[:, 0::2] | (packed[:, 1::2] << 4)
    words = np.ascontiguousarray(packed).view("<u4")
    return {
        "weight": ("U32", list(words.shape), words.tobytes()),
        "scales": ("BF16", list(scale.shape), scale_b),
        "biases": ("BF16", list(lo.shape), bias_b),
    }


def build_synthetic_checkpoint(directory: pathlib.Path, seed: int = 3801):
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "eos_token_id": [255, 254],
        "tie_word_embeddings": False,
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "quantization_config": {"group_size": 32, "bits": 4, "mode": "affine"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "full_attention_interval": 4,
            "num_attention_heads": 6,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "attn_output_gate": True,
            "attention_bias": False,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_value_head_dim": 32,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "vocab_size": 256,
            "max_position_embeddings": 512,
        },
    }
    tc = cfg["text_config"]
    h = tc["hidden_size"]
    inter = tc["intermediate_size"]
    heads = tc["num_attention_heads"]
    kv_heads = tc["num_key_value_heads"]
    head_dim = tc["head_dim"]
    hk = tc["linear_num_key_heads"]
    hv = tc["linear_num_value_heads"]
    dk = tc["linear_key_head_dim"]
    dv = tc["linear_value_head_dim"]
    conv_k = tc["linear_conv_kernel_dim"]
    key_dim = hk * dk
    val_dim = hv * dv
    conv_dim = 2 * key_dim + val_dim
    vocab = tc["vocab_size"]
    group_size = cfg["quantization"]["group_size"]

    rng = np.random.default_rng(seed)
    raw = {}

    def add(name, shape, kind="matrix"):
        if kind == "norm":
            raw[name] = rng.normal(1.0, 0.12, shape).astype(np.float32)
        elif kind == "conv":
            raw[name] = rng.normal(0.0, 0.35, shape).astype(np.float32)
        elif kind == "alog":
            raw[name] = rng.uniform(-1.5, 0.5, shape).astype(np.float32)
        elif kind == "dtbias":
            raw[name] = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        else:
            raw[name] = rng.normal(0.0, 1.0 / math.sqrt(shape[-1]), shape).astype(np.float32)

    add("language_model.model.embed_tokens.weight", (vocab, h))
    raw["language_model.model.embed_tokens.weight"] *= np.float32(math.sqrt(h) * 0.5)
    add("language_model.lm_head.weight", (vocab, h))
    add("language_model.model.norm.weight", (h,), "norm")

    for i, lt in enumerate(tc["layer_types"]):
        p = f"language_model.model.layers.{i}."
        add(p + "input_layernorm.weight", (h,), "norm")
        add(p + "post_attention_layernorm.weight", (h,), "norm")
        add(p + "mlp.gate_proj.weight", (inter, h))
        add(p + "mlp.up_proj.weight", (inter, h))
        add(p + "mlp.down_proj.weight", (h, inter))
        if lt == "linear_attention":
            add(p + "linear_attn.in_proj_qkv.weight", (conv_dim, h))
            add(p + "linear_attn.in_proj_z.weight", (val_dim, h))
            add(p + "linear_attn.in_proj_b.weight", (hv, h))
            add(p + "linear_attn.in_proj_a.weight", (hv, h))
            add(p + "linear_attn.out_proj.weight", (h, val_dim))
            add(p + "linear_attn.conv1d.weight", (conv_dim, conv_k, 1), "conv")
            add(p + "linear_attn.A_log", (hv,), "alog")
            add(p + "linear_attn.dt_bias", (hv,), "dtbias")
            add(p + "linear_attn.norm.weight", (dv,), "norm")
        else:
            add(p + "self_attn.q_proj.weight", (heads * head_dim * 2, h))
            add(p + "self_attn.k_proj.weight", (kv_heads * head_dim, h))
            add(p + "self_attn.v_proj.weight", (kv_heads * head_dim, h))
            add(p + "self_attn.o_proj.weight", (h, heads * head_dim))
            add(p + "self_attn.q_norm.weight", (head_dim,), "norm")
            add(p + "self_attn.k_norm.weight", (head_dim,), "norm")

    tensors = {}
    for name, values in raw.items():
        if values.ndim == 2:
            base = name[:-len(".weight")]
            for suffix, entry in quantize_affine4(values, group_size).items():
                tensors[base + "." + suffix] = entry
        else:
            _, payload = encode_bf16(values)
            tensors[name] = ("BF16", list(values.shape), payload)

    header, data = {"__metadata__": {"format": "mlx"}}, bytearray()
    for name in sorted(tensors):
        tdtype, shape, payload = tensors[name]
        header[name] = {"dtype": tdtype, "shape": shape, "data_offsets": [len(data), len(data) + len(payload)]}
        data.extend(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))
    (directory / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return cfg


def load_decoded_weights(directory: pathlib.Path):
    raw = (directory / "model.safetensors").read_bytes()
    hdr_len = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + hdr_len])
    blob = raw[8 + hdr_len :]
    group = json.loads((directory / "config.json").read_text())["quantization"]["group_size"]

    def read_bf16(name):
        info = hdr[name]
        s0, s1 = info["data_offsets"]
        u16 = np.frombuffer(blob[s0:s1], dtype="<u2").reshape(info["shape"])
        return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    def read_mat(prefix):
        w_info = hdr[prefix + ".weight"]
        if w_info["dtype"] != "U32":
            return read_bf16(prefix + ".weight")
        s0, s1 = w_info["data_offsets"]
        words = np.frombuffer(blob[s0:s1], dtype="<u4").reshape(w_info["shape"])
        packed = words.view(np.uint8).reshape(w_info["shape"][0], w_info["shape"][1] * 4)
        codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float64)
        codes[:, 0::2] = packed & 0x0F
        codes[:, 1::2] = packed >> 4
        scales = read_bf16(prefix + ".scales")
        biases = read_bf16(prefix + ".biases")
        r, c = codes.shape
        return (codes.reshape(r, c // group, group) * scales[..., None] + biases[..., None]).reshape(r, c)

    return hdr, read_bf16, read_mat


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def rmsnorm(x, w, eps):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv * w


def apply_partial_rope(x, pos_offset, rot, theta):
    # x: [T, heads, dim]
    t, heads, _ = x.shape
    half = rot // 2
    inv_freq = 1.0 / (theta ** (2.0 * np.arange(half, dtype=np.float64) / rot))
    pos = (np.arange(t, dtype=np.float64) + pos_offset)[:, None, None]
    ang = pos * inv_freq[None, None, :]
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy()
    x1, x2 = x[..., :half], x[..., half:rot]
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rot] = x2 * c + x1 * s
    return out


class NumpyQwen35:
    def __init__(self, directory: pathlib.Path):
        cfg = json.loads((directory / "config.json").read_text())
        tc = cfg["text_config"]
        self.tc = tc
        _, bf16, mat = load_decoded_weights(directory)
        self.embed = mat("language_model.model.embed_tokens")
        self.lm_head = mat("language_model.lm_head")
        self.final_norm = bf16("language_model.model.norm.weight")
        self.layers = []
        for i, lt in enumerate(tc["layer_types"]):
            p = f"language_model.model.layers.{i}."
            d = {
                "type": lt,
                "in_norm": bf16(p + "input_layernorm.weight"),
                "post_norm": bf16(p + "post_attention_layernorm.weight"),
                "gate_proj": mat(p + "mlp.gate_proj"),
                "up_proj": mat(p + "mlp.up_proj"),
                "down_proj": mat(p + "mlp.down_proj"),
            }
            if lt == "linear_attention":
                d.update({
                    "in_proj_qkv": mat(p + "linear_attn.in_proj_qkv"),
                    "in_proj_z": mat(p + "linear_attn.in_proj_z"),
                    "in_proj_b": mat(p + "linear_attn.in_proj_b"),
                    "in_proj_a": mat(p + "linear_attn.in_proj_a"),
                    "out_proj": mat(p + "linear_attn.out_proj"),
                    "conv1d": bf16(p + "linear_attn.conv1d.weight")[:, :, 0],
                    "A_log": bf16(p + "linear_attn.A_log"),
                    "dt_bias": bf16(p + "linear_attn.dt_bias"),
                    "norm": bf16(p + "linear_attn.norm.weight"),
                })
            else:
                d.update({
                    "q_proj": mat(p + "self_attn.q_proj"),
                    "k_proj": mat(p + "self_attn.k_proj"),
                    "v_proj": mat(p + "self_attn.v_proj"),
                    "o_proj": mat(p + "self_attn.o_proj"),
                    "q_norm": bf16(p + "self_attn.q_norm.weight"),
                    "k_norm": bf16(p + "self_attn.k_norm.weight"),
                })
            self.layers.append(d)

    def new_state(self):
        tc = self.tc
        st = []
        conv_dim = 2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"] + tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
        for lt in tc["layer_types"]:
            if lt == "linear_attention":
                st.append({
                    "conv": np.zeros((conv_dim, 0), dtype=np.float64),
                    "rec": np.zeros((tc["linear_num_value_heads"], tc["linear_value_head_dim"], tc["linear_key_head_dim"]), dtype=np.float64),
                })
            else:
                st.append({
                    "k": np.zeros((0, tc["num_key_value_heads"], tc["head_dim"]), dtype=np.float64),
                    "v": np.zeros((0, tc["num_key_value_heads"], tc["head_dim"]), dtype=np.float64),
                })
        return {"pos": 0, "layers": st}

    def forward(self, tokens, state):
        tc = self.tc
        eps = tc["rms_norm_eps"]
        t = len(tokens)
        x = self.embed[tokens].copy()
        traces = [x.copy()]
        hk, hv, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        key_dim, val_dim = hk * dk, hv * dv
        taps = tc["linear_conv_kernel_dim"]
        heads, kvh, dim = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        rot = round(dim * tc["rope_parameters"]["partial_rotary_factor"])
        theta = tc["rope_parameters"]["rope_theta"]

        for l, s in zip(self.layers, state["layers"]):
            xn = rmsnorm(x, l["in_norm"], eps)
            if l["type"] == "linear_attention":
                qkv = xn @ l["in_proj_qkv"].T
                z = (xn @ l["in_proj_z"].T).reshape(t, hv, dv)
                b = xn @ l["in_proj_b"].T
                a = xn @ l["in_proj_a"].T

                # Causal depthwise conv
                hist = s["conv"]  # [C, <= taps-1]
                seq_c = qkv.T  # [C, t]
                buf = np.concatenate([np.zeros((seq_c.shape[0], max(0, (taps - 1) - hist.shape[1])), dtype=np.float64), hist, seq_c], axis=1)
                conv_out = np.zeros_like(seq_c)
                w_conv = l["conv1d"]
                for k in range(taps):
                    conv_out += buf[:, buf.shape[1] - t - (taps - 1) + k : buf.shape[1] - (taps - 1) + k] * w_conv[:, k : k + 1]
                s["conv"] = buf[:, -(taps - 1) :]
                act = silu(conv_out.T)
                q = act[:, :key_dim].reshape(t, hk, dk)
                k = act[:, key_dim : 2 * key_dim].reshape(t, hk, dk)
                v = act[:, 2 * key_dim :].reshape(t, hv, dv)

                q = q / (np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * math.sqrt(dk))
                k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)

                sp_in = a + l["dt_bias"][None, :]
                sp = np.where(sp_in > 20.0, sp_in, np.log1p(np.exp(sp_in)))
                g = np.exp(-np.exp(l["A_log"])[None, :] * sp)
                beta = sigmoid(b)

                rec = s["rec"]
                ys = np.empty((t, hv, dv), dtype=np.float64)
                rep = hv // hk
                for r in range(t):
                    qr = np.repeat(q[r], rep, axis=0)
                    kr = np.repeat(k[r], rep, axis=0)
                    rec = rec * g[r][:, None, None]
                    kv_mem = np.einsum("hdk,hk->hd", rec, kr)
                    delta = (v[r] - kv_mem) * beta[r][:, None]
                    rec = rec + delta[:, :, None] * kr[:, None, :]
                    ys[r] = np.einsum("hdk,hk->hd", rec, qr)
                s["rec"] = rec
                normed = rmsnorm(ys, l["norm"], eps) * silu(z)
                branch = normed.reshape(t, val_dim) @ l["out_proj"].T
            else:
                qg = (xn @ l["q_proj"].T).reshape(t, heads, 2 * dim)
                q = qg[..., :dim]
                gate = qg[..., dim:].reshape(t, heads * dim)
                k = (xn @ l["k_proj"].T).reshape(t, kvh, dim)
                v = (xn @ l["v_proj"].T).reshape(t, kvh, dim)
                q = rmsnorm(q, l["q_norm"], eps)
                k = rmsnorm(k, l["k_norm"], eps)
                q = apply_partial_rope(q, state["pos"], rot, theta)
                k = apply_partial_rope(k, state["pos"], rot, theta)
                s["k"] = np.concatenate([s["k"], k], axis=0)
                s["v"] = np.concatenate([s["v"], v], axis=0)
                kt, vt = s["k"], s["v"]
                rep = heads // kvh
                kt_rep = np.repeat(kt, rep, axis=1)
                vt_rep = np.repeat(vt, rep, axis=1)
                scale = 1.0 / math.sqrt(dim)
                total_k = kt.shape[0]
                att = np.empty((t, heads, dim), dtype=np.float64)
                for r in range(t):
                    vis = state["pos"] + r + 1
                    scores = np.einsum("hd,khd->hk", q[r], kt_rep[:vis]) * scale
                    mx = np.max(scores, axis=-1, keepdims=True)
                    w = np.exp(scores - mx)
                    w /= np.sum(w, axis=-1, keepdims=True)
                    att[r] = np.einsum("hk,khd->hd", w, vt_rep[:vis])
                gated = att.reshape(t, heads * dim) * sigmoid(gate)
                branch = gated @ l["o_proj"].T

            x = x + branch
            mn = rmsnorm(x, l["post_norm"], eps)
            glu = silu(mn @ l["gate_proj"].T) * (mn @ l["up_proj"].T)
            x = x + glu @ l["down_proj"].T
            traces.append(x.copy())

        state["pos"] += t
        logits = rmsnorm(x, self.final_norm, eps) @ self.lm_head.T
        return np.stack(traces, axis=0), logits


def run_eval(binary, ckpt, workdir, tokens, positions, feed, knobs=None, greedy=GREEDY_N):
    tok_file = workdir / "tokens.json"
    pos_file = workdir / "positions.json"
    tok_file.write_text(json.dumps({"tokens": tokens}))
    pos_file.write_text(json.dumps({"positions": positions}))
    logits_p = workdir / f"logits_{feed}_{int(bool(knobs))}.f32"
    hidden_p = workdir / f"hidden_{feed}_{int(bool(knobs))}.f32"
    cmd = [
        str(binary), "eval", str(ckpt),
        "--tokens-file", str(tok_file),
        "--positions-file", str(pos_file),
        "--logits-out", str(logits_p),
        "--hidden-out", str(hidden_p),
        "--greedy", str(greedy),
        "--feed", str(feed),
        "--context", str(len(tokens) + greedy + 16),
    ]
    if knobs:
        cmd += ["--knobs", str(knobs)]
    else:
        cmd += ["--knob", f"batch={max(feed, 1)}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    logits = np.fromfile(logits_p, dtype="<f4").reshape(len(positions), summary["vocab"])
    hidden = np.fromfile(hidden_p, dtype="<f4").reshape(summary["traces"], summary["tokens"], summary["trace_width"])
    return summary, logits, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=pathlib.Path)
    ap.add_argument("--knobs", type=pathlib.Path, default=pathlib.Path("src/mega/presets/qwen38.json"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="qwen35-verify-") as tmp:
        tmp_p = pathlib.Path(tmp)
        ckpt = tmp_p / "ckpt"
        build_synthetic_checkpoint(ckpt)

        model = NumpyQwen35(ckpt)
        st_seq = model.new_state()
        ref_hidden, ref_logits = model.forward(PROMPT, st_seq)
        st_step = model.new_state()
        step_traces, step_logits = [], []
        for tok in PROMPT:
            tr, lg = model.forward([tok], st_step)
            step_traces.append(tr[:, 0, :])
            step_logits.append(lg[0])
        step_hidden = np.stack(step_traces, axis=1)
        step_logits = np.stack(step_logits, axis=0)
        self_diff = max(float(np.max(np.abs(step_hidden - ref_hidden))), float(np.max(np.abs(step_logits - ref_logits))))
        print(f"NumPy f64 seq vs step max_abs: {self_diff:.3e}")
        assert self_diff < 1e-9, self_diff

        ref_greedy = []
        cur_tok = int(np.argmax(ref_logits[-1]))
        for _ in range(GREEDY_N):
            ref_greedy.append(cur_tok)
            _, lg = model.forward([cur_tok], st_seq)
            cur_tok = int(np.argmax(lg[0]))

        positions = list(range(len(PROMPT)))
        sum1, lg1, hd1 = run_eval(args.binary, ckpt, tmp_p, PROMPT, positions, feed=1)
        sum4, lg4, hd4 = run_eval(args.binary, ckpt, tmp_p, PROMPT, positions, feed=4)
        assert np.array_equal(lg1.view(np.uint32), lg4.view(np.uint32)), "feed=1 vs feed=4 logits differ!"
        assert np.array_equal(hd1.view(np.uint32), hd4.view(np.uint32)), "feed=1 vs feed=4 hidden differ!"
        assert sum1["greedy"] == sum4["greedy"], "feed=1 vs feed=4 greedy differ!"

        lg_err = float(np.max(np.abs(lg1.astype(np.float64) - ref_logits)))
        hd_err = float(np.max(np.abs(hd1.astype(np.float64) - ref_hidden)))
        top1_match = sum(int(a == b) for a, b in zip(sum1["top1"], ref_logits.argmax(1)))
        print(f"nano ref feed=1/4 bitwise=True logits_err={lg_err:.3e} hidden_err={hd_err:.3e} top1={top1_match}/{len(PROMPT)} greedy_eq={sum1['greedy'] == ref_greedy}")
        assert lg_err < 1e-3 and hd_err < 1e-3
        assert top1_match == len(PROMPT)
        assert sum1["greedy"] == ref_greedy

        sum_k, lg_k, hd_k = run_eval(args.binary, ckpt, tmp_p, PROMPT, positions, feed=4, knobs=args.knobs)
        lg_k_err = float(np.max(np.abs(lg_k.astype(np.float64) - ref_logits)))
        hd_k_err = float(np.max(np.abs(hd_k.astype(np.float64) - ref_hidden)))
        print(f"nano {args.knobs.name} feed=4 logits_err={lg_k_err:.3e} hidden_err={hd_k_err:.3e} greedy_eq={sum_k['greedy'] == ref_greedy}")
        assert lg_k_err < 1e-3 and hd_k_err < 1e-3
        assert sum_k["top1"] == sum1["top1"]
        assert sum_k["greedy"] == ref_greedy

        # Compare against mlx_lm.utils.load_model (both native BF16 and float32 activations)
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx_lm.models.cache import make_prompt_cache

        mlx_bf16, _ = load_model(ckpt)
        bf16_lg = np.array(mlx_bf16(mx.array([PROMPT]), cache=make_prompt_cache(mlx_bf16))[0].astype(mx.float32))
        bf16_err = float(np.max(np.abs(bf16_lg - ref_logits)))
        bf16_top1 = sum(int(a == b) for a, b in zip(bf16_lg.argmax(1), ref_logits.argmax(1)))
        print(f"mlx_lm native bf16 vs NumPy f64: logits_max_abs={bf16_err:.3e} top1={bf16_top1}/{len(PROMPT)}")

        def cast_f32(tree):
            if isinstance(tree, dict):
                return {k: cast_f32(val) for k, val in tree.items()}
            if isinstance(tree, list):
                return [cast_f32(val) for val in tree]
            if isinstance(tree, mx.array) and tree.dtype == mx.bfloat16:
                return tree.astype(mx.float32)
            return tree

        mlx_f32, _ = load_model(ckpt)
        mlx_f32.update(cast_f32(mlx_f32.parameters()))
        mlx_cache = make_prompt_cache(mlx_f32)
        mlx_lg = np.array(mlx_f32(mx.array([PROMPT]), cache=mlx_cache)[0].astype(mx.float32))
        mlx_err = float(np.max(np.abs(mlx_lg - ref_logits)))
        mlx_top1 = sum(int(a == b) for a, b in zip(mlx_lg.argmax(1), ref_logits.argmax(1)))
        mlx_greedy = []
        tok = int(mlx_lg[-1].argmax())
        for _ in range(GREEDY_N):
            mlx_greedy.append(tok)
            step_lg = np.array(mlx_f32(mx.array([[tok]]), cache=mlx_cache)[0, -1].astype(mx.float32))
            tok = int(step_lg.argmax())
        print(f"mlx_lm f32 vs NumPy f64: logits_max_abs={mlx_err:.3e} top1={mlx_top1}/{len(PROMPT)} greedy_eq={mlx_greedy == ref_greedy}")
        assert mlx_err < 1e-4
        assert mlx_top1 == len(PROMPT)
        assert mlx_greedy == ref_greedy

        # Verify nano session multi-turn continuation and reset() zeroing recurrent states
        proc = subprocess.Popen(
            [str(args.binary), "session", str(ckpt), "--context", "128"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin and proc.stdout
        reqs = [
            {"id": 1, "tokens": PROMPT[:7], "max_tokens": 0, "return_logits": True},
            {"id": 2, "tokens": PROMPT[7:], "max_tokens": GREEDY_N, "return_logits": True},
            {"id": 3, "tokens": PROMPT[:7], "max_tokens": 0, "reset": True, "return_logits": True},
        ]
        resps = []
        for req in reqs:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            resps.append(json.loads(proc.stdout.readline()))
        proc.stdin.close()
        rc = proc.wait(timeout=30)
        assert rc == 0, proc.stderr.read()

        t1_lg = np.array(resps[0]["final_logits"], dtype=np.float32)
        t2_prefill_lg = np.array(resps[1]["prefill_logits"], dtype=np.float32)
        t3_lg = np.array(resps[2]["final_logits"], dtype=np.float32)
        assert np.array_equal(t2_prefill_lg.view(np.uint32), lg1[-1].view(np.uint32)), "session turn1+turn2 logits differ from eval!"
        assert resps[1]["tokens"] == sum1["greedy"], "session turn2 greedy differs from eval!"
        assert np.array_equal(t3_lg.view(np.uint32), t1_lg.view(np.uint32)), "session reset did not restore recurrent/conv/KV state bit-for-bit!"
        print("nano session turn1+turn2 & reset bitwise parity: PASS")

    print("ALL QWEN35 SYNTHETIC VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
