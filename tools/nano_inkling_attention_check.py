#!/usr/bin/env python3
"""All-vocabulary logits of the tuned Inkling 2 Mini path under its relative
attention knobs, on one token sequence (a JSON array of ids; a real transcript
says more than seeded ids).

    python3 tools/nano_inkling_attention_check.py zig-out/bin/nano CHECKPOINT ids.json OUT_DIR

Run it under the inference lock (tools/guarded_run.py). NumPy is required. Each arm is one `nano eval` over the same ids,
with logits kept at 24 positions. Arms:

    reference-1  tuned knobs without relative_split, feed 1 (mk_relative_attention)
    split-1      tuned knobs (relative_split, relative_split_4), feed 1
    stripes-1    tuned knobs plus relative_stripes, feed 1
    tiled-128    tuned knobs, feed 128 (mk_relative_attention_prefill)
    stripes-128  tuned knobs plus relative_stripes, feed 128

Feed 1 sends every row through the single-row (decode) kernels, feed 128
through the batched (prefill) ones. For each pair it reports the worst and RMS
absolute difference, top-1 agreement and the first position that differs; and
for each arm its smallest and median top-1 margin.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

TUNED = ("batch=128 expert_simd=true expert_wide_dot=true expert_gate_pair=true expert_prefill=true "
         "expert_wide_k=true expert_bf16_tiles=true expert_mlx_scale=true bf16_linear_tile=true "
         "bf16_prefill_tile=8 bf16_prefill_matrix=true bf16_decode_vec=true fuse_bf16_cast=true "
         "concurrent_scratch=true relative_tile_prefill=true").split()
SPLIT = ["relative_split=true", "relative_split_4=true"]
ARMS = {
    "reference-1": (1, []),
    "split-1": (1, SPLIT),
    "stripes-1": (1, SPLIT + ["relative_stripes=true"]),
    "tiled-128": (128, SPLIT),
    "stripes-128": (128, SPLIT + ["relative_stripes=true"]),
}
PAIRS = [("reference-1", "split-1"), ("reference-1", "stripes-1"), ("split-1", "stripes-1"),
         ("split-1", "tiled-128"), ("tiled-128", "stripes-128"), ("stripes-1", "stripes-128")]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("nano", type=Path)
p.add_argument("checkpoint", type=Path)
p.add_argument("ids", type=Path)
p.add_argument("out", type=Path)
args = p.parse_args()

ids = json.loads(args.ids.read_text())
if len(ids) < 24:
    p.error("at least 24 token ids are needed")
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "tokens.json").write_text(json.dumps(ids))
positions = sorted({round(i * (len(ids) - 1) / 23) for i in range(24)})
runs = {}
result = {"tokens": len(ids), "positions": positions, "arms": {}, "pairs": {}}
for name, (feed, extra) in ARMS.items():
    logits = args.out / f"{name}.f32"
    command = [str(args.nano), "eval", str(args.checkpoint), "--context", str(len(ids) + 8),
               "--tokens-file", str(args.out / "tokens.json"), "--feed", str(feed),
               "--positions", ",".join(map(str, positions)), "--logits-out", str(logits)]
    for knob in (*TUNED, *extra):
        command += ["--knob", knob]
    summary = json.loads(subprocess.check_output(command, text=True))
    values = np.fromfile(logits, dtype=np.float32).reshape(len(positions), summary["vocab"])
    logits.unlink()
    top = np.sort(values, axis=1)
    margin = top[:, -1] - top[:, -2]
    runs[name] = (values, summary["top1"])
    result["arms"][name] = {"smallest_top1_margin": float(margin.min()), "median_top1_margin": float(np.median(margin))}
for a, b in PAIRS:
    x, y = runs[a][0], runs[b][0]
    worst = np.abs(x - y).max(axis=1)
    differs = [positions[i] for i in range(len(positions)) if worst[i] > 0]
    result["pairs"][f"{a} vs {b}"] = {
        "max_abs_diff": float(worst.max()),
        "rms_diff": float(np.sqrt(np.mean((x - y) ** 2))),
        "top1_agree": int(sum(u == v for u, v in zip(runs[a][1], runs[b][1]))),
        "first_position_that_differs": differs[0] if differs else None,
        "max_abs_diff_at_first_three_positions": [float(v) for v in worst[:3]],
    }
print(json.dumps(result, indent=1))
