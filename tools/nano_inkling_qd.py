#!/usr/bin/env python3
"""Inkling 2 Mini MAP-Elites QD grid runner and renderer.

Mirrors tools/nano_qd.py / benchmarks/lfm3-qd/ for Inkling 2 Mini across
context lengths (4K, 16K, 32K) using counterbalanced ABBA round interleaving
under tools/guarded_run.py.

Usage:
    python3 tools/nano_inkling_qd.py run zig-out/bin/nano CHECKPOINT \
        --lock LOCK --archive benchmarks/inkling-qd \
        --candidates src/mega/candidates/inkling/*.json \
        --lengths 4096,16384,32768 --rounds 2 --repeats 1 --decode-tokens 64 \
        --parity-tokens .zig-cache/final/cake_toks.json
    python3 tools/nano_inkling_qd.py render --archive benchmarks/inkling-qd
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_ARCHIVE = REPO / "benchmarks/inkling-qd"
DEFAULT_LENGTHS = (4096, 16384, 32768)
CONTEXT_MAP = {4096: 4200, 16384: 16512, 32768: 32900}
FAMILY_ORDER = (
    "long-context",
    "megakernel-v1",
    "qwegn-lut",
    "qwegn-moe",
    "qwegn-inline",
    "qwegn-qkvr",
    "qwegn-full",
)
PUBLISHED_BASELINES = [
    ("mlx-lm#2086 (80c M3U, raw)", 4096, 1516.3, 32.07),
    ("mlx-lm#2086 (80c M3U, 60c-scaled)", 4096, 1137.2, 32.07),
    ("mlx-lm#2086 (80c M3U, raw)", 16384, 1472.3, 32.50),
    ("mlx-lm#2086 (80c M3U, 60c-scaled)", 16384, 1104.2, 32.50),
]


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(length):
    if length <= 4096:
        return "<=4096"
    if length <= 16384:
        return "<=16384"
    return ">16384"


def bucket_sort_key(bkt):
    order = {"<=4096": 0, "<=16384": 1, ">16384": 2}
    return order.get(bkt, 99)


def dispatch_bin(objective, disp):
    if objective == "decode":
        if disp <= 1300:
            return "<=1300"
        if disp <= 1700:
            return "<=1700"
        return ">1700"
    if disp <= 1900:
        return "<=1900"
    if disp <= 2200:
        return "<=2200"
    if disp <= 2500:
        return "<=2500"
    return ">2500"


def guarded_cmd(argv, lock_path, timeout=240):
    if lock_path:
        return [
            sys.executable,
            str(HERE / "guarded_run.py"),
            "--lock",
            str(lock_path),
            "--timeout",
            str(timeout),
            "--",
        ] + argv
    return argv


class Archive:
    def __init__(self, directory):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.dir / "archive.lock"
        self.archive_path = self.dir / "archive.json"
        self.trials_path = self.dir / "trials.jsonl"
        self.elites_path = self.dir / "elites.md"
        self.lock_file = None
        self.data = None

    def __enter__(self):
        self.lock_file = open(self.lock_path, "a+")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        if self.archive_path.exists():
            self.data = json.loads(self.archive_path.read_text())
        else:
            self.data = {"version": 1, "model": "Inkling-2-Mini", "cells": {}}
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                tmp = self.archive_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data, indent=1) + "\n")
                os.replace(tmp, self.archive_path)
                self.elites_path.write_text(render_markdown(self.data))
        finally:
            if self.lock_file:
                self.lock_file.close()

    def append_trial(self, record):
        with open(self.trials_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def run_eval_cake(binary, checkpoint, knobs_path, tokens_path, lock_path, out_f32):
    tokens = json.loads(pathlib.Path(tokens_path).read_text())
    n_tok = len(tokens)
    ctx = max(40, n_tok + 32)
    cmd = [
        str(binary),
        "eval",
        str(checkpoint),
        "--knobs",
        str(knobs_path),
        "--tokens-file",
        str(tokens_path),
        "--context",
        str(ctx),
        "--feed",
        str(min(8, n_tok)),
        "--positions",
        "all",
        "--logits-out",
        str(out_f32),
        "--greedy",
        "16",
    ]
    proc = subprocess.run(guarded_cmd(cmd, lock_path, timeout=240), capture_output=True, text=True, check=True)
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    logits = np.fromfile(out_f32, dtype=np.float32).reshape(n_tok, summary["vocab"])
    return summary, logits


def run_bench_once(binary, checkpoint, prompt_len, knobs_path, decode_tokens, repeats, lock_path):
    ctx = CONTEXT_MAP.get(prompt_len, prompt_len + decode_tokens + 128)
    cmd = [
        str(binary),
        "bench",
        str(checkpoint),
        "--knobs",
        str(knobs_path),
        "--context",
        str(ctx),
        "--prompt-tokens",
        str(prompt_len),
        "--decode-tokens",
        str(decode_tokens),
        "--repeats",
        str(repeats),
        "--decode-detail",
        "1",
    ]
    proc = subprocess.run(guarded_cmd(cmd, lock_path, timeout=240), capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    detail = None
    for line in proc.stderr.splitlines():
        if line.startswith("{"):
            try:
                detail = json.loads(line)
            except json.JSONDecodeError:
                pass
    if detail:
        res["decode_detail"] = detail
    return res


def family_rank(fam):
    if fam in FAMILY_ORDER:
        return (0, FAMILY_ORDER.index(fam), fam)
    return (1, 0, fam)


def render_markdown(data):
    cells = data.get("cells", {})
    lines = [
        "# Inkling 2 Mini MAP-Elites megakernel archive",
        "",
        "Generated by `tools/nano_inkling_qd.py` from `archive.json`; do not edit.",
        "Measured on `thinkingmachines--Inkling-2-Mini` using ABBA counterbalanced rounds under `tools/guarded_run.py`.",
        "",
    ]
    buckets = sorted({c["bucket"] for c in cells.values()}, key=bucket_sort_key)
    families = sorted({c["family"] for c in cells.values()}, key=family_rank)

    # Index cells by (objective, family, bucket)
    by_ofb = {}
    for cell in cells.values():
        key = (cell["objective"], cell["family"], cell["bucket"])
        prev = by_ofb.get(key)
        if prev is None or cell["tokens_per_second"] > prev["tokens_per_second"]:
            by_ofb[key] = cell

    for objective, disp_label, title in (
        ("prefill", "disp/fwd", "2D MAP-Elites Grid: Prefill Throughput (tok/s)"),
        ("decode", "disp/tok", "2D MAP-Elites Grid: Decode Throughput (tok/s)"),
    ):
        best_per_bkt = {}
        for bkt in buckets:
            vals = [by_ofb[(objective, fam, bkt)]["tokens_per_second"] for fam in families if (objective, fam, bkt) in by_ofb]
            if vals:
                best_per_bkt[bkt] = max(vals)

        lines += [
            "## %s" % title,
            "",
            "| family | bin | %s | cake parity | %s |" % (
                disp_label,
                " | ".join("%s" % bkt for bkt in buckets),
            ),
            "| --- | --- | ---: | --- | %s |" % " | ".join("---" for _ in buckets),
        ]
        for fam in families:
            fam_cells = [by_ofb[(objective, fam, bkt)] for bkt in buckets if (objective, fam, bkt) in by_ofb]
            if not fam_cells:
                continue
            sample = fam_cells[0]
            dbin = sample["dispatch_bin"]
            dval = "%.1f" % sample["dispatches_per_token"]
            parity = sample.get("parity") or {}
            if parity:
                pstr = "max=%.2g, %d/%d, greedy=%s" % (
                    parity["max_abs_diff"],
                    parity["top1_agree"],
                    parity["positions"],
                    "eq" if parity["greedy_equal"] else "diff",
                )
            else:
                pstr = sample.get("admission", "-")

            row_cols = []
            for bkt in buckets:
                c = by_ofb.get((objective, fam, bkt))
                if not c:
                    row_cols.append("-")
                    continue
                val = c["tokens_per_second"]
                rnds = " / ".join("%.1f" % x for x in c["rounds"])
                ref_c = by_ofb.get((objective, "megakernel-v1", bkt))
                if ref_c and fam != "megakernel-v1":
                    delta_s = ", %+.1f%%" % (100.0 * (val - ref_c["tokens_per_second"]) / ref_c["tokens_per_second"])
                else:
                    delta_s = ""
                txt = "%.2f (%s%s)" % (val, rnds, delta_s)
                if bkt in best_per_bkt and abs(val - best_per_bkt[bkt]) < 1e-9:
                    txt = "**%s**" % txt
                row_cols.append(txt)

            lines.append("| `%s` | `%s` | %s | %s | %s |" % (fam, dbin, dval, pstr, " | ".join(row_cols)))
        lines.append("")

    lines += [
        "## Occupied MAP-Elites cells",
        "",
        "| objective | bucket | dispatch_bin | family | candidate | tokens/s | rounds | disp | gpu_us | prompt | admission | trial |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    ordered = sorted(
        cells.values(),
        key=lambda c: (c["objective"], bucket_sort_key(c["bucket"]), c["dispatch_bin"], family_rank(c["family"])),
    )
    for c in ordered:
        rnds = " / ".join("%.2f" % x for x in c["rounds"])
        gpu_us = "%.1f" % c["gpu_us"] if c.get("gpu_us") is not None else "-"
        lines.append(
            "| %s | %s | %s | %s | `%s` | %.2f | %s | %.1f | %s | %d | %s | %s |"
            % (
                c["objective"],
                c["bucket"],
                c["dispatch_bin"],
                c["family"],
                c["candidate"]["name"],
                c["tokens_per_second"],
                rnds,
                c["dispatches_per_token"],
                gpu_us,
                c["prompt_tokens"],
                c["admission"],
                c["trial"],
            )
        )

    lines += [
        "",
        "## Published reference baselines (not candidates)",
        "",
        "| engine | prompt | prefill tok/s | decode tok/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for eng, prm, pf, dec in PUBLISHED_BASELINES:
        lines.append("| %s | %d | %.1f | %.2f |" % (eng, prm, pf, dec))
    lines.append("")
    return "\n".join(lines)


def cmd_run(args):
    binary = args.binary.resolve()
    checkpoint = args.checkpoint.resolve()
    lock_path = args.lock.resolve() if args.lock else None
    parity_tokens = args.parity_tokens.resolve()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    candidates = []
    for p in args.candidates:
        c = json.loads(pathlib.Path(p).read_text())
        candidates.append((pathlib.Path(p).resolve(), c))
    candidates.sort(key=lambda item: family_rank(item[1].get("family", item[1]["name"])))

    if not candidates:
        raise SystemExit("no candidates supplied")

    binary_sha = sha256_file(binary)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Stage 1: Admission / cake_toks parity check vs megakernel-v1 (or first candidate)
    ref_idx = next((i for i, (_, c) in enumerate(candidates) if c["name"] == args.reference), 0)
    admitted = []
    with tempfile.TemporaryDirectory(prefix="inkling-qd-") as tmp:
        workdir = pathlib.Path(tmp)
        ref_path, ref_cand = candidates[ref_idx]
        ref_knobs_file = workdir / "ref_knobs.json"
        ref_knobs_file.write_text(json.dumps(ref_cand["knobs"]))
        print("[admission] evaluating reference '%s' on %s" % (ref_cand["name"], parity_tokens.name), flush=True)
        ref_summary, ref_logits = run_eval_cake(binary, checkpoint, ref_knobs_file, parity_tokens, lock_path, workdir / "ref.f32")

        for idx, (cpath, cand) in enumerate(candidates):
            cname = cand["name"]
            fam = cand.get("family", cname)
            kfile = workdir / ("knobs-%d.json" % idx)
            kfile.write_text(json.dumps(cand["knobs"]))
            if idx == ref_idx:
                summ, lg = ref_summary, ref_logits
            else:
                summ, lg = run_eval_cake(binary, checkpoint, kfile, parity_tokens, lock_path, workdir / ("cand-%d.f32" % idx))
            nan_cnt = int(np.isnan(lg).sum())
            diff = np.abs(lg - ref_logits)
            max_abs = float(diff.max())
            top1_agree = int(sum(u == v for u, v in zip(summ["top1"], ref_summary["top1"])))
            n_pos = len(ref_summary["positions"])
            geq = summ["greedy"] == ref_summary["greedy"]
            is_ok = (nan_cnt == 0) and (top1_agree == n_pos)
            parity_info = {
                "tokens_file": parity_tokens.name,
                "positions": n_pos,
                "nan_count": nan_cnt,
                "max_abs_diff": max_abs,
                "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
                "top1_agree": top1_agree,
                "greedy_equal": geq,
            }
            print(
                "  %-15s (%s): nan=%d max_abs=%.3e top1=%d/%d greedy_eq=%s -> %s"
                % (cname, fam, nan_cnt, max_abs, top1_agree, n_pos, geq, "ADMITTED" if is_ok else "REJECTED"),
                flush=True,
            )
            if not is_ok:
                with Archive(args.archive) as a:
                    a.append_trial(
                        {
                            "trial": "%s-%s" % (timestamp, cname),
                            "time": now_iso(),
                            "candidate": {"name": cname, "knobs": cand["knobs"], "family": fam},
                            "parent": cand.get("parent"),
                            "verdict": "rejected",
                            "admission": {"admitted": False, "mode": "inkling-empirical", "parity": parity_info},
                            "binary_sha256": binary_sha,
                        }
                    )
                continue
            admitted.append((cand, kfile, parity_info))

        # Stage 2: ABBA counterbalanced timing per prompt length, saving incrementally
        if not admitted:
            raise SystemExit("no admitted candidates to benchmark")

        results_by_name = {
            cand["name"]: {"cand": cand, "parity": pinfo, "timings": {}}
            for cand, _, pinfo in admitted
        }

        for length in lengths:
            print("=== LENGTH %d (rounds=%d ABBA, repeats=%d, decode=%d) ===" % (length, args.rounds, args.repeats, args.decode_tokens), flush=True)
            for rnd in range(args.rounds):
                order = admitted if rnd % 2 == 0 else list(reversed(admitted))
                for cand, kfile, _ in order:
                    cname = cand["name"]
                    r = run_bench_once(binary, checkpoint, length, kfile, args.decode_tokens, args.repeats, lock_path)
                    results_by_name[cname]["timings"].setdefault(length, []).append(r)
                    det = r.get("decode_detail") or {}
                    print(
                        "  [L=%5d r%d] %-14s prefill=%7.2f tok/s (disp=%.1f)  decode=%5.2f tok/s (disp=%.1f gpu_us=%.1f redos=%s)"
                        % (
                            length,
                            rnd,
                            cname,
                            r["prefill_tokens_per_second"],
                            r["dispatches_per_prefill_forward"],
                            r["decode_tokens_per_second"],
                            r["dispatches_per_decode_token"],
                            det.get("gpu_us", -1.0),
                            det.get("gate_redos", "-"),
                        ),
                        flush=True,
                    )

            # Commit current length cells to archive.json / elites.md immediately
            bkt = bucket(length)
            with Archive(args.archive) as a:
                for cand, _, pinfo in admitted:
                    cname = cand["name"]
                    fam = cand.get("family", cname)
                    runs = results_by_name[cname]["timings"][length]
                    trial_id = "%s-%s" % (timestamp, cname)
                    for obj in ("prefill", "decode"):
                        key_tps = "%s_tokens_per_second" % obj
                        rnd_vals = [float(r[key_tps]) for r in runs]
                        med_tps = float(statistics.median(rnd_vals))
                        disp_key = "dispatches_per_prefill_forward" if obj == "prefill" else "dispatches_per_decode_token"
                        disp = float(runs[0][disp_key])
                        dbin = dispatch_bin(obj, disp)
                        cell_key = "%s|%s|%s|%s" % (obj, bkt, dbin, fam)
                        gpu_vals = [float(r["decode_detail"]["gpu_us"]) for r in runs if "decode_detail" in r]
                        med_gpu_us = float(statistics.median(gpu_vals)) if gpu_vals else None
                        a.data["cells"][cell_key] = {
                            "objective": obj,
                            "bucket": bkt,
                            "dispatch_bin": dbin,
                            "family": fam,
                            "candidate": {"name": cname, "knobs": cand["knobs"]},
                            "tokens_per_second": med_tps,
                            "rounds": rnd_vals,
                            "prompt_tokens": length,
                            "dispatches_per_token": disp,
                            "gpu_us": med_gpu_us,
                            "parity": pinfo,
                            "admission": "inkling-empirical",
                            "trial": trial_id,
                            "binary_sha256": binary_sha,
                            "worktree": str(REPO),
                            "updated": now_iso(),
                            "parent": cand.get("parent"),
                        }

        with Archive(args.archive) as a:
            for cand, _, pinfo in admitted:
                cname = cand["name"]
                fam = cand.get("family", cname)
                a.append_trial(
                    {
                        "trial": "%s-%s" % (timestamp, cname),
                        "time": now_iso(),
                        "candidate": {"name": cname, "knobs": cand["knobs"], "family": fam},
                        "parent": cand.get("parent"),
                        "hypothesis": cand.get("hypothesis"),
                        "expected": cand.get("expected"),
                        "admission": {"admitted": True, "mode": "inkling-empirical", "parity": pinfo},
                        "timings": {str(k): v for k, v in results_by_name[cname]["timings"].items()},
                        "verdict": "elite",
                        "binary_sha256": binary_sha,
                    }
                )
    print("Wrote %s/archive.json and %s/elites.md" % (args.archive, args.archive))
    return 0


def cmd_render(args):
    with Archive(args.archive):
        pass
    print("Rendered %s/elites.md" % args.archive)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("binary", type=pathlib.Path)
    r.add_argument("checkpoint", type=pathlib.Path)
    r.add_argument("--lock", type=pathlib.Path, required=True)
    r.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    r.add_argument("--candidates", nargs="+", required=True)
    r.add_argument("--reference", default="megakernel-v1")
    r.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--decode-tokens", type=int, default=64)
    r.add_argument("--parity-tokens", type=pathlib.Path, default=REPO / ".zig-cache/final/cake_toks.json")
    ren = sub.add_parser("render")
    ren.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    args = p.parse_args()
    sys.exit({"run": cmd_run, "render": cmd_render}[args.command](args))


if __name__ == "__main__":
    main()
