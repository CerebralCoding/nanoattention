"""Pools nano_inkling_tasks.py result files into one summary without machine paths.

    python3 tools/nano_inkling_tasks_summary.py TASKS.json OUT.json RESULT.json [RESULT.json ...]

Per task: every sample of each engine (seconds to first token, decode tokens a
second, the generated ids). Per prompt length: medians over that length's tasks
and rounds, and how many paired samples native won.
"""
import json
import statistics
import sys
from pathlib import Path

tasks = json.loads(Path(sys.argv[1]).read_text())
out_path = Path(sys.argv[2])
med = statistics.median
samples = []
generate = None
for i, path in enumerate(sys.argv[3:]):
    data = json.loads(Path(path).read_text())
    generate = data["generate"]
    for s in data["samples"]:
        # mlx-lm's own default prefill chunk is 2,048 tokens; other chunk sizes are separate arms.
        if s["engine"] == "mlx" and s.get("chunk") != 2048:
            s = {**s, "engine": f"mlx{s['chunk']}"}
        samples.append({**s, "round": i + 1})
info = {t["name"]: t for t in tasks["tasks"]}
out = {"generate": generate, "warmup_tokens": len(tasks["warmup"]), "tasks": {}, "lengths": {}}
for name, t in info.items():
    row = {"kind": t["kind"], "prompt_tokens": t["length"], "instruction": t["instruction"], "files": t["files"], "engines": {}}
    for s in samples:
        if s["task"] != name:
            continue
        e = row["engines"].setdefault(s["engine"], {"first_seconds": [], "decode_tps": [], "tokens": [], "rounds": [], "chunk": s.get("chunk", s.get("batch"))})
        e["rounds"].append(s["round"])
        e["first_seconds"].append(s["first_seconds"])
        e["decode_tps"].append(s["decode_tps"])
        e["tokens"].append(s["tokens"])
    out["tasks"][name] = row
for n in sorted({t["length"] for t in info.values()}):
    rows = [r for r in out["tasks"].values() if r["prompt_tokens"] == n]
    summary = {}
    engines = ["native"] + sorted({e for r in rows for e in r["engines"] if e != "native"})
    for engine in engines:
        first = [x for r in rows for x in r["engines"].get(engine, {}).get("first_seconds", [])]
        decode = [x for r in rows for x in r["engines"].get(engine, {}).get("decode_tps", [])]
        if first:
            summary[engine] = {"samples": len(first), "first_token_seconds": med(first), "prefill_tps": n / med(first), "decode_tps": med(decode)}
    for other in engines[1:]:
        pairs_first = pairs_decode = wins_first = wins_decode = same_ids = 0
        for r in rows:
            a, b = r["engines"].get("native"), r["engines"].get(other)
            if not a or not b:
                continue
            # Pair samples of the same round (a result file is one round).
            ra, rb = a["rounds"], b["rounds"]
            for i, round_a in enumerate(ra):
                if round_a not in rb:
                    continue
                j = rb.index(round_a)
                pairs_first += 1
                wins_first += a["first_seconds"][i] < b["first_seconds"][j]
                wins_decode += a["decode_tps"][i] > b["decode_tps"][j]
            same_ids += a["tokens"][0][:generate - 1] == b["tokens"][0][:generate - 1]
        if pairs_first:
            summary[f"native_vs_{other}"] = {"pairs": pairs_first, "first_token_wins": wins_first, "decode_wins": wins_decode,
                                            "tasks_with_identical_ids": same_ids, "tasks": len(rows)}
    out["lengths"][str(n)] = summary
out_path.write_text(json.dumps(out, indent=1) + "\n")
names = ["native"] + sorted({e for v in out["lengths"].values() for e in v if not e.startswith("native")})
print("| Prompt | " + " | ".join(f"{e} first / decode" for e in names) + " |")
for n, v in out["lengths"].items():
    print(f"| {int(n):,} | " + " | ".join(f"{v[e]['first_token_seconds']:.2f} s / {v[e]['decode_tps']:.1f}" if e in v else "-" for e in names) + " |")
for n, v in out["lengths"].items():
    print(n, {k: (x["first_token_wins"], x["decode_wins"], x["pairs"], x["tasks_with_identical_ids"]) for k, x in v.items() if k.startswith("native_vs")})
