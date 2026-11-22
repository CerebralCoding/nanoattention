#!/usr/bin/env python3
"""Replay the requests of a recorded agent session on one engine, timing each.

    <python> -B tools/inkling_agent_replay.py REQUESTS.jsonl OUT.json CHECKPOINT --engine native|mlx
        [--binary nano] [--mlx-python PY] [--context 16384]

REQUESTS.jsonl is what `tools/inkling_adapter.py --record` wrote: per request
the prompt's IDs, the new suffix after the reused prefix, and the IDs the
engine generated. The replay gives the engine the same work: one fresh worker
(warmed on a throwaway request), then each request's recorded suffix with
the recorded reuse, generating as many tokens as the recording did. The
engines can disagree on greedy tokens; when this engine's tokens differ
from the recording the cache would hold different text than the next request
assumes, so before the next request the cache is rebuilt, untimed, from
the recorded prompt prefix (reset, one prefill, no generation). The timed
numbers are then the same prefill and decode lengths on either engine, and
`diverged` says how often the engine's own tokens differed. Run it under the
inference lock (tools/guarded_run.py).
"""
import argparse
import json
from pathlib import Path
import sys

from coding_multiturn import NativeSession
import inkling_adapter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("requests", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--engine", choices=("native", "mlx"), required=True)
    p.add_argument("--binary", type=Path)
    p.add_argument("--mlx-python", type=Path)
    p.add_argument("--context", type=int, default=16384)
    args = p.parse_args()
    records = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if args.engine == "native":
        command = inkling_adapter.native_command(args.binary.resolve(), args.checkpoint.resolve(), args.context)
    else:
        command = inkling_adapter.mlx_command(args.mlx_python.absolute(), args.checkpoint.resolve(), args.context)
    rows, serial, diverged = [], 0, 0
    with NativeSession(command, timeout=900) as worker:
        def ask(tokens, limit, reset):
            nonlocal serial
            serial += 1
            return worker.request(dict(id=serial, tokens=tokens, max_tokens=limit, reset=reset, return_logits=False))
        ask([(7 * i + 3) % 20000 + 1000 for i in range(inkling_adapter.WARM_TOKENS)], 4, True)  # warm-up, not timed
        stale = False
        for index, r in enumerate(records):
            suffix, reset = r["suffix_ids"], r["reset"]
            if stale and not reset:  # rebuild the cache the recording had, untimed
                ask(r["prompt_ids"][:len(r["prompt_ids"]) - len(suffix)], 0, True)
            stale = False
            limit = len(r["generated_ids"])
            reply = ask(suffix, limit, reset)
            same = reply["tokens"] == r["generated_ids"]
            diverged += not same
            stale = not same
            rows.append(dict(index=index, prompt_tokens=len(r["prompt_ids"]), prefill_tokens=len(suffix),
                             reused_prefix_tokens=reply["reused_prefix_tokens"], generated=len(reply["tokens"]),
                             prefill_seconds=reply["prefill_seconds"], decode_seconds=reply["decode_seconds"],
                             same_tokens=same,
                             recorded_prefill_seconds=r["prefill_seconds"], recorded_decode_seconds=r["decode_seconds"]))
            if not same:
                rows[-1]["first_difference"] = next((i for i, (a, b) in enumerate(zip(reply["tokens"], r["generated_ids"])) if a != b),
                                                    min(len(reply["tokens"]), limit))
    total_prefill = sum(x["prefill_seconds"] for x in rows)
    total_decode = sum(x["decode_seconds"] for x in rows)
    generated = sum(x["generated"] for x in rows)
    result = dict(engine=args.engine, requests=rows, prefill_seconds=total_prefill, decode_seconds=total_decode,
                  total_seconds=total_prefill + total_decode, generated_tokens=generated,
                  decode_tokens_per_second=generated / total_decode if total_decode else 0.0, diverged=diverged)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("{}: {} requests, prefill {:.2f} s, decode {:.2f} s ({:.1f} tok/s), diverged {}".format(
        args.engine, len(rows), total_prefill, total_decode, result["decode_tokens_per_second"], diverged), file=sys.stderr)


if __name__ == "__main__":
    main()
