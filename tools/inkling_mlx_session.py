#!/usr/bin/env python3
"""A persistent Inkling 2 Mini session on mlx-lm, speaking `nano session`'s JSONL protocol.

    <python with mlx-lm> -B tools/inkling_mlx_session.py CHECKPOINT [--chunk 4096] [--context N]

One request per line on stdin: {id, tokens, max_tokens, reset, return_logits}.
`tokens` are only the new suffix; with `reset` the cache is a new `make_cache()`.
Every generated token, EOS included, is committed to the cache, as in the native
session. One response line each, with the accounting fields tools/codegraff_adapter.py
checks. Prefill is mlx-lm's own loop (generate.py `_prefill`: chunks evaluated for
cache state, then the last token alone for logits); decode enqueues the next
step before reading back the previous token, as mlx-lm's generation loop does.
The same code times mlx-lm in tools/nano_inkling_tasks.py.
`return_logits` is not supported. A throwaway request runs before the first line is read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", type=Path)
p.add_argument("--chunk", type=int, default=4096)
p.add_argument("--context", type=int, default=16384)
args = p.parse_args()

config = json.loads((args.checkpoint / "config.json").read_text())
eos = config["eos_token_id"]
eos = set(eos if isinstance(eos, list) else [eos])
model, _ = load_model(args.checkpoint, model_config={"vision_config": None})
state = {"cache": model.make_cache(), "position": 0}


def argmax(logits):
    return mx.argmax(logits[0, -1]).astype(mx.int32)


def run(request):
    tokens, limit = request["tokens"], request["max_tokens"]
    if request.get("reset"):
        state["cache"] = model.make_cache()
        state["position"] = 0
        mx.clear_cache()
    cache, before = state["cache"], state["position"]
    if before + len(tokens) + limit > args.context:
        raise ValueError("context exhausted")
    y = mx.array(tokens)
    started = time.perf_counter()
    while y.size > 1:
        n = min(args.chunk, y.size - 1)
        model(y[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[n:]
        mx.clear_cache()
    current = argmax(model(y[None], cache=cache))
    mx.eval(current)
    prefilled = time.perf_counter()
    out, last = [], None
    while len(out) < limit:
        # Commit `current` to the cache by running it; its successor is computed behind the read-back.
        last = argmax(model(current.reshape(1, 1), cache=cache))
        mx.async_eval(last)
        value = int(current.item())
        out.append(value)
        if value in eos:
            break
        current = last
    if last is not None:
        mx.eval([c.state for c in cache])
    finished = time.perf_counter()
    state["position"] = before + len(tokens) + len(out)
    return dict(id=request.get("id", 0), tokens=out, eos=bool(out and out[-1] in eos), cache_before=before,
                cache_after=state["position"], prefill_tokens=len(tokens), decode_forward_tokens=len(out),
                reused_prefix_tokens=before, prefill_seconds=prefilled - started, decode_seconds=finished - prefilled,
                kv_allocated_bytes=0, kv_valid_bytes=0, kv_storage="mlx", prefill_forwards=1, dispatches=0,
                prefill_logits=None, final_logits=None)


run({"id": 0, "tokens": [(7 * i + 3) % 20000 + 1000 for i in range(300)], "max_tokens": 4, "reset": True})
print("mlx session: ready, chunk={}".format(args.chunk), file=sys.stderr, flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    try:
        response = run(json.loads(line))
    except Exception as error:  # the adapter treats a missing response as a failed worker
        print("mlx session failed: {}".format(error), file=sys.stderr, flush=True)
        sys.exit(1)
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
