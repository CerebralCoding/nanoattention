#!/usr/bin/env python3
"""Inkling 2 Mini on distinct real tasks, every request from an empty cache.

    nano_inkling_tasks.py build CHECKPOINT TASKS.json [--root REPO]
    nano_inkling_tasks.py run NANO CHECKPOINT TASKS.json OUT.json --mlx-python PY --lock LOCK
        [--engines native,mlx,exo] [--only NAME ...] [--generate 32]
        [--native-batch 4096] [--mlx-chunk 2048] [--exo-chunk 2048] [--exo-runner CMD ...]

`build` (needs `transformers`) writes twelve prompts: three kinds of real text
from a repository (Zig source, Python tools, Markdown documentation) at 4,096,
8,192, 16,384 and 32,768 prompt tokens. Every prompt is a different slice of
text followed by a different instruction, rendered with the checkpoint's chat
template; no two prompts share any document text. It also writes a 4,400-token
warm-up prompt from a fourth kind of text (the Lean proofs) that no task uses.

`run` times each task once per engine, each in a fresh process under the
inference lock, engine order rotating from task to task:

  native  `nano session`: the warm-up prompt, then the task with `reset: true`.
          The reply's `reused_prefix_tokens` and `cache_before` must be 0.
  mlx     mlx-lm in its own process: the warm-up prompt on its own cache, then
          the task on a new `make_cache()`. Prefill is mlx-lm's own loop
          (generate.py `_prefill`): chunks of `--mlx-chunk` tokens evaluated for
          cache state only, then the last token alone for logits. Decode feeds
          each argmax back with the next step enqueued before the previous
          token is read, as mlx-lm's generation loop does.
  exo     optional, through `--exo-runner`, a command given
          `FIXTURE FOLDER LENGTH GENERATE CHUNK` that prints one JSON sample
          (the exo checkout and adapter are not part of this repository).
          exo's benchmark route warms up on the prompt it then times, with its
          prefix cache off; that can only help exo.

Native and mlx are never timed on a prompt the process has already seen, and
nothing a timed request computes existed before it started. Prefill chunk sizes
are each engine's own setting: sweep them (`--native-batch`, `--mlx-chunk`)
before comparing, since the best size differs. Reported per sample: seconds to the first
generated token, the prefill rate that implies, and decode tokens a second over
the single-token forwards that follow. Greedy; token ids are kept so outputs
can be compared, and they need not agree between engines.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LENGTHS = (4096, 8192, 16384, 32768)
KINDS = {
    "zig": (("src/mega/*.zig", "src/*.zig"), "//"),
    "python": (("tools/*.py",), "#"),
    "docs": (("docs/*.md", "benchmarks/*.md", "README.md"), "<!--"),
}
WARMUP_KIND = (("proofs/*.lean",), "--")
INSTRUCTIONS = {
    ("zig", 4096): "Review the Zig code above. List the three most likely bugs, each with the function it is in.",
    ("zig", 8192): "Explain, for a new contributor, what the Zig code above does and how its main functions call each other.",
    ("zig", 16384): "Which functions in the Zig code above allocate memory, and who frees it? Answer as a table.",
    ("zig", 32768): "Write documentation comments for the five most important public functions in the Zig code above.",
    ("python", 4096): "Write pytest unit tests for the Python code above. Cover the argument parsing first.",
    ("python", 8192): "Describe every command-line flag the Python scripts above accept, grouped by script.",
    ("python", 16384): "The Python code above shells out to other programs. List each call and what could make it fail.",
    ("python", 32768): "Refactor plan: which helpers in the Python code above are duplicated and could be shared?",
    ("docs", 4096): "Summarize the document above in five bullet points for someone deciding whether to use this project.",
    ("docs", 8192): "List every limitation or caveat the documents above admit to, one line each.",
    ("docs", 16384): "Extract every benchmark figure in the documents above into one table: what was measured, the value, the unit.",
    ("docs", 32768): "Write a one-page changelog from the documents above, oldest change first.",
}
WARMUP_INSTRUCTION = "State in one sentence what the Lean file above proves."
# Longer than one prefill chunk of either engine, so chunk-sized buffers, kernels
# and graphs exist before the timed request; none of its text is in any task.
WARMUP_TOKENS = 4400
KNOBS = ("expert_simd=true expert_wide_dot=true expert_gate_pair=true expert_prefill=true expert_wide_k=true "
         "expert_bf16_tiles=true expert_mlx_scale=true bf16_linear_tile=true bf16_prefill_tile=8 "
         "bf16_prefill_matrix=true bf16_decode_vec=true fuse_bf16_cast=true concurrent_scratch=true "
         "relative_split=true relative_split_4=true relative_tile_prefill=true relative_stripes=true "
         "scratch_reuse=true expert_tile_rows=4096 bf16_matrix_small=true").split()
MARK = "@@DOCUMENT@@"


def build(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False)
    tok.chat_template = (args.checkpoint / "chat_template.jinja").read_text()

    def corpus(globs, comment):
        """One kind's files, each under a header naming it, as token ids with the file each id came from."""
        ids, files = [], []
        for pattern in globs:
            for path in sorted(args.root.glob(pattern)):
                text = f"{comment} file: {path.relative_to(args.root)}\n{path.read_text(errors='replace')}\n\n"
                part = tok.encode(text, add_special_tokens=False)
                ids += part
                files += [str(path.relative_to(args.root))] * len(part)
        return ids, files

    def render(document, instruction, length):
        """A user turn of exactly `length` tokens: the template, the start of `document`, the instruction."""
        text = tok.apply_chat_template([{"role": "user", "content": f"{MARK}\n\n{instruction}"}], tokenize=False,
                                       add_generation_prompt=False, reasoning_effort="low")
        before, after = text.split(MARK)
        head = tok.encode(before, add_special_tokens=False)
        tail = tok.encode(after, add_special_tokens=False)
        room = length - len(head) - len(tail)
        if room <= 0 or room > len(document):
            raise SystemExit(f"cannot fit a {length}-token prompt ({len(document)} document tokens)")
        return head + document[:room] + tail, room

    tasks = []
    for kind, (globs, comment) in KINDS.items():
        ids, files = corpus(globs, comment)
        at = 0
        for length in LENGTHS:
            prompt, used = render(ids[at:], INSTRUCTIONS[kind, length], length)
            assert len(prompt) == length
            tasks.append({"name": f"{kind}-{length}", "kind": kind, "length": length,
                          "instruction": INSTRUCTIONS[kind, length], "document_tokens": [at, at + used],
                          "files": sorted(set(files[at:at + used])), "ids": prompt})
            at += used  # the next task starts where this one's text ended
    ids, _ = corpus(*WARMUP_KIND)
    warm, _ = render(ids, WARMUP_INSTRUCTION, WARMUP_TOKENS)
    args.out.write_text(json.dumps({"reasoning_effort": "low", "warmup": warm, "tasks": tasks}) + "\n")
    for t in tasks:
        print(f"{t['name']:<14} {len(t['ids']):>6} tokens from {len(t['files'])} file(s), document tokens {t['document_tokens']}")


def mlx_worker(argv):
    checkpoint, tasks_path, name, chunk, generate = argv[0], Path(argv[1]), argv[2], int(argv[3]), int(argv[4])
    import mlx.core as mx
    from mlx_lm.utils import load_model

    data = json.loads(tasks_path.read_text())
    task = next(t for t in data["tasks"] if t["name"] == name)
    model, _ = load_model(Path(checkpoint), model_config={"vision_config": None})

    def run(ids, count):
        cache = model.make_cache()
        y = mx.array(ids)
        t0 = time.perf_counter()
        while y.size > 1:  # mlx-lm generate.py _prefill
            n = min(chunk, y.size - 1)
            model(y[:n][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[n:]
            mx.clear_cache()
        pred = mx.argmax(model(y[None], cache=cache)[0, -1])
        mx.eval(pred)
        tokens = [int(pred.item())]
        t1 = time.perf_counter()
        prev = None
        for _ in range(count):
            pred = mx.argmax(model(pred.astype(mx.int32).reshape(1, 1), cache=cache)[0, -1])
            mx.async_eval(pred)
            if prev is not None:
                tokens.append(int(prev.item()))
            prev = pred
        tokens.append(int(prev.item()))
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, tokens

    run(data["warmup"], 8)
    mx.clear_cache()
    first, decode, tokens = run(task["ids"], generate)
    print(json.dumps({"first_seconds": first, "decode_seconds": decode, "decode_forwards": generate,
                      "tokens": tokens[:generate], "chunk": chunk, "peak_memory_bytes": mx.get_peak_memory()}))


def guarded(args, command, timeout):
    """`command` under tools/guarded_run.py and the lock, waiting while another run holds it."""
    guard = [sys.executable, str(Path(__file__).with_name("guarded_run.py")), "--lock", str(args.lock),
             "--timeout", str(timeout), "--"]
    for _ in range(360):
        done = subprocess.run(guard + command, capture_output=True, text=True)
        if done.returncode == 0:
            return done.stdout
        if "BlockingIOError" in done.stderr or "Not enough normal-pressure" in done.stderr:
            time.sleep(10)
            continue
        raise RuntimeError(f"{command[0]} failed:\n{done.stderr[-1500:]}")
    raise RuntimeError("the inference lock never became free")


def native_session(argv):
    """One `nano session`: the warm-up, then the task from an empty cache. Prints one JSON sample."""
    nano, checkpoint, tasks_path, name, batch, generate = argv[0], argv[1], Path(argv[2]), argv[3], int(argv[4]), int(argv[5])
    data = json.loads(tasks_path.read_text())
    task = next(t for t in data["tasks"] if t["name"] == name)
    command = [nano, "session", checkpoint, "--context", str(max(len(task["ids"]), len(data["warmup"])) + generate + 8), "--knob", f"batch={batch}"]
    for knob in KNOBS:
        command += ["--knob", knob]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def ask(request):
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("nano session closed: " + proc.stderr.read()[-600:])
        reply = json.loads(line)
        if "error" in reply:
            raise SystemExit(f"nano session: {reply}")
        return reply

    ask({"id": 1, "tokens": data["warmup"], "max_tokens": 8, "reset": True})
    reply = ask({"id": 2, "tokens": task["ids"], "max_tokens": generate, "reset": True})
    proc.stdin.close()
    proc.wait(timeout=120)
    if reply["reused_prefix_tokens"] != 0 or reply["cache_before"] != 0 or reply["prefill_tokens"] != len(task["ids"]):
        raise SystemExit(f"the task did not start from an empty cache: {reply}")
    print(json.dumps({"first_seconds": reply["prefill_seconds"], "decode_seconds": reply["decode_seconds"],
                      "decode_forwards": reply["decode_forward_tokens"], "tokens": reply["tokens"], "eos": reply["eos"],
                      "batch": batch, "reused_prefix_tokens": reply["reused_prefix_tokens"],
                      "prefill_forwards": reply["prefill_forwards"]}))


def run(args):
    data = json.loads(args.tasks.read_text())
    tasks = [t for t in data["tasks"] if not args.only or t["name"] in args.only]
    engines = args.engines.split(",")
    here = str(Path(__file__).resolve())
    results = json.loads(args.out.read_text()) if args.out.exists() else {"generate": args.generate, "samples": []}
    for index, task in enumerate(tasks):
        n = task["length"]
        order = engines[index % len(engines):] + engines[:index % len(engines)]
        for engine in order:
            budget = 240 + n // 12
            if engine == "native":
                command = [sys.executable, here, "native-session", str(args.nano.resolve()), str(args.checkpoint.resolve()),
                           str(args.tasks.resolve()), task["name"], str(args.native_batch), str(args.generate)]
            elif engine == "mlx":
                command = [str(args.mlx_python.absolute()), here, "mlx-worker", str(args.checkpoint.resolve()),
                           str(args.tasks.resolve()), task["name"], str(args.mlx_chunk), str(args.generate)]
            elif engine == "exo":
                if not args.exo_runner:
                    raise SystemExit("--exo-runner is required for the exo engine")
                folder = args.out.resolve().parent / f"{args.out.stem}-exo-{task['name']}-{int(time.time())}"
                fixture = args.out.resolve().parent / f"{args.out.stem}-fixture-{task['name']}.json"
                fixture.write_text(json.dumps({"bos": task["ids"][0], "code": task["ids"][1:], "decode": [],
                                               "generation": args.generate}) + "\n")
                command = [*args.exo_runner, str(fixture), str(folder), str(n), str(args.generate), str(args.exo_chunk)]
                budget = 400 + n // 5
            else:
                raise SystemExit(f"unknown engine {engine}")
            sample = json.loads(guarded(args, command, budget).strip().splitlines()[-1])
            sample.update(task=task["name"], kind=task["kind"], length=n, engine=engine)
            sample["first_token_tps"] = n / sample["first_seconds"]
            sample["decode_tps"] = sample["decode_forwards"] / sample["decode_seconds"]
            results["samples"].append(sample)
            args.out.write_text(json.dumps(results, indent=1) + "\n")
            print(f"{task['name']:<14} {engine:<7} first token {sample['first_seconds']:7.2f} s ({sample['first_token_tps']:7.1f} tok/s)  "
                  f"decode {sample['decode_tps']:6.2f} tok/s over {sample['decode_forwards']} forwards", flush=True)


def main(argv):
    if argv[:1] == ["mlx-worker"]:
        return mlx_worker(argv[1:])
    if argv[:1] == ["native-session"]:
        return native_session(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("checkpoint", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    r = sub.add_parser("run")
    r.add_argument("nano", type=Path)
    r.add_argument("checkpoint", type=Path)
    r.add_argument("tasks", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--mlx-python", type=Path, required=True)
    r.add_argument("--lock", type=Path, required=True)
    r.add_argument("--engines", default="native,mlx")
    r.add_argument("--only", nargs="*")
    r.add_argument("--generate", type=int, default=32)
    r.add_argument("--native-batch", type=int, default=4096)
    r.add_argument("--mlx-chunk", type=int, default=2048)
    r.add_argument("--exo-chunk", type=int, default=2048)
    r.add_argument("--exo-runner", nargs="+")
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
