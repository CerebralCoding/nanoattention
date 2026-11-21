# Inkling 2 Mini at 4K to 32K: twelve real tasks, cold cache

Twelve different prompts built from real text (Zig source, Python tools, Markdown
documentation, each at 4,096, 8,192, 16,384 and 32,768 prompt tokens, each with
its own instruction), every one timed from an empty cache. With the opt-in knobs
in [docs/MEGAKERNEL.md](../docs/MEGAKERNEL.md#inkling-2-mini-experimental):

- **Decode:** native is 22 to 25% faster than the pinned mlx-lm reference and 43
  to 53% faster than exo's MLX adapter, in every paired sample.
- **First token against exo's adapter:** native is 24 to 26% sooner, in every
  pair.
- **First token against mlx-lm depends on mlx-lm's prefill chunk.** At its
  default of 2,048 tokens native is 0.6 to 4.1% sooner, in all 36 pairs. With
  mlx-lm set to 4,096-token chunks, its best size here, native is **0.5 to 4.4%
  later** and wins 2 of 24 pairs. Native does not beat a tuned mlx-lm on prefill.

One Apple M3 Ultra (60 GPU cores, 256 GB), the original
`thinkingmachines/Inkling-2-Mini` checkpoint (`ed40e7da625f`), `nano` built
ReleaseSafe, `mlx==0.32.4`, `mlx-lm` at `0c4c99192230`. Pooled samples:
[`inkling-long-context.json`](inkling-long-context.json).

## Protocol

`tools/nano_inkling_tasks.py`. `build` cuts the repository's own text into
twelve prompts that share no document text, renders each with the checkpoint's
chat template, and writes a 4,400-token warm-up prompt from a fourth kind of
text (the Lean proofs) that no task uses. `run` gives every (task, engine) its
own process under one inference lock, engine order rotating from task to task:

- **native:** `nano session` answers the warm-up prompt, then the task with
  `reset: true`. The runner fails unless the reply reports zero reused prefix
  tokens and an empty cache. 4,096-row prefill batches.
- **mlx:** the warm-up prompt on its own cache, then the task on a new
  `make_cache()`. Prefill is mlx-lm's own loop (chunks evaluated for cache state,
  then the last token alone for logits); decode enqueues the next step before
  reading back the previous token, as mlx-lm's generation loop does.
- **exo:** exo `a60b95698ca8` serving a local text-only MLX-VLM adapter
  (`434a1bedd1e8`) with `--no-batch` and direct cold prefill, through its HTTP
  benchmark route with the prefix cache off. That route cannot warm up on a
  different prompt, so exo warms up on the prompt it is then timed on, which can
  only help exo. Stock exo does not load this checkpoint; this is not a
  comparison with an exo release.

Each sample is the seconds from the request to the first generated token and
the decode rate over the 31 single-token forwards after it. Greedy. Five rounds
of native; mlx-lm at its default chunk in rounds 1 to 3 and at 4,096 in rounds 4
and 5; exo in round 1.

## Results

Medians over a length's three tasks and all rounds: seconds to first token /
decode tokens a second. The last three columns are native's first-token time and
decode rate relative to that engine (negative time is sooner).

| Prompt | Native | MLX, 2,048 chunk | MLX, 4,096 chunk | exo adapter | vs MLX 2,048 | vs MLX 4,096 | vs exo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 3.30 s / 37.8 | 3.44 s / 30.6 | 3.28 s / 30.5 | 4.45 s / 26.5 | -4.1% / +24% | +0.5% / +24% | -26.0% / +43% |
| 8,192 | 6.74 s / 37.4 | 6.95 s / 30.2 | 6.60 s / 29.9 | 8.89 s / 25.6 | -3.0% / +24% | +2.2% / +25% | -24.1% / +46% |
| 16,384 | 14.06 s / 36.5 | 14.28 s / 29.9 | 13.62 s / 29.7 | 18.56 s / 24.9 | -1.6% / +22% | +3.2% / +23% | -24.3% / +46% |
| 32,768 | 30.24 s / 35.0 | 30.41 s / 28.3 | 28.96 s / 28.4 | 40.53 s / 23.0 | -0.6% / +24% | +4.4% / +23% | -25.4% / +53% |

| Against | Paired samples | Native sooner to first token | Native faster decode | Tasks with the same first 31 IDs |
| --- | ---: | ---: | ---: | ---: |
| MLX, 2,048 chunk | 36 | 36 | 36 | 8 of 12 |
| MLX, 4,096 chunk | 24 | 2 | 24 | 6 of 12 |
| exo adapter | 12 | 12 | 12 | 7 of 12 |

The two first-token wins against 4,096-chunk MLX are both at 4K. The narrowest
win against 2,048-chunk MLX is at 32K, where native's slowest sample on a task
is 0.2% ahead of MLX's fastest.

Per task, medians of that task's samples:

| Task | Native | MLX, 2,048 chunk | MLX, 4,096 chunk | exo adapter |
| --- | ---: | ---: | ---: | ---: |
| zig-4096 | 3.28 / 37.9 | 3.44 / 30.1 | 3.31 / 30.2 | 4.44 / 26.5 |
| zig-8192 | 6.73 / 37.4 | 6.93 / 30.2 | 6.60 / 29.1 | 8.92 / 25.6 |
| zig-16384 | 14.05 / 36.4 | 14.30 / 29.3 | 13.61 / 24.0 | 18.56 / 24.9 |
| zig-32768 | 30.24 / 35.0 | 30.40 / 28.1 | 28.96 / 26.6 | 40.47 / 23.0 |
| python-4096 | 3.30 / 37.8 | 3.44 / 30.6 | 3.29 / 31.0 | 4.45 / 25.8 |
| python-8192 | 6.74 / 37.4 | 6.93 / 29.5 | 6.59 / 29.5 | 8.89 / 25.6 |
| python-16384 | 14.06 / 36.5 | 14.26 / 30.0 | 13.59 / 30.1 | 18.49 / 24.9 |
| python-32768 | 30.17 / 35.0 | 30.62 / 28.3 | 28.96 / 28.6 | 40.53 / 22.6 |
| docs-4096 | 3.31 / 37.8 | 3.45 / 31.0 | 3.28 / 30.0 | 4.45 / 26.5 |
| docs-8192 | 6.76 / 37.4 | 6.95 / 30.2 | 6.60 / 30.6 | 8.87 / 25.9 |
| docs-16384 | 14.10 / 36.5 | 14.28 / 30.1 | 13.69 / 23.9 | 18.96 / 25.1 |
| docs-32768 | 30.27 / 35.0 | 30.40 / 28.5 | 29.06 / 28.4 | 40.83 / 23.0 |

The 4,096-chunk MLX arm has two samples a task; its 24.0 and 23.9 at 16K are
medians of one ordinary and one slow sample.

**Outputs.** Native produced the same 32 IDs for a task in all five rounds. The
engines agree on the first 31 IDs for only some tasks (table above), and
mlx-lm's own output changes with its chunk size on five of them. Every reply
decodes to on-topic reasoning text about the task; where they differ they
differ in wording a few tokens in ("Need answer review of Zig code. Need
inspect carefully. User asks…" against "…Need list three most likely bugs…").
So these rows compare speed on outputs that are not always the same tokens.

### Chunk sizes

Each engine was swept before the rounds, on one or two tasks:

| Engine | Chunk | Result |
| --- | ---: | --- |
| native | 128 | 606 tok/s prefill, python-16384 |
| native | 512 | 924 tok/s, docs-4096 |
| native | 4,096 | 1,243 tok/s (3.30 s), 4K tasks |
| mlx-lm | 1,024 | 31.71 s to first token, zig-32768 |
| mlx-lm | 2,048 (default) | about 30.4 s |
| mlx-lm | 2,560 / 3,584 | 30.13 s / 29.30 s |
| mlx-lm | 4,096 | 29.04 s |
| mlx-lm | 8,192 | 29.17 s |
| exo adapter | 2,048 | 4.52 s docs-4096, 18.58 s python-16384 |
| exo adapter | 4,096 | 4.89 s, 20.27 s |

Native could not take batches above 512 rows before `scratch_reuse` (the
scratch arena exceeded its limit).

### Where native's prefill time goes

A profiled 32K prefill in 4,096-row chunks (3.84 s a chunk on average):
`mk_expert_block_up_nv` 30%, `mk_bf16_prefill_matrix32` 18%,
`mk_relative_prefill_tiles` 17%, `mk_expert_block_down_nv` 16%. Fitting the four
lengths, native's cost per chunk is about 2.5% above 4,096-chunk MLX's and its
cost that grows with context (global-layer attention) about 19% above, which is
why the gap widens from 0.5% at 4K to 4.4% at 32K.

## What changed

All opt-in; the reference executor's defaults are unchanged.

- **`relative_stripes`** (decode): every stored K and V row is a bfloat16 value
  held in a float, so each store also writes its upper 16 bits to a mirror and
  attention reads the mirror, the same numbers at half the memory traffic. A
  single row's keys are cut into stripes of 128 to 512, one threadgroup of 16
  SIMD groups per (KV head, stripe), the three query heads of a KV head sharing
  every load (`mk_relative_long_keys`, `mk_relative_long_merge`).
- **`relative_stripes`** (prefill): under 512 rows, the 16-key tiles of
  `mk_relative_attention_prefill` in two or four stripes of keys, fragments
  loaded straight from the mirrors (`mk_relative_prefill_stripes`). From 512
  rows, `mk_relative_prefill_tiles` (`src/mega/inkling_tiles.metal`), adapted
  from mlx-lm's batched relative attention kernel (MIT; see
  `THIRD_PARTY_NOTICES.md`) to read the mirrors and the ring-cache layout.
- **`scratch_reuse`**: scratch states whose lifetimes do not meet share arena
  words (`exec.packScratch`). No arithmetic changes. It is what lets Inkling
  take batches above 512 rows.
- **`expert_tile_rows`**: how many token rows have their expert routes listed
  and multiplied together. Each of the 128 experts sees about a sixteenth of the
  rows, so 512 rows leave most 32-row tiles partly empty; 4,096 fill them.
- **`bf16_matrix_small`**: the BF16 matrix tiles also take the small
  projections (relative, router, controller) that otherwise each launch a small
  tile kernel.

`nano attnbench` times one layer's attention on synthetic K and V. A single
decode row over a global layer went from 318 to 60 us at 4,096 keys and from
2,234 to 305 us at 32,768; over a full sliding window from 100 to 27 us. A
4,096-row prefill chunk over 32,768 keys takes 320 ms with the tiled kernel,
165 ms in stripes and 115 ms with the staged tiles; mlx-lm's own operation takes
108 ms on the same shapes.

Tried and removed, none faster on a real task: 64-row and 64-column expert
tiles, paired expert weight fragments, a 16-row dense matrix tile, and for
prefill attention 32-key tiles, 64 or 128 query rows a threadgroup, FP16 against
bfloat16 fragments, a head-major mirror layout, and one staged tile shared by
the three query heads of a KV head.

## Correctness

- `zig build test` and `zig build test-mega` pass. Kernel tests compare the
  striped kernels with `mk_relative_attention` over mirrors written by
  `mk_pack_bf16` (a wrapped sliding window, uneven stripes, more stripes than a
  row tile has tiles, a partial query tile, a one-key window), and the staged
  tiles with `mk_relative_attention_prefill` (bit-identical on an unwrapped
  cache, within 1e-4 on a wrapped window). `packScratch` has its own test.
- **Bit-identical to the path they replace:** `scratch_reuse` (logits at feeds
  512 and 1) and `expert_tile_rows`.
- **Not bit-identical:** `relative_stripes` and `bf16_matrix_small`; both
  associate sums differently. On a 420-token transcript
  (`tools/nano_inkling_attention_check.py`, 24 positions), striped decode
  differs from split decode by at most 3.53 (RMS 0.36) and striped prefill from
  tiled prefill by at most 3.06 (RMS 0.38), top token equal at 24 of 24; the
  existing paths differ from each other by as much. `bf16_matrix_small` on a
  4,096-token prompt at five positions: maximum logit differences from 0.15 to
  13.6 (RMS 2.5), top token equal at 5 of 5; changing only the feed size from
  128 to 4,096 moves the logits as far (maximum 13.1). Small differences are
  amplified by bfloat16 rounding and expert routing.
- A real greedy reply changes wording: "Write an LRU cache in Python..." matched
  the previous tuned path for 20 tokens, then diverged (639 tokens instead of
  744, both ending on EOS, both correct code).
- MLX logits parity for Inkling remains unestablished, as before.

## Superseded: seeded prompts

The first version of this file reported native ahead of mlx-lm on prefill by
about 5% at every length. That measurement fed both engines one seeded
pseudo-random token stream per length in 128-token chunks and warmed each
process up on the prompt it then timed. It held for that setup only: mlx-lm
prefills about 1.8 times faster in 2,048-token chunks than in 128-token ones
(1,147 against 640 tokens a second at 16K), and random IDs are not a task.
Those samples are kept in
[`inkling-long-context-seeded.json`](inkling-long-context-seeded.json); the
decode leads they showed (22 to 30% over mlx-lm, 42 to 60% over exo's adapter)
are in line with the real-task ones above, the prefill lead is not.

## Limits

One Mac, one request at a time, greedy, 32 generated tokens, text from one
repository. exo is exo's server around a local adapter, warmed on the timed
prompt, with one round. The 4,096-chunk MLX arm has two rounds. Token IDs
differ between engines on a third to a half of the tasks. `relative_stripes`
needs head size 128 and three query heads per KV head (other shapes keep the
existing kernels) and its mirrors add 50% to K and V memory. Nothing here
measures sampling, serving, streaming or multi-node exo.

## Reproduce

```sh
python3 tools/nano_inkling_tasks.py build "$CHECKPOINT" tasks.json   # needs transformers
python3 tools/nano_inkling_tasks.py run zig-out/bin/nano "$CHECKPOINT" tasks.json r1.json \
  --mlx-python "$MLX_PYTHON" --lock "$NANO_LOCK" --engines native,mlx
python3 tools/nano_inkling_tasks.py run zig-out/bin/nano "$CHECKPOINT" tasks.json r4.json \
  --mlx-python "$MLX_PYTHON" --lock "$NANO_LOCK" --engines native,mlx --mlx-chunk 4096
python3 tools/nano_inkling_tasks_summary.py tasks.json summary.json r1.json r4.json
python3 tools/guarded_run.py --lock "$NANO_LOCK" --timeout 300 -- \
  zig-out/bin/nano attnbench 20 1024 4096 32768 rows=4096
```

`build` reads the repository it is run in, so prompts built at another commit
contain different text. The knob list is `KNOBS` in the tool. The exo arm
(`--engines native,mlx,exo --exo-runner ...`) needs the pinned exo checkout, its
environment and the local adapter; they are not part of this repository.
