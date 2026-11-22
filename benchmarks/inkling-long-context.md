# Inkling 2 Mini at 4K to 32K: twelve real tasks, cold cache

Twelve different prompts built from real text (Zig source, Python tools, Markdown
documentation, each at 4,096, 8,192, 16,384 and 32,768 prompt tokens, each with
its own instruction), every one timed from an empty cache. With the opt-in knobs
in [docs/MEGAKERNEL.md](../docs/MEGAKERNEL.md#inkling-2-mini-experimental):

- **Decode:** native is 29 to 36% faster than the pinned mlx-lm reference and 53
  to 63% faster than exo's MLX adapter, in every paired sample.
- **First token against exo's adapter:** native is 29 to 31% sooner, in every
  pair.
- **First token against mlx-lm at 4,096-token chunks, its best size here:**
  native is 2.7 to 4.6% sooner, in all 36 pairs. The narrowest pair is 0.6%.
- **First token against mlx-lm at its default 2,048-token chunks:** native is
  7.6 to 9.6% sooner, in 46 of 48 pairs. Both losses are in one round in which
  both engines ran up to 20% slow.

The lead over 4,096-chunk mlx-lm narrows with context (4.6% at 4K, 2.7% at 32K).
Nothing here was measured past 32K.

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
the decode rate over the 31 single-token forwards after it. Greedy. Nine rounds,
each native against one other engine: mlx-lm at 4,096-token chunks in rounds 1
to 3, at its default 2,048 in rounds 4, 5, 7 and 8, exo's adapter in rounds 6
and 9. Every sample of every round is in the file; none was dropped.

The Mac was not idle. Unrelated compile jobs ran on its CPU throughout (load
average 8 to 14 around rounds 7 to 9, the only ones where it was recorded).
Round 4 was hit hardest: native took 34.10 s on a task it takes 28.1 s on in the
other rounds, mlx-lm 35.28 s against 30.3 s.

## Results

Medians over a length's three tasks and all rounds: seconds to first token /
decode tokens a second. The last three columns are native's first-token time and
decode rate relative to that engine (negative time is sooner).

| Prompt | Native | MLX, 2,048 chunk | MLX, 4,096 chunk | exo adapter | vs MLX 2,048 | vs MLX 4,096 | vs exo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 3.13 s / 39.6 | 3.46 s / 29.1 | 3.28 s / 29.5 | 4.45 s / 25.4 | -9.6% / +36% | -4.6% / +34% | -29.7% / +56% |
| 8,192 | 6.36 s / 39.2 | 6.97 s / 29.0 | 6.61 s / 29.5 | 8.91 s / 25.6 | -8.8% / +36% | -3.8% / +33% | -28.6% / +53% |
| 16,384 | 13.19 s / 38.3 | 14.46 s / 28.8 | 13.56 s / 29.4 | 18.52 s / 24.0 | -8.8% / +33% | -2.8% / +30% | -28.8% / +59% |
| 32,768 | 28.13 s / 36.4 | 30.44 s / 27.6 | 28.90 s / 28.2 | 40.54 s / 22.3 | -7.6% / +32% | -2.7% / +29% | -30.6% / +63% |

| Against | Paired samples | Native sooner to first token | Native faster decode | Tasks with the same first 31 IDs |
| --- | ---: | ---: | ---: | ---: |
| MLX, 2,048 chunk | 48 | 46 | 48 | 8 of 12 |
| MLX, 4,096 chunk | 36 | 36 | 36 | 6 of 12 |
| exo adapter | 24 | 24 | 24 | 7 of 12 |

The narrowest pair against 4,096-chunk MLX is python-16384 in round 2: 13.68 s
against 13.76 s, 0.6%. The two losses against 2,048-chunk MLX are both in round
4: zig-8192 (7.64 s against 7.35 s) and python-16384 (15.65 s against 15.56 s).
Pairs are within a round. Across rounds the claim is weaker: native's slowest
sample of a task (round 4 for eight of them) is behind 4,096-chunk MLX's fastest
on all twelve.

mlx-lm's decode medians are a few percent under the previous version's rounds
(27.6 to 29.5 here, 28.3 to 30.6 there), most likely the CPU load above, and its
slow samples (14 to 20 tokens a second, seven of 84) are in the file. Against
the previous rounds' medians native's decode lead would be 28 to 30%.

Per task, medians of that task's samples (native nine, MLX four and three, exo
two):

| Task | Native | MLX, 2,048 chunk | MLX, 4,096 chunk | exo adapter |
| --- | ---: | ---: | ---: | ---: |
| zig-4096 | 3.10 / 39.6 | 3.44 / 29.1 | 3.27 / 28.8 | 4.46 / 25.5 |
| zig-8192 | 6.33 / 39.2 | 6.97 / 28.7 | 6.56 / 29.7 | 8.89 / 25.7 |
| zig-16384 | 13.14 / 38.2 | 14.73 / 29.3 | 13.67 / 29.1 | 18.50 / 20.7 |
| zig-32768 | 28.13 / 36.4 | 30.57 / 28.1 | 28.92 / 27.9 | 40.89 / 21.8 |
| python-4096 | 3.12 / 39.6 | 3.48 / 29.4 | 3.28 / 29.5 | 4.77 / 23.3 |
| python-8192 | 6.40 / 39.3 | 7.02 / 29.7 | 6.62 / 28.1 | 9.01 / 24.2 |
| python-16384 | 13.14 / 38.3 | 14.59 / 26.3 | 13.76 / 29.4 | 18.53 / 24.2 |
| python-32768 | 28.06 / 36.6 | 30.44 / 26.8 | 28.85 / 28.7 | 40.71 / 22.2 |
| docs-4096 | 3.13 / 39.4 | 3.46 / 27.5 | 3.29 / 30.1 | 4.45 / 25.5 |
| docs-8192 | 6.36 / 38.9 | 7.02 / 28.1 | 6.63 / 29.8 | 8.91 / 25.0 |
| docs-16384 | 13.19 / 38.4 | 14.37 / 29.3 | 13.54 / 29.4 | 18.65 / 18.7 |
| docs-32768 | 28.14 / 36.3 | 30.82 / 26.7 | 28.90 / 28.2 | 40.51 / 22.5 |

**Outputs.** Native produced the same 32 IDs for a task in all nine rounds, and
they are the IDs the previous version's build produced. The engines agree on the
first 31 IDs for only some tasks (table above), and mlx-lm's own output changes
with its chunk size on five of them. Every reply decodes to on-topic reasoning
text about the task; where they differ they differ in wording a few tokens in
("Need answer review of Zig code. Need inspect carefully. User asks…" against
"…Need list three most likely bugs…"). So these rows compare speed on outputs
that are not always the same tokens.

### Chunk sizes

Each engine was swept before the rounds, on one or two tasks:

| Engine | Chunk | Result |
| --- | ---: | --- |
| native | 128 | 606 tok/s prefill, python-16384 |
| native | 512 | 924 tok/s, docs-4096 |
| native | 4,096 | 1,309 tok/s (3.13 s), 4K tasks |
| native | 8,192 | does not fit: scratch offsets are 32-bit words (`ArenaTooLarge`) |
| mlx-lm | 1,024 | 31.71 s to first token, zig-32768 |
| mlx-lm | 2,048 (default) | about 30.4 s |
| mlx-lm | 2,560 / 3,584 | 30.13 s / 29.30 s |
| mlx-lm | 4,096 | 29.04 s |
| mlx-lm | 8,192 | 29.17 s |
| exo adapter | 2,048 | 4.52 s docs-4096, 18.58 s python-16384 |
| exo adapter | 4,096 | 4.89 s, 20.27 s |

Native could not take batches above 512 rows before `scratch_reuse` (the
scratch arena exceeded its limit). The mlx-lm and exo sweeps are the previous
version's; mlx-lm's medians at both chunk sizes are within 0.5% of what they
were then.

### Where native's prefill time goes

A profiled 32K prefill in 4,096-row chunks: `mk_expert_block_up_nv` 33%,
`mk_bf16_prefill_matrix32` 19%, `mk_relative_prefill_tiles` 17%,
`mk_expert_block_down_nv` 17%. Per chunk native takes 3.13 s at 4K and 3.52 s at
32K, 4,096-chunk MLX 3.28 s and 3.61 s. Fitting the four lengths, native's cost
per chunk is about 5% under MLX's and its cost that grows with context
(global-layer attention) still about 19% over, which is why the lead narrows.
On that fit the two would meet near 70K tokens; that is an extrapolation, not a
measurement.

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

Added since the first real-task version of this file (which had native 0.5 to
4.4% behind 4,096-chunk MLX); together they took docs-4096 from 3.31 s to 3.13 s
and zig-32768 from 30.25 s to 28.13 s in back-to-back runs, with the same
output IDs:

- **`tail_trim=512`**: row liveness (`exec.rowNeeds`). Walking the program
  backwards, each op gets the rows something still reads. Nothing reads most
  rows of the last layer's attention output and feed-forward: the caches after
  them keep a 4-tap convolution's last three rows and the head reads the rows
  with logits. Those ops run on the chunk's last 512 rows instead of 4,096.
  Every K, V and convolution row that is stored is still computed. 512 rather
  than a handful because the tuned kernels are chosen by row count.
- **`relative_bias16`**: a layer's distance bias (24 heads x 1,024 distances a
  row, rounded to bfloat16, read only by that layer's attention) is written and
  read as 16-bit words.
- **`relative_bias_d8`**: that projection with two queries and eight distances
  a thread, each input loaded once instead of once per distance.
- **`conv_ring`**: the short convolutions step the ring slot from tap to tap
  instead of taking a 64-bit remainder for every tap of every channel.
- **`expert_lists_split`**: each expert's route list from 32 segments scanned
  at once instead of one scan end to end.
- **Not a knob:** the loop controllers' two normalized halves and two gate
  projections are written straight into the vector that is read next, instead
  of being copied there. 288 fewer launches a forward (3,043 to 2,755 for a
  prefill chunk, 2,995 to 2,707 for a decode token), which is where decode's
  gain from 37.8 to 39.6 tokens a second at 4K comes from. The default program
  has fewer ops; every value is computed as before.
- `mk_relative_prefill_tiles` carries a tile's ring slot from tile to tile
  instead of dividing for every staged vector.

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
the three query heads of a KV head. 8,192-row batches do not fit (above).
`tail_trim=1` is a few milliseconds from `tail_trim=512` and changes bits.

## Correctness

- `zig build test` and `zig build test-mega` pass. Kernel tests compare the
  striped kernels with `mk_relative_attention` over mirrors written by
  `mk_pack_bf16` (a wrapped sliding window, uneven stripes, more stripes than a
  row tile has tiles, a partial query tile, a one-key window), and the staged
  tiles with `mk_relative_attention_prefill` (bit-identical on an unwrapped
  cache, within 1e-4 on a wrapped window). `packScratch` has its own test.
- **Bit-identical to the path they replace:** `scratch_reuse` (logits at feeds
  512 and 1) and `expert_tile_rows`.
- **The five newer knobs and the controller change, together, are
  bit-identical** to the previous build with the previous knobs: all-vocabulary
  logits of python-8192 fed in 4,096-row chunks at positions 4,095, 6,000 and
  8,191, the last one after a chunk in which the trim applied. Each has a kernel
  or engine test against the path it replaces (`rowNeeds` on the lowered
  program; trimmed against untrimmed logits on the fixture for a chunk without
  logits, a chunk with them and a single row; the packed and eight-distance
  bias; the ring-stepped convolutions on a wrapped ring; the segmented route
  lists). `tail_trim=1` is not bit-identical with the tuned knobs (the last
  position's logits move by up to 0.10, same top token): a tail under 512 rows
  goes through the smaller-batch kernels.
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

## Superseded: the first real-task version

The previous version of this file (same tasks, same protocol, five rounds) had
native 0.5 to 4.4% later to the first token than 4,096-chunk mlx-lm, winning 2
of 24 pairs. Its samples are kept in
[`inkling-long-context-previous.json`](inkling-long-context-previous.json).
mlx-lm's first-token medians there (3.28, 6.60, 13.62, 28.96 s) are the ones
measured again here (3.28, 6.61, 13.56, 28.90 s); native's moved from 3.30,
6.74, 14.06 and 30.24 s to 3.13, 6.36, 13.19 and 28.13 s.

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
repository, prompts of at most 32,768 tokens. The machine was shared with
unrelated compile jobs while these rounds ran. The 4,096-chunk MLX arm has
three rounds and its lead is 2.7% at 32K, narrowing with length. exo is exo's
server around a local adapter, warmed on the timed prompt, with two rounds.
Token IDs differ between engines on a third to a half of the tasks.
`relative_stripes` needs head size 128 and three query heads per KV head (other
shapes keep the existing kernels) and its mirrors add 50% to K and V memory.
Nothing here measures sampling, serving, streaming or multi-node exo.

## Reproduce

```sh
python3 tools/nano_inkling_tasks.py build "$CHECKPOINT" tasks.json   # needs transformers
python3 tools/nano_inkling_tasks.py run zig-out/bin/nano "$CHECKPOINT" tasks.json r1.json \
  --mlx-python "$MLX_PYTHON" --lock "$NANO_LOCK" --engines native,mlx --mlx-chunk 4096
python3 tools/nano_inkling_tasks.py run zig-out/bin/nano "$CHECKPOINT" tasks.json r4.json \
  --mlx-python "$MLX_PYTHON" --lock "$NANO_LOCK" --engines native,mlx
python3 tools/nano_inkling_tasks_summary.py tasks.json summary.json r1.json r4.json
python3 tools/guarded_run.py --lock "$NANO_LOCK" --timeout 300 -- \
  zig-out/bin/nano attnbench 20 1024 4096 32768 rows=4096
```

`build` reads the repository it is run in, so prompts built at another commit
contain different text. The knob list is `KNOBS` in the tool. The exo arm
(`--engines native,mlx,exo --exo-runner ...`) needs the pinned exo checkout, its
environment and the local adapter; they are not part of this repository.
