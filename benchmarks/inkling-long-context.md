# Inkling 2 Mini: striped relative attention at 4K to 32K

`--knob relative_stripes=true` (on top of the tuned Inkling knobs in
[docs/MEGAKERNEL.md](../docs/MEGAKERNEL.md#inkling-2-mini-experimental)) changes how
relative attention reads its K and V caches. Before it, native decode fell from 31
to 23 tokens a second between 4K and 32K prompt tokens and lost to the pinned
mlx-lm reference at 32K; native prefill trailed it from 8K up. With it, native
leads both references at every measured length, on decode and on prefill.

One Apple M3 Ultra (60 GPU cores, 256 GB), the original
`thinkingmachines/Inkling-2-Mini` checkpoint (`ed40e7da625f`), `nano` built
ReleaseSafe. Seeded token IDs, a cold cache, 128-token prefill chunks, then 25
greedy tokens; decode rates time the 24 intervals after the first. Each engine
runs in its own process under one inference lock, one warmup then one timed run a
process, engine order alternating between rounds. Tokens a second, medians.
Raw samples: [`inkling-long-context.json`](inkling-long-context.json).

## Against mlx-lm, run directly

`tools/nano_inkling_bench.py --decode-mode greedy`; `mlx==0.32.4`, `mlx-lm` at
`0c4c99192230`. MLX enqueues the next step before reading back the previous token.

| Prompt | Native decode | MLX decode | Lead | Native prefill | MLX prefill | Lead | Rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 36.61 | 28.36 | +29% | 710.7 | 676.7 | +5.0% | 5 |
| 8,192 | 36.00 | 28.65 | +26% | 695.0 | 662.8 | +4.9% | 5 |
| 16,384 | 35.16 | 28.89 | +22% | 672.6 | 640.3 | +5.0% | 5 |
| 32,768 | 33.99 | 26.21 | +30% | 630.0 | 597.7 | +5.4% | 2 |

Native won decode in all 17 rounds and prefill in 16; the one prefill loss was a
16K round where native took 26.1 s instead of its usual 24.3 s. At 8K, 16K and
32K both engines produced the same 25 token IDs in every round. **At 4K they did
not**: MLX's IDs differ from native's from the first generated token, as they did
before this change, so the 4K row compares speed on different outputs.

The same workload before the change (same checkpoint, machine and protocol, two
rounds, an earlier session): native decode 31.33 / 29.69 / 26.89 / 22.78 and
prefill 670.9 / 640.9 / 599.1 / 531.6, against MLX decode 30.35 / 29.94 / 27.17 /
27.57 and prefill 679.0 / 664.4 / 641.4 / 599.5. MLX's decode figures move by a
few percent between sessions (27.57, then 26.21, at 32K).

## Against exo's MLX adapter

exo `a60b95698ca8` serving a local text-only MLX-VLM adapter (`434a1bedd1e8`) with
`--no-batch` and direct cold prefill, through its HTTP benchmark endpoint; token
IDs and timing are recorded inside its request runner. Stock exo does not load
this checkpoint, so this is not a comparison with an exo release. "First token"
is prompt execution to the first generated token; the two prefills hand off their
last two tokens differently.

| Prompt | Native decode | exo decode | Lead | Native first token | exo first token | Pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 36.50 | 25.27 | +44% | 5.76 s | 6.86 s | 2 |
| 8,192 | 35.72 | 25.09 | +42% | 11.81 s | 13.77 s | 2 |
| 16,384 | 35.09 | 24.54 | +43% | 24.43 s | 28.66 s | 3 |
| 32,768 | 33.69 | 21.01 | +60% | 52.48 s | 64.51 s | 2 |

Native won decode and first token in all nine pairs, and all 25 token IDs matched
in every pair at every length. exo's 16K decode samples were 24.54, 16.00 and
24.65; the third pair was added because of the slow one, and the median keeps all
three.

## What changed

Every K and V row Inkling stores is a bfloat16 value held in a float. With the
knob, each store also writes the row's upper 16 bits to a mirror, and attention
reads the mirror: the same numbers at half the memory traffic.

- **Decode** (`mk_relative_long_keys`, `mk_relative_long_merge`): the keys a row
  sees are cut into stripes of 128 to 512, one threadgroup of 16 SIMD groups per
  (KV head, stripe). The three query heads of a KV head share every key and value
  load. A stripe is reduced against its own maximum and the stripes are merged
  against the global one.
- **Prefill** (`mk_relative_prefill_stripes`, `mk_relative_prefill_merge`): the
  16-key tiles of `mk_relative_attention_prefill`, cut into two stripes of keys
  for a sliding window and four for a global layer, with the K and V fragments
  loaded straight from the mirrors as bfloat16 matrices instead of being staged
  through threadgroup memory as FP16.

`nano attnbench` times one layer's attention on synthetic K and V. On this
machine, a single decode row over a global layer went from 318 to 60 us at 4,096
keys and from 2,234 to 305 us at 32,768; over a full sliding window from 100 to
27 us. A 128-row prefill chunk went from 1,740 to 895 us at 4,096 keys, from
14,018 to 6,487 us at 32,768, and from 506 to 276 us on a sliding window. A
decode-only profile at a 4K prompt put attention at 17% of GPU time before the
change, so most of a token is still the dense projections, experts and norms.

## Correctness

- `zig build test` and `zig build test-mega` pass. The new kernel tests compare
  the striped kernels with `mk_relative_attention` over mirrors written by
  `mk_pack_bf16`: a wrapped sliding window, uneven stripes, more stripes than a
  row tile has tiles, a partial query tile, and a one-key window (decode within 1e-5, prefill
  within 1e-4 of the tiled kernel).
- The knob is **not bit-identical** to the paths it replaces: the dot products
  and the softmax sum are associated differently. On a 420-token chat transcript
  (`tools/nano_inkling_attention_check.py`, all-vocabulary logits at 24
  positions), striped decode differs from split decode by at most 3.53 (RMS
  0.36) and striped prefill from tiled prefill by at most 3.06 (RMS 0.38), with
  the top token equal at 24 of 24 positions. The existing paths differ from each
  other by as much on the same transcript: the reference kernel against split
  decode by 4.69 (RMS 0.39), split decode against tiled prefill by 5.73 (RMS
  0.45). Small differences are amplified by bfloat16 rounding and expert routing.
- On the seeded fixtures the 25 greedy IDs are those of the previous tuned path
  at all four lengths. On a real prompt they are not: a greedy reply to "Write an
  LRU cache in Python..." matched the previous path's for its first 20 tokens and
  then took different wording (639 tokens instead of 744, both ending on EOS,
  both correct code).
- MLX logits parity for Inkling remains unestablished, as before.

## Limits

One Mac, one request at a time, seeded prompts, 25 generated tokens, no sampling.
The knob needs head size 128 and three query heads per KV head (Inkling's
geometry); other shapes fall back to the existing kernels. The mirrors add 50% to
K and V memory. The 32K MLX row has two rounds. Nothing here measures serving, streaming
or multi-node exo.

## Reproduce

```sh
python3 tools/nano_inkling_bench.py zig-out/bin/nano "$CHECKPOINT" \
  --mlx-python "$MLX_PYTHON" --lock "$NANO_LOCK" --prompt 16384 --batch 128 --decode 24 \
  --decode-mode greedy --rounds 3 --timeout 720 \
  --knob expert_simd=true --knob expert_wide_dot=true --knob expert_gate_pair=true \
  --knob expert_prefill=true --knob expert_wide_k=true --knob expert_bf16_tiles=true \
  --knob expert_mlx_scale=true --knob bf16_linear_tile=true --knob bf16_prefill_tile=8 \
  --knob bf16_prefill_matrix=true --knob bf16_decode_vec=true --knob fuse_bf16_cast=true \
  --knob concurrent_scratch=true --knob relative_split=true --knob relative_split_4=true \
  --knob relative_tile_prefill=true --knob relative_stripes=true
python3 tools/guarded_run.py --lock "$NANO_LOCK" --timeout 300 -- zig-out/bin/nano attnbench 20 1024 4096 32768
python3 tools/guarded_run.py --lock "$NANO_LOCK" --timeout 900 -- \
  python3 tools/nano_inkling_attention_check.py zig-out/bin/nano "$CHECKPOINT" ids.json /tmp/attention-check
```

The exo arm needs the pinned exo checkout, its environment and the local adapter;
it is not part of this repository.
