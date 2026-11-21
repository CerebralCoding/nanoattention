# Liquid decode: append-only KV specialization

`nano` can run the dense LFM2.5-1.2B-Instruct MLX 4-bit checkpoint through the same IR as LFM3. LFM2.5 uses ungated full causal GQA (32 query heads, 8 KV heads, 64 dimensions), two prenorm residuals per layer, and a tied output head. Only the raw-token commands are covered here; the native chat template is still LFM3-only. The upstream BF16 checkpoint was not tested and has a different convolution weight shape.

`--knob decode_attention=3` specializes the existing exact split decode attention for append-only KV, four query heads per KV head, and 64 or 128 dimensions. It removes modulo addressing and per-dimension bounds checks, and uses direct addressing for complete value blocks. The score accumulation, per-SIMD recurrence, and final merge order are unchanged. Ring caches and other shapes still use the old path. Defaults are unchanged.

## 4,096-token greedy decode

Apple M3 Ultra, 60 GPU cores, 256 GiB RAM. Native ReleaseFast with the `exact` preset and FP32 activations/KV. Stock MLX 0.32.4 with pipelined BF16 activations, MLX-VLM 0.8.0 for LFM3 and MLX-LM 0.32.2 for LFM2.5. Each native/MLX pair uses the same 4-bit affine checkpoint and input token IDs. Each run fills the entire context and times 128 greedy forwards; the engines are run separately in three interleaved rounds, three timed repetitions per process after warmup. Medians in tokens/s:

| Checkpoint | Native split (2) | Native append-only (3) | Stock MLX, pipelined |
| --- | ---: | ---: | ---: |
| `LiquidAI/LFM3-1.7B-MLX-4bit` (`91f34088d694`) | 239.85 | **247.30** | 228.55 |
| `mlx-community/LFM2.5-1.2B-Instruct-4bit` (`dee2f8a2786e`) | 321.92 | 337.92 | **391.57** |

LFM3's candidate outran MLX in all three rounds, but improved on the native baseline in only two. LFM2.5's candidate beat the old native path in all three rounds and still trailed MLX. Prefill is not a win: native median ~1,478 / 1,985 tok/s versus MLX ~4,551 / 6,868 respectively. This is a shared machine and the results are workload-specific.

In a second LFM2.5 session, the retained 32-score chain reached 389.81 tok/s and a 16-score variant reached 394.94 tok/s, versus 477.16 tok/s for pipelined MLX. The smaller block lost one of three head-to-head rounds and the median gain was within run-to-run variation, so it was removed. A two-round screen of existing matvec tiles did not produce a consistent win either.

## Correctness and attribution

- Headless Metal tests compare all output bits against the generic attention kernel at both full shapes, gated and ungated, around partial and complete blocks. Existing sliding and ragged cases still exercise the fallback.
- For each full checkpoint, native variants 2 and 3 matched all vocabulary logits bit for bit: the last prefill position and 32 single-token forwards on 64- and 4,096-token prefixes.
- LFM2.5 also matched stock FP32 MLX top-1 at all 66 checked positions, with max absolute logit difference 0.000084 at 64 and 0.000149 at 4,096. This is not BF16 parity or a broad generation-quality claim.
- A separate 32-token LFM2.5 profile with GPU gating disabled for attribution showed the two attention kernels falling from ~635 to ~491 µs per token. Quantized linear and fused MLP kernels took ~2,159 µs per token after the change. Profiling changes launch scheduling; these times are not throughput or an exact upper bound for the optimized schedule.

## Projection and MLP tuning

`decode_q4_static=true` specializes the existing exact decode tile for fixed input widths 2,048 / 7,680 / 8,192, group-64 affine INT4, and BF16 scale/bias metadata. This also covers fused projections and SwiGLU where applicable. `fuse_add_norm=true` fuses an eligible 2,048-wide residual addition and following RMSNorm, storing the original residual and preserving reduction order. Neither setting changes the attention math, the precision, or the default preset.

The q4 specialization by itself reached 430.40 vs 394.32 tok/s for LFM2.5 and 307.67 vs 283.51 for LFM3 in same-session comparisons to the append-only attention baseline. Both models matched all native logits bit for bit at the last prefill position and 32 single-token forwards at both 64 and 4,096 prompt tokens. The combined candidate passed the same checks and `zig build test-mega -Doptimize=ReleaseFast`.

The combined confirmation used alternating engine order (five LFM2.5 rounds, three LFM3 rounds), with the same counts, weights and prompt IDs as above:

| 4,096 prompt tokens | Native decode (tok/s) | Pipelined MLX decode (tok/s) | Native / MLX prefill (tok/s) |
| --- | ---: | ---: | ---: |
| LFM3-1.7B | **313.25** | 278.23 | 1,746 / 5,467 |
| LFM2.5-1.2B | 470.10 | **482.73** | 2,335 / 8,143 |

LFM3 won all three rounds. LFM2.5 won two rounds in a different three-round trial, but lost all five in the confirmation. In the latest LFM2.5 run the retained schedule reached 474.02 vs 482.21 tok/s; 64-thread and 256-thread linear groups reached 474.36 and 383.41, with no repeatable gain. This is not a verified LFM2.5 win. Packed-halfword loads, smaller attention blocks, down-projection retile, skipping unit rescales, and a fixed-trip matvec loop did not show consistent gains and were removed.

## Reproduce

Build and test with `zig build -Doptimize=ReleaseFast` and `zig build test-mega -Doptimize=ReleaseFast`. With `CKPT` set to one local checkpoint, `NANO_MLX_PYTHON` pointing to the reference Python, and `NANO_LOCK` pointing to an existing inference lock file:

```sh
mkdir -p target
"$NANO_MLX_PYTHON" -I tools/nano_bench.py tokens "$CKPT" \
  --baseline target/no-legacy-baseline.json --out target/liquid-tokens.json
python3 -c 'import json; k=json.load(open("src/mega/presets/exact.json")); k.update(decode_attention=3, decode_q4_static=True, fuse_add_norm=True); json.dump(k, open("target/liquid-tuned.json", "w"))'
python3 tools/nano_bench.py versus zig-out/bin/nano "$CKPT" \
  exact=src/mega/presets/exact.json tuned=target/liquid-tuned.json \
  --bench-tokens target/liquid-tokens.json --lengths 4096 --decode-tokens 128 \
  --greedy --mlx mlx_bf16_async --rounds 3 --repeats 3 --output target/liquid-bench.json
```
