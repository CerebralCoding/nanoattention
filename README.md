<h1 align="center">nanoattention</h1>

<p align="center"><strong>Flint inference, close to the metal.</strong></p>

<p align="center">A native Zig + Metal engine for Apple Silicon.<br>Explicit memory. Persistent KV caches. Lean-checked execution contracts.</p>

<p align="center">
  <img alt="Apple Silicon macOS" src="https://img.shields.io/badge/platform-Apple%20Silicon-555">
  <img alt="Built in Zig 0.17" src="https://img.shields.io/badge/built%20in-Zig%200.17-f7a41d?logo=zig&logoColor=white">
  <img alt="GPU backend Metal" src="https://img.shields.io/badge/GPU-Metal-555">
  <img alt="Original code MIT licensed" src="https://img.shields.io/badge/original%20code-MIT-44cc11">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#use-with-codegraff">CodeGraff</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#correctness-and-proofs">Proofs</a> ·
  <a href="#development">Development</a>
</p>

Run [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) locally with a
small, inspectable inference stack. Zig maps the checkpoint, owns buffers and KV
state, and schedules the model. Metal kernels execute attention, packed quantized
projections, expert routing, and the vocabulary head on the GPU.

**The native runtime does not link MLX, PyTorch, or CUDA.** Optional Python tools
handle tokenization, chat templates, and the local HTTP bridge—not model math.
This is a focused, single-sequence inference engine, not a general tensor framework.

## Quick start

### 1. Build the engine

You need an Apple Silicon Mac, Xcode or Command Line Tools with the macOS SDK,
**Zig 0.17.0**, **Lean/Lake 4.35.0**, and **Python 3.9+**. The Lean project uses
`Std` only. Builds check the generated proof bindings before compiling the engine.

```sh
git clone https://github.com/justrach/nanoattention.git
cd nanoattention
zig build -Doptimize=ReleaseSafe
```

This repository is private; cloning requires access. The executable is
`zig-out/bin/flint`. There is no runtime or model auto-install step.

### 2. Get the weights

Flint is developed by **Recursive**. The official model and original BF16
checkpoint are on [Hugging Face: RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint).
The weights retain Recursive's **[Apache-2.0 license](https://huggingface.co/RecursiveSI/flint/blob/main/LICENSE)**;
we do not relicense them under this engine's MIT license.

[codegraff.com/flint](https://codegraff.com/flint) is an **unofficial download mirror**
for the tested, modified MLX-style affine-INT4 export—not a byte-for-byte mirror of
the original BF16 checkpoint or a model authored by us. It retains the same
upstream Apache-2.0 terms, original license, attribution, and modification notice;
no endorsement by Recursive is implied.

The mirror includes all five weight shards, tokenizer, chat template,
configuration, license, and checksums. Allow roughly **22.58 GB of disk space**,
plus enough unified memory for weights, working buffers, and your chosen context.
The download does not install or start an inference server.

Point the commands below at the directory containing the complete checkpoint:

```sh
./zig-out/bin/flint inspect /path/to/flint-int4
```

`inspect` checks model configuration; it is not a full checkpoint integrity check.
Use the download's checksums to verify the files.

### 3. Generate text

In a Python environment with `transformers` and `tokenizers` installed:

```sh
python3 -B tools/chat.py /path/to/flint-int4 \
  "Write a Python binary search function." \
  --context 2048 --max-tokens 64 --staged-row-tile 2
```

The helper uses only the checkpoint's local tokenizer and chat template. All
inference runs in the native executable. Generation is greedy.

<details>
<summary>Work directly with token IDs</summary>

Create `prompt.json` as an array of IDs produced by the checkpoint's tokenizer,
then run:

```sh
./zig-out/bin/flint generate /path/to/flint-int4 \
  --tokens-file prompt.json --context 2048 --max-tokens 64 --staged-row-tile 2
```

The result is a JSON object containing generated `tokens`. EOS is included when
reached. `--context` bounds prompt plus generated positions. Raw text is not a
valid token-ID prompt. `flint eval` exposes vocabulary logits for diagnostics.

</details>

## What it does

- **Native quantized inference.** Load mmap-backed safetensors, including sharded
  MLX affine INT4/INT8 exports, without expanding the full checkpoint to FP32.
- **GPU-side model work.** Packed projections, fused QKV launches, routed/shared
  experts, normalization, attention, and greedy selection use Metal kernels.
- **Persistent conversations.** A JSONL session keeps weights and per-layer KV
  alive. Compatible follow-up turns reuse the exact cached token prefix; changed
  histories trigger an explicit reset rather than a pretend cache hit.
- **Long-context attention.** Sliding layers use bounded ring caches; global
  layers retain the full configured prefix. Flint applies RoPE to sliding layers
  and NoPE to global layers.
- **Checked kernel contracts.** Lean proves selected integer scheduling and
  indexing properties; generated bindings connect those contracts to Zig and Metal.
- **Measured experiments.** Optional BF16 global KV, history-based speculation,
  and tuning candidates are separate from the ordinary FP32-KV decode baseline.

The loader also supports F32/F16/BF16 tensors and mixed-precision embeddings.
NVIDIA NVFP4 is not supported. The tested INT4 export has received more full-model
validation than the published BF16 checkpoint.

## Use with CodeGraff

Keep the model local while CodeGraff reads files, runs tools, and handles the coding
conversation. The owned launcher starts one loopback adapter and one persistent
native worker, then selects `nanoattention/flint` for the chosen workspace.

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1400 -- \
  python3 -B tools/graff_native.py /path/to/flint-int4 \
  --graff /path/to/graff --workspace /path/to/coding-workspace \
  --context 32768 --max-tokens 4096 --timeout 1200 -- \
  -p "Add tests for the parser, then fix any failures."
```

Use the tokenizer-enabled Python environment. Omit the arguments after `--` for
an interactive terminal session. Run local inference jobs sequentially under the
same exclusive lock.

Compatible Graff builds use the built-in local provider. Older binaries use a
temporary project router, without overwriting an existing router or changing global
provider settings. The launcher cleans up its owned processes on exit. Normal
Graff tool permissions still apply; the launcher is not itself an OS sandbox.

The Python bridge provides OpenAI-compatible chat and function-tool translation;
**it never executes tools**. It accepts one active model request at a time, and
SSE output is buffered rather than token-live. See the
[setup, lifecycle, and verification guide](docs/CODEGRAFF.md).

## Performance

Decode speed depends on context, kernel configuration, and workload. The benchmark
reports retain configurations, methodology, negative results, and correctness gates
rather than treating the fastest run as a universal speed claim.

| Workload | Recorded native decode rate | Evidence |
| --- | ---: | --- |
| Short coding prompt, 21-token prefix / 64 teacher-forced steps | 120.40 tok/s | [Matched local MLX comparison](benchmarks/README.md) |
| Seeded random raw IDs, filled 16K / 256 ordinary steps | 106.09 tok/s | [Raw decode experiments](benchmarks/raw-decode.md) |
| Seeded random raw IDs, filled 32K / 256 ordinary steps | 93.19 tok/s | [Raw decode experiments](benchmarks/raw-decode.md) |

These are historical measurements on an **Apple M3 Ultra, 60 GPU cores, 256 GB
unified memory**, with the INT4 checkpoint and native FP32 activations/KV. Loading,
prefill, and warmup are excluded. Rows come from different trials/configurations;
they are not a paired comparison or a fresh benchmark of this documentation change.
Raw teacher-forced rates are not free-running coding-agent throughput.

The short-context matched trial measured stock MLX FP32 at 96.84 tok/s and BF16 at
91.96 tok/s. That establishes only that workload: native serial prefill remains
substantially slower than MLX's batched prefill. Recursive's published `rec`
numbers use different hardware and protocols; **this is not a matched win over
`rec`, overall MLX parity, or a claim of 130 tok/s at long context**.

More: [long-context sessions](benchmarks/larger-context-sessions.md),
[raw attention tuning](benchmarks/raw-attention-hillclimb.md), and
[prepared-input experiments that did not improve speed](benchmarks/raw-prepared-inputs.md).

<details>
<summary>Experimental modes</summary>

- **BF16 global KV** reduces global-cache storage, while sliding KV and model math
  remain FP32. It can change logits and generated tokens; it is not stock MLX BF16
  inference. See [measurements](benchmarks/bf16-kv.md).
- **Greedy speculation** uses committed-history drafts and batched target
  verification with sliding-cache rollback. It is disabled by default, preserves
  target greedy tokens in the tested cases, and can be slower on real coding
  tasks. See [contracts and limitations](docs/SPECULATIVE.md) and
  [workload-dependent results](benchmarks/speculative-hillclimb.md).
- **Resident execution** interprets the generated plan in one GPU dispatch, with
  `--resident-workers 1..32` and checked dimension/context limits. It is slower than
  the optimized staged path. Ordinary staged execution uses multiple kernels;
  one command-buffer submission is not one kernel.

</details>

## Correctness and proofs

Correctness is checked at several levels: headless Metal tests, synthetic
checkpoints against independent NumPy, bounded full-model MLX comparisons, and
Lean-generated binding checks. Numerical agreement is reported with its tested
precision, context, and tolerance—not as a blanket quality or bitwise-MLX claim.

For long-context coding, recorded sequential FP32 MLX checks cover every greedy
choice through 28,623 cached tokens and six full-vocabulary boundary rows, not
all logits at every position. Raw kernel candidates use separate bitwise admission
against the ordinary native reference. Neither establishes full-window task quality.

**Lean proves selected integer properties:** execution-plan order, guarded coverage,
index bounds, packed INT4 reconstruction, and launch geometry. Builds verify exact
bindings consumed by Zig and Metal. **It does not prove floating-point equivalence,
whole-shader correctness, GPU memory semantics, forward progress, or speed.**
See [proof scope and assumptions](proofs/README.md).

## Development

Run from the repository root. NumPy is needed for synthetic numerical checks;
MLX/MLX-LM are optional reference-test dependencies, not runtime dependencies.

```sh
zig build test -Doptimize=ReleaseSafe
zig build proofs
python3 -B proofs/test_verifier.py
zig test --dep plan -Mroot=proofs/check_bindings.zig -Mplan=src/resident_plan.zig
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/verify.py zig-out/bin/flint --staged-row-tile 2
```

| Location | Responsibility |
| --- | --- |
| [`src/model.zig`](src/model.zig) | Model execution, expert scheduling, and persistent KV state |
| [`src/weights.zig`](src/weights.zig) | Checkpoint mappings and quantized tensor descriptors |
| [`src/metal.zig`](src/metal.zig), [`src/metal.m`](src/metal.m) | Metal bridge, cached pipelines, barriers, and completion handling |
| [`src/`](src/) | Metal kernels, generated plans, and native tests |
| [`proofs/`](proofs/) | Lean contracts, emitters, and binding verification |
| [`tools/`](tools/) | Tokenizer/HTTP bridges, parity checks, and benchmark drivers |
| [`benchmarks/`](benchmarks/) | Reproducible measurements and rejected experiments |

### Current boundaries

Apple Silicon only. Single sequence, inference only. No training/autograd, general
lazy tensor graph, multi-device execution, concurrent multi-user batching, or native
tokenizer/template renderer. Ordinary generation is greedy, not Flint's recommended
stochastic sampling. The Python adapter rejects unsupported sampling and reasoning
controls rather than silently ignoring them. Full batched prefill and broad
model-quality parity remain unverified. mmap avoids an extra full weight copy; it
does not remove physical-memory or Metal-buffer limits.

## License

Original code and contributions are **[MIT licensed](LICENSE)**. Adapted Recursive
`rec` components retain Apache-2.0 terms; adapted MLX components retain MIT terms
and notices. The top-level MIT license does not replace those upstream obligations.
Model weights are not bundled and remain separately licensed.

See [third-party notices](THIRD_PARTY_NOTICES.md), [provenance](PROVENANCE.md), and
[`licenses/`](licenses/).

<p align="center"><sub>Built in Zig · Executed on Metal · Contracts checked in Lean</sub></p>
