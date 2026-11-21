# Source provenance and license scope

This repository started as the standalone native Zig/Metal Flint engine, exported
from the verified engine source snapshot `56166c5` without parent history or
checkpoint files. The later Liquid AI port adds the `nano` executable and
related sources, tools, and benchmark evidence without merging the earlier
branch history or bundling model assets.

The repository contains no Rust/CUDA implementation files or Rust/CUDA build
dependencies from the earlier workspace. `build.zig` compiles local Zig entry
points and `src/metal.m`, links the macOS Foundation/Metal frameworks, and checks
the local Lean-generated bindings. Native checkpoint loading, model scheduling,
KV ownership, protocol adapters, proofs, tests and tuning tools are kept here.

## Licensing boundaries

- Original implementation and contributions: MIT, as stated in `LICENSE`.
- MLX-derived quantized arithmetic and attention/numerical scheduling: preserve
  Apple copyright/MIT notices in the shaders and `licenses/mlx-MIT.txt`.
- Recursive `rec` resident scheduling/barrier adaptation: preserve attribution,
  modification notices, and `licenses/rec-APACHE-2.0.txt`.
- Checkpoints and tokenizers: not included in this source repository. Their
  licenses are separate; the source MIT license does not relicense model files.

`THIRD_PARTY_NOTICES.md` records the retained upstream components and modifications.
No third-party adaptation is represented as exclusively original code. The
integer proofs certify their documented abstract contracts, not authorship,
floating-point equivalence, compiler correctness, or GPU forward progress.
