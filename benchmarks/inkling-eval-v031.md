# nanoattention Eval & Perf Report (`nano 0.3.1`)

- **Checkpoint:** `thinkingmachines--Inkling-2-Mini` (`model_type=inkling2_loop`)
- **Draft:** `inkling-2-mini-dflash2` · **Mode:** `both` · **Knobs:** `src/mega/presets/inkling.json`
- **Suites:** `core` · **Repeats:** `2` · **Timestamp:** `2026-11-27T07:59:28Z`

## Config Summary

| Config | Pass | Quality | Coherent | Parity | Leaks | Prefill (`tok/s`) | Coherent Decode (`tok/s`) | All Decode (`tok/s`) | Short (`tok/s`) | 4K+ (`tok/s`) | `mean_step` | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`3p_hidden`** | 6/6 | 100.0% | 6/6 | 6/6 | 0 | 1153.9 | **110.23** | 110.23 | 117.48 | 92.43 | 4.39 | 1.69× |
| **`3p_off`** | 6/6 | 100.0% | 6/6 | 6/6 | 0 | 1155.1 | **107.90** | 107.90 | 114.59 | 90.52 | 4.18 | 1.66× |
| **`3p_adaptive`** | 6/6 | 100.0% | 6/6 | 6/6 | 0 | 1117.7 | **99.98** | 99.98 | 105.12 | 79.71 | 3.78 | 1.54× |
| **`2p_off`** | 0/6 | 0.0% | 2/6 | 6/6 | 0 | 1567.6 | **117.81** | 132.98 | 127.63 | 152.08 | 4.06 | 1.55× |

## Per-Task Breakdown

| Config | Task | In / Out | Pass | Coherence | Parity | Base (`tok/s`) | Decode (`tok/s`) | Speedup | `mean_step` | Reason |
| :--- | :--- | ---: | :---: | :--- | :---: | ---: | ---: | :---: | :---: | :--- |
| `3p_hidden` | `coding_merge_intervals` | 169 / 580 | PASS | clean | PASS | 65.81 | 112.16 | 1.70× | 4.26 | all_7_tests_passed |
| `3p_hidden` | `coding_binary_search` | 180 / 369 | PASS | clean | PASS | 65.85 | 113.98 | 1.73× | 4.50 | all_tests_and_log_reads_passed |
| `3p_hidden` | `math_mod7` | 121 / 779 | PASS | clean | PASS | 65.70 | 115.83 | 1.76× | 4.72 | exact_mod7_roots |
| `3p_hidden` | `math_batches` | 129 / 397 | PASS | clean | PASS | 65.98 | 121.24 | 1.84× | 5.51 | exact_batches_20_44_34 |
| `3p_hidden` | `json_extract` | 162 / 227 | PASS | clean | PASS | 65.97 | 141.42 | 2.14× | 6.68 | exact_json_match |
| `3p_hidden` | `code_4k` | 3990 / 755 | PASS | clean | PASS | 63.67 | 92.43 | 1.45× | 3.45 | exact_code_4k_answer |
| `3p_off` | `coding_merge_intervals` | 169 / 539 | PASS | clean | PASS | 65.76 | 101.45 | 1.54× | 3.67 | all_7_tests_passed |
| `3p_off` | `coding_binary_search` | 180 / 632 | PASS | clean | PASS | 65.08 | 112.66 | 1.73× | 4.33 | all_tests_and_log_reads_passed |
| `3p_off` | `math_mod7` | 121 / 947 | PASS | clean | PASS | 64.92 | 115.32 | 1.78× | 4.83 | exact_mod7_roots |
| `3p_off` | `math_batches` | 129 / 415 | PASS | clean | PASS | 65.91 | 123.55 | 1.87× | 5.19 | exact_batches_20_44_34 |
| `3p_off` | `json_extract` | 162 / 232 | PASS | clean | PASS | 65.79 | 141.77 | 2.15× | 6.44 | exact_json_match |
| `3p_off` | `code_4k` | 3990 / 840 | PASS | clean | PASS | 63.21 | 90.52 | 1.43× | 3.26 | exact_code_4k_answer |
| `3p_adaptive` | `coding_merge_intervals` | 169 / 792 | PASS | clean | PASS | 65.32 | 90.07 | 1.38× | 3.07 | all_7_tests_passed |
| `3p_adaptive` | `coding_binary_search` | 180 / 487 | PASS | clean | PASS | 65.50 | 103.65 | 1.58× | 4.31 | all_tests_and_log_reads_passed |
| `3p_adaptive` | `math_mod7` | 121 / 817 | PASS | clean | PASS | 65.50 | 107.02 | 1.63× | 4.30 | exact_mod7_roots |
| `3p_adaptive` | `math_batches` | 129 / 445 | PASS | clean | PASS | 65.79 | 124.18 | 1.89× | 5.30 | exact_batches_20_44_34 |
| `3p_adaptive` | `json_extract` | 162 / 249 | PASS | clean | PASS | 65.82 | 136.74 | 2.08× | 5.93 | exact_json_match |
| `3p_adaptive` | `code_4k` | 3990 / 537 | PASS | clean | PASS | 62.74 | 79.79 | 1.27× | 2.80 | exact_code_4k_answer |
| `2p_off` | `coding_merge_intervals` | 169 / 1024 | **FAIL** | low_tail_diversity:0.276 | PASS | 88.34 | 131.10 | 1.48× | 3.84 | truncated_in_thinking:no_python_code_found |
| `2p_off` | `coding_binary_search` | 180 / 1024 | **FAIL** | repeated_line_x4 | PASS | 82.27 | 137.29 | 1.67× | 4.20 | truncated_in_thinking:no_python_code_found |
| `2p_off` | `math_mod7` | 121 / 1280 | **FAIL** | clean | PASS | 87.64 | 119.60 | 1.36× | 3.56 | truncated_in_thinking:missing_FINAL_ANSWER_line |
| `2p_off` | `math_batches` | 129 / 768 | **FAIL** | repeated_4gram_x94 | PASS | 86.58 | 136.81 | 1.58× | 4.20 | truncated_in_thinking:missing_FINAL_ANSWER_line |
| `2p_off` | `json_extract` | 162 / 512 | **FAIL** | clean | PASS | 88.54 | 113.60 | 1.28× | 3.28 | truncated_in_thinking:invalid_or_non_object_json |
| `2p_off` | `code_4k` | 3990 / 1536 | **FAIL** | low_tail_diversity:0.101 | PASS | 84.51 | 152.08 | 1.80× | 5.04 | truncated_in_thinking:wrong_or_missing_RING_GUARD_SLOTS:None |
