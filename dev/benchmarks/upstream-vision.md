# Upstream vision loading validation — 2026-09-23

The loader reads MLX vision tensors from the same safetensors checkpoint as the
language model, or the companion GGUF mmproj from the selected GGUF repository.
Both adapters prepare file-backed weights for the existing vision operator graph.
The new typed shader instantiations read F32 weights directly. BF16 matrices
remain BF16; F16 is promoted exactly to F32. Activations remain BF16.

## Sources and independent checks

- MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`, snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- GGUF: `unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf`, SHA-256
  `356dfaa3111376a4f7165e32e8749713378d1700b37cf52e0c50d9f23322334d`.
- Existing packed vision oracle: `incoai-internal/Qwen3.6-35B-A3B-Splash`,
  snapshot `35842315ef66c3971de3f11553f27e127a75056a`.

All MLX prepared bytes match the existing packed vision file, including padding.
All 333 GGUF prepared tensors match an independent Python source reader, including
F32 parameters, temporal patch combination, and MLP padding. The GGUF numerical
reference reads the source tensors directly and runs the full encoder in NumPy
FP32; it does not consume the native prepared file to generate expected outputs.

On M5 Pro 20 cores, with Metal shader validation:

| Vision source | Fixture | Relative L1 error vs FP32 | Worst row cosine |
| --- | --- | ---: | ---: |
| MLX BF16 | 12×12 patches | 0.0280 | 0.9989 |
| GGUF mixed BF16/F32 | 6×8 patches | 0.0160 | 0.9996 |

These are different fixtures and source weights, so the errors are not a ranking
of formats. Both satisfy the existing vision gate (relative error < 3%, every
row cosine > 0.995). Both pass deterministic repeated encoding, a 64×64 grid,
the maximum 128×128 grid, and reuse of the small fixture after the large grid.

The F32 precision test passes on M3 Max (Apple9) and M5 Pro (Apple10), with shader
validation enabled. Cancellation-sensitive inputs ensure that sub-BF16 differences
in F32 matrix weights, bias, LayerNorm parameters, and position-table interpolation
survive until output rounding. This checks actual GPU computation, not only dtype
labels or file serialization. The full real-source encoder gate above was run on
M5; the M3 check was the precision test.

## Integration and coverage

- The CPU gate includes independent tiny safetensors/GGUF layout oracles, mixed
  BF16/F16/F32 tensors, padding, warm-cache reuse, and invalid shape/type/metadata
  rejection before publishing a prepared file.
- The Metal gate includes the F32 precision regression test.
- On M5, loading the actual MLX 35B source with an independently exported unchanged
  Q4 draft and `--language-only` passes native bootstrap and B1–B4 inference. Output
  token hashes, accepted draft counts and decode batch counts match the earlier
  packed-model baseline for the same benchmark prompt. This is a correctness
  check, not a new throughput claim.
- Text-only memory admission and warmup auditing permit zero vision weight bytes.
  Missing measurements still fail for multimodal plans; unexpected vision bytes
  fail for text-only plans. HTTP image requests are rejected before decoding;
  native image requests are rejected before scheduling, leaving text service usable.

Evidence, the independent real GGUF reader and logs are retained outside the
repository in `splash-upstream-loading-20260923/`. Independent packed DFlash2 assets
must be published to the selected draft repositories before automatic public
model loading can be shipped. Local integration uses `--draft-model` with the
exported assets; no new draft quantization is involved.

## BF16 execution regression

M5 Pro 20-core ABBA comparison, all four runs on AC power, shader validation off.
The baseline uses the pre-change vision shader and packed weights. The candidate
uses the typed shader and the actual upstream MLX source assembly. Each run
measures five 4096-patch encodes and also checks the small fixture and maximum grid.

| Run | Median GPU ms |
| --- | ---: |
| Baseline A | 278.18 |
| Candidate B | 278.04 |
| Candidate B | 277.11 |
| Baseline A | 277.90 |

Mean of run medians: 278.04 → 277.58 ms (−0.17%, effectively unchanged).
The fixture embedding SHA-256 is identical in all four runs:
`7946f077435ef45d0a596461a9d9a234ff458c805c007bb3ea1af7296bd230f9`.
This measures vision execution, not whole-model generation speed.

The native CPU and Metal suites pass. Python regression runs: 548 engine tests
(3 skipped), plus the server/installer/package tests and the new source resolver
checks. Formatting, lint and architectural dependency checks pass.

M3 Max (Apple9) ABBA, using the same packed BF16 vision weights to isolate the
shader change: baseline 530.53 / 530.66 ms; candidate 530.54 / 530.61 ms.
Mean of medians is effectively unchanged (530.60 → 530.58 ms). The small fixture
embedding SHA-256 matches across all runs:
`29577c9d8aa329c5edd0cf2b90368107f18e15cb3cc23e7950b43c5ed8b81b1c`.
All four runs pass the maximum grid and deterministic arena-reuse checks.
Both Apple9 and Apple10 also pass the strengthened F32 cancellation test using
weight/bias differences of 2^-16, which would be lost by an F16/TF32 pre-rounding.
