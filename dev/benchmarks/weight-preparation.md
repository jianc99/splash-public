# Prepared-weight storage validation (2026-09-23)

Baseline: PR 114 at `d25020e0fb0291f371ab4a319015f8012087b0e2`.
This comparison covers the loading/storage and operator-layout refactor, not a
new quantization kernel. The production metallib is byte-identical to baseline:
`b61f7cbea458952315c88953f16f31ef2226aee682a81d2e242cd660c7c9745b`.

## Implementation boundaries

- `AffineCheckpoint` reads checkpoint configuration and a bounded safetensors
  index. `AffineTarget` maps architectural tensor names to the existing affine
  ABI; it does not dequantize and requantize the target.
- `GgufImage` retains the existing format planner. `GgufPreparation` executes it
  in bounded row/K tiles using the existing repack kernel and direct byte copies.
- `PreparedWeights` owns file identity, writer exclusion, temporary generations,
  integrity verification and publication. `WeightFile` has only a read-only
  file-backed constructor; loaded models retain its mapped views.
- `Projection`, `MoeWeights` and `EmbeddingWeights` carry the operator's storage
  contract. Planning and scratch sizing follow actual loaded weights, including
  mixed layer layouts. The vocabulary projection reserves decode scratch only.
- Frontend protocols and engine scheduling are unchanged. Startup admission
  counts prepared target bytes and checks headroom before conversion chunks.

The source-package contract and its limits are documented in
[DEVELOPMENT.md](../../DEVELOPMENT.md#upstream-target-weights). In particular,
`--model` still selects a compatible support package; this change does not infer
draft/tokenizer compatibility from arbitrary upstream repository names.

## Correctness and failure handling

- Production build, architecture checks and the complete CPU suite passed.
- Complete Metal suite passed with shader validation on M5 Pro 20 GPU cores.
- Bounded GGUF repack passed on M3 Max and M5 Pro: all eight supported formats,
  multiple row tiles, a K dimension crossing the 8192-column chunk, row
  permutation and offsets above 4 GiB, compared bytewise with the CPU reference.
- Native 35B affine preparation matched every byte of all 40 layers, head and
  embedding in the released package.
- Native 27B matched all packed weights, parameters, other tensors and padding
  across 64 layers, head and embedding. The small GDN decay vector differs by
  at most one float ULP from the released package's MLX exponential. The adapter
  computes `float(-exp(double(A_log)))`. Its optional real-source oracle permits
  at most two ULP only in this named section; it requires exact bytes elsewhere.
- All measured B1–B4 output hashes, accepted draft counts and decode-batch counts
  matched baseline, for both source formats and both target architectures.
- Cache tests cover warm reuse, same-size corruption, damaged digest proofs,
  interrupted writers, injected ENOSPC, rejected pressure checks, concurrent
  builders, source mutation and path replacement. The cache tests also passed
  AddressSanitizer/UndefinedBehaviorSanitizer; the existing sanitizer suite passed.
- Checkpoint tests cover bounded reads, duplicate and overlapping tensors,
  malformed dtypes/shapes/ranges, truncated files and quantization metadata.
  Installer tests cover pinned multi-shard sources, both target schemas, GGUF
  source assembly, offline reuse and restoring the previous installation after
  publication failure. All 54 installer tests passed.
- Same-shaped prepared weights with different content identities are verified
  to produce different runtime cache namespaces.

## Performance method

M3 Max 40 GPU cores (Apple9) and M5 Pro 16 GPU cores (Apple10), AC power, one
benchmark at a time under the machine's GPU lock. `backend-benchmark --scenario
decode --samples 3` performs production bootstrap/warmup and generates 64 tokens
per lane at B1–B4. Each model runs baseline/candidate/candidate/baseline.
Reported ratios are baseline median GPU time divided by candidate median GPU
time, so 1.000 means unchanged. These are measurements, not a guarantee for every
power state, prompt or untested chip.

The first pass used existing packed affine packages and existing GGUF packages.
The second pass used upstream affine checkpoints with the same support assets,
and repeated the GGUF comparison after the final workspace cleanup.

### Existing packages

| Device | Target | B1 | B2 | B3 | B4 |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 Max | affine27 | 0.999 | 1.000 | 0.999 | 0.999 |
| M3 Max | affine35 | 1.004 | 0.997 | 0.998 | 0.996 |
| M3 Max | gguf27 | 1.000 | 1.000 | 1.001 | 1.001 |
| M3 Max | gguf35 | 0.998 | 0.998 | 1.000 | 1.001 |
| M5 Pro | affine27 | 0.995 | 1.000 | 1.003 | 0.997 |
| M5 Pro | affine35 | 0.999 | 1.006 | 0.999 | 0.998 |
| M5 Pro | gguf35 | 1.002 | 0.994 | 1.005 | 1.003 |

### Locally prepared affine and GGUF

| Device | Target | B1 | B2 | B3 | B4 |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 Max | affine27 | 1.004 | 1.067 | 1.035 | 1.037 |
| M3 Max | affine35 | 1.008 | 1.020 | 1.013 | 1.006 |
| M3 Max | gguf27 | 1.003 | 1.001 | 1.000 | 1.000 |
| M3 Max | gguf35 | 0.999 | 1.000 | 1.000 | 0.998 |
| M5 Pro | affine27 | 1.000 | 0.997 | 1.000 | 0.998 |
| M5 Pro | affine35 | 0.999 | 1.004 | 1.000 | 0.998 |
| M5 Pro | gguf35 | 1.005 | 1.008 | 1.001 | 1.000 |

Existing-package decode stayed within approximately 0.6% of baseline. The M3
native-affine round had substantial timing drift within and across runs; its
apparent gains are not claimed as an optimization. Native affine on M5 stayed
within 0.4%, and final GGUF runs stayed within 0.8%. Output and acceptance agreement
held in every run.

The single startup 2048-token prefill was noisy. A separate repeated prefill
probe uses the production executor, an initial warmup, fresh request state for
each repetition and unprofiled fused GPU time. Its source is derived from
`dev/benchmarks/decode_profile.mm`; it changes no production operation.

M5 GGUF 35B, ABBA, seven repetitions per process: baseline median 683.871 ms,
candidate 683.525 ms (ratio 1.0005). The startup-only difference did not persist.

M5 native affine 27B showed timing drift of roughly 8% within individual
processes. The initial ABBA pooled ratio was 0.9696. A reverse BAAB run with
12 repetitions per process, the first five excluded from the steady comparison,
gave a pooled ratio of 1.0171. The adjacent baseline/candidate run-median ratios
in that second round were 0.9963 and 1.0028. The apparent difference reversed
with run order/longer warmup; it is not evidence of a consistent prefill
regression or gain. Display/system idle were also prevented in the second round.
Raw samples are retained rather than claiming an exact universal percentage.


## Memory and startup

No new system swap-out pages were recorded across either 28-run comparison.
M5 did fault in a small number of previously swapped pages (64 in the first
comparison, 44 in the second); this is not a zero-system-swap claim.
The 27B preparation/oracle process reported about 21.1 MiB peak private footprint;
file-backed residency and OS/driver caches are separate. The 64 MiB preparation
headroom reserve includes capped metadata plus small staging buffers.

A warm 35B weight-open-only oracle took 72 ms locally after content verification
proofs were established. This excludes engine startup, GPU warmup and tokenizer
loading. First preparation and full byte-oracle timings are not serving-startup
benchmarks. The cache requires an additional prepared-target disk copy; original
upstream files are retained.

## Reproduction and evidence

Build and run `make check-source check-native-cpu`, `make test-engine-metal` and
`make test-sanitizers` using the supported Python toolchain. Installer tests are
`python -m unittest dev.tests.test_models`. Build the optional
`build/engine-tests/affine-source-oracle` and run it as:

```sh
build/engine-tests/affine-source-oracle build/splash.metallib SOURCE_CHECKPOINT PACKED_PACKAGE
```

Use `SPLASH_WEIGHT_CACHE` to select a fresh cache for cold preparation or the
same cache for reuse. Raw JSON, power/swap telemetry, binary hashes, standalone
probe sources and logs are retained in the local evidence directory
`splash-model-storage-design-20260923/` next to this checkout. Baseline and
candidate executables were frozen before remote runs; the JSON records retain
their build IDs and SHA-256 hashes.
