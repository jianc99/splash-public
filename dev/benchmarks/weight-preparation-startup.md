# Weight preparation startup checks (2026-09-23)

This follow-up hardens the source-loading refactor after `960844d`. It fixes
installed Hub snapshot identification, separates conversion admission from normal
loading, bypasses the converter lock on immutable cache hits, preflights all
missing model artifacts with a 2 GiB disk reserve, and preallocates each output.
Completed layers survive an interrupted load; abandoned writes are removed under
the converter lock on retry. New entries record their source, and cold conversion
reports artifact progress. Preparation identities now fingerprint conversion code
and its storage ABI at build time. Inference kernels are unchanged.

## Measured preparation time

Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, source SHA-256
`ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
The 40 layers, head and embedding produce 42 artifacts totaling 22,143,172,608 bytes.

| Device | Previous 256-row executor | Batched executor | Reuse prepared artifacts |
| --- | ---: | ---: | ---: |
| M3 Max, 40 GPU cores | 74.98 s | 29.74 s | 0.641 s |
| M5 Pro, 16 GPU cores | 50.82 s | 25.87 s; final repeat 24.30 s | 0.539 s; final repeat 0.533 s |

These are **weight-preparation timings**, including backend creation, source
verification, image planning, file conversion, output hashing and publication.
They exclude tokenizer/server startup, final model residency and inference warmup.
Both executors use the same hardened cache writer; the comparison isolates the
old executor from the new batched conversion. Cold means an empty preparation
cache and no source hash proof, not a forcibly evicted OS filesystem cache.
M5 ran candidate then previous; M3 ran previous then candidate. The M5 final
repeat includes source sidecars and progress reporting. These samples demonstrate
a large reduction in preparation work, not a universal startup-speed guarantee.

Every artifact SHA-256 matches between executors and between devices. The paired
runs recorded no change in system swap-in/out counters. M3 had no allocated swap;
M5 had pre-existing swap usage which did not increase. Peak process RSS was about
65–68 MB for the batched executor versus 34–36 MB previously. Staging payload is
bounded to 32 MiB and retains the existing 64 MiB conversion admission reserve;
the bound does not scale with tensor, layer or expert count.

A real upstream affine 35B checkpoint was also prepared and compared against all
40 layers, head and embedding of the released package: every byte matched. The
81.07 s oracle run includes full output comparison and is not a cold-loading
benchmark. Reopening the resulting affine artifacts, without comparison, took
0.064 s on the local M5 Pro. Ordinary serving still performs residency and warmup.

## Regression coverage

- The installer fixtures now use the real installation → snapshot → blob link
  structure, including offline reuse and verification.
- Native cache tests cover warm admission, cancellation, a warm load while an
  unrelated converter holds the lock, whole-model disk rejection, the disk
  reserve, stale staging cleanup, corruption, interrupted/failed writes and
  same-key concurrent writers. AddressSanitizer/UndefinedBehaviorSanitizer pass.
- A small, standalone affine checkpoint has an independent byte-layout oracle:
  GDN and full-attention fusion, padding, distinct gate/up weights, head and
  embedding. A warm loader must succeed even when conversion is forbidden.
  Unsanitized convolution storage is rejected before any new cache publication.
- GGUF tests cover every supported quantization format, head permutations,
  multiple row batches, extremely wide rows, copy/repack offsets beyond 4 GiB,
  staging lifetime/bounds, and warm admission. Each process gets a fresh cache.
  Shader validation passed on M3 Max and both M5 Pro devices; M3 repeated the
  test from a fresh cache. The final staging-bound assertion also passes locally.
- The full CPU suite and 72 Python installer/build-identity checks pass.
- Prepared-code fingerprint tests distinguish conversion changes from unrelated
  source changes. Native build-identity verification passes.
- Before the shader separation described below, the production metallib at
  `2ca5691` remained byte-identical to the previous implementation:
  `b61f7cbea458952315c88953f16f31ef2226aee682a81d2e242cd660c7c9745b`.

## Preparation fingerprint follow-up

The GGUF preparation and copy kernels now live in `shared/gguf_repack.metal`.
Their bodies were moved verbatim, and all remaining inference source is unchanged.
The preparation fingerprint includes this file, the host planner/executor and
the shared storage/parameter ABI. It no longer includes `gguf_linear.metal` or
the inference helpers in `common/quant_formats.h`. Tests edit those inference
files and the Apple9 decode file independently to check that they leave the
preparation identity unchanged; each preparation input must still change it.

This changes the development cache identity once. Future edits to those inference
implementations will no longer force weight preparation. Shared ABI changes
conservatively invalidate the cache. Shader separation changes the metallib
container, so the earlier binary-identity claim does not apply to this follow-up.
The production build, 72 Python checks and native build-identity verification
pass after separation. On the local M5 Pro, both GGUF repack and full projection
tests pass with Metal shader validation (zero failures). This follow-up did not
repeat the full-model preparation timing or the earlier cross-device runs.

## Remaining boundaries

The prepared target still occupies an additional disk copy. Completed generations
are explicitly removable while Splash is stopped; automatic eviction is not
implemented. A disk reserve cannot prevent another process from filling the disk.
File backing avoids the whole-model anonymous conversion allocation but does not
make pages reclaimable while Metal holds them resident. This work neither adds
new model architectures nor automatically selects support assets for arbitrary
upstream model IDs.
