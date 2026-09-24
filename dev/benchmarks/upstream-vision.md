# Upstream vision loading validation — 2026-09-23

The loader reads MLX vision tensors from the same safetensors checkpoint as the
language model, or the companion GGUF mmproj from the selected GGUF repository.
Both adapters prepare the packed `vision/model.bin` layout (MDFV0001, every
tensor BF16), so all sources run the existing vision operator graph and shaders.
A BF16 source tensor is copied. An F32 or F16 tensor is converted only when every
value is exactly a BF16; otherwise preparation fails, naming the tensor and file.

## Sources

- MLX: `mlx-community/Qwen3.6-35B-A3B-4bit` at
  `38740b847e4cb78f352aba30aa41c76e08e6eb46` and `mlx-community/Qwen3.8-27B-4bit`
  at `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. The vision tower is BF16 and
  lives entirely in shard 1.
- GGUF: `unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf`, SHA-256
  `356dfaa3111376a4f7165e32e8749713378d1700b37cf52e0c50d9f23322334d`, and
  `unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf` at
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`, SHA-256
  `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`.
- Packed oracles: `vision/model.bin` of `incoai-internal/Qwen3.6-35B-A3B-Splash`
  at `35842315ef66c3971de3f11553f27e127a75056a`, and the 27B file shared by
  `incoai-internal/Qwen3.8-27B-Splash` and `incoai-internal/Qwen3.8-27B-Splash-GGUF`.

Each mmproj holds 334 tensors: 110 BF16 matrices and 224 F32 tensors (every 1-D
tensor, both patch-embedding frames and the position table). No F32 value has
non-zero low 16 bits: 0 of 4,829,936 elements (35B) and 0 of 4,833,008 (27B).

## Prepared files

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| 35B mmproj-BF16 | 901,939,200 | `20ba816b644c7221d33d1b188612d096dae3e96600be2dc3d4bd1600c0d30d00` |
| 35B MLX | 901,939,200 | `20ba816b644c7221d33d1b188612d096dae3e96600be2dc3d4bd1600c0d30d00` |
| 27B mmproj-BF16 | 930,250,752 | `8973858e75ec3464f626d5f9111dbceda58f3bac5737ea5693767686003c65df` |
| 27B MLX | 930,250,752 | `8973858e75ec3464f626d5f9111dbceda58f3bac5737ea5693767686003c65df` |

Every prepared file is byte-identical to the packed `vision/model.bin` of the same
model, including padding. Preparation of each file took under 1 s once the
source digests were known (35B on the M5 Pro, 27B on the M3 Max).

## Execution

The vision shader, operator and loader are the BF16 code that preceded upstream
loading; the metallib holds its 12 vision functions. The fixture embedding
SHA-256, with Metal shader validation, is the same before (`fd7b4cc`) and after:

| Machine | Model and fixture | Embedding SHA-256 |
| --- | --- | --- |
| M5 Pro 20 cores (Apple10) | 35B packed, `qwen3.6-35b-a3b` | `7946f077435ef45d0a596461a9d9a234ff458c805c007bb3ea1af7296bd230f9` |
| M5 Pro 16 cores (Apple10) | 35B packed, `qwen3.6-35b-a3b` | `7946f077435ef45d0a596461a9d9a234ff458c805c007bb3ea1af7296bd230f9` |
| M3 Max (Apple9) | 27B packed, `qwen3.8-27b` | `011d9121bf52f85a26e7eff28e9c9459a48eb0bb36d585ff8e7ef7621cd3a37b` |

All pass the vision gate (35B: relative error 0.0280, worst row cosine 0.9989;
27B: 0.0129, 1.0000), deterministic repeated encoding, the 64×64 and maximum
128×128 grids, and reuse of the small fixture after the large grid. Loaded
through source-model roots on the M5 Pro 20, the 35B mmproj and MLX sources give
the packed embedding. Before this change the mmproj ran the F32 kernels on these
same values from a 911,245,312-byte file, and its 35B embedding differed
(`c1a62666d5f2f6ecd0e0ecf6036e82737eee34ff3086c5e621a993502601ba28`, relative
error 0.0295, worst row cosine 0.9976).

## Integration and coverage

- The CPU gate prepares tiny MLX and GGUF towers whose tensors take every dtype
  (BF16, F16, F32, including signed zero, infinity and 2^-24) and compares them
  with an independently serialized packed file: patch reordering, padding and
  alignment. Inexact F32 and F16 values, unsupported dtypes, shape mismatches, a
  quantized MLX tower, missing or set deepstack metadata, a wrong LayerNorm
  epsilon and an mmproj tensor the tower does not use are rejected by message,
  without publishing a file. The MLX vision identity survives a text-shard change
  and changes with `config.json` or the vision shard.
- The whole-model disk check budgets the vision artifact with the target images;
  the Metal gate checks this for the affine and GGUF target loaders.
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
