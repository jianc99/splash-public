# Embedded GGUF tokenizer validation — 2026-09-23

The upstream loader now builds its tokenizer and model configuration from the
selected GGUF's metadata. It does not fetch a tokenizer/configuration from Qwen,
MLX, or another repository. GGUF sidecars cannot override embedded metadata.
Vision metadata comes from the selected snapshot's mmproj. Draft selection is
unchanged; the independently prepared DFlash2 assets are still required.

## Sources

Real metadata was read from the existing Unsloth GGUF files on the M3 Max:

| Source | Hub content SHA-256 |
| --- | --- |
| Qwen3.8-27B-UD-Q4_K_M.gguf | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` |
| 35B mmproj-BF16.gguf | `356dfaa3111376a4f7165e32e8749713378d1700b37cf52e0c50d9f23322334d` |

Both language models declare `gpt2/qwen35`, 248,320 vocabulary entries,
247,587 merges, 27 control tokens, six user-defined tokens and 243 unused entries.
The qwen35 profile uses its combining-mark-aware split pattern, NFC and byte-level
BPE. Vocabulary IDs, merge order, token types and template text come from the file.
Unknown profiles, malformed metadata and unsupported automatic BOS/EOS insertion
fail explicitly. The metadata reader has byte/count bounds and does not read the
weight tensor payload.

27B declares 65 blocks including one MTP layer: the generated configuration
correctly describes 64 target transformer layers. 35B declares 40 layers.

## Tokenizer and template checks

On the local M5, each reconstructed tokenizer was checked against the existing
local MLX Qwen3.6 tokenizer, which shares the vocabulary, merges and pre-tokenizer
with these two GGUFs. This reference was used only for testing; loading a GGUF
never accesses it.

- All 248,077 shared vocabulary IDs match.
- 1,042 text cases per model match exactly in input IDs and character offsets.
  Cases include Chinese/Japanese/Korean, combining marks, emoji, whitespace,
  control characters, code, JSON, numbers and every added token, plus 1,000
  deterministic combinations (seed 77231).
- Decoding with special tokens retained matches the reference. With special
  tokens skipped, decoding follows the GGUF's control-token flags. These flags
  intentionally differ from the HF tokenizer for some FIM tokens.
- The embedded template is preserved byte-for-byte. Sixteen rendering cases per
  model cover plain/system messages, tool calls and responses, image placeholders,
  tools enabled/disabled, and thinking enabled/disabled. Rendering matches the
  original embedded template and the resulting token IDs match the reference.
- Tokenizer reload and all comparisons run with network connections disabled.
- The existing in-memory system-message compatibility handles both verified
  GGUF templates: 27B otherwise rejects later system messages; 35B otherwise
  silently skips them. Neither cached template nor message order is rewritten.

## Integration and regression tests

M3 Max 40-core, existing cached GGUF weights, prepared local DFlash2 assets,
`HF_HUB_OFFLINE=1`, language-only and an 8K context:

- Both 27B and 35B assemble, pass frontend tokenizer validation, start the native
  engine, and complete two real Chat Completions requests, including a later
  system message. Replies are nonempty and finish normally.
- 27B uses its already-cached revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`
  because this machine has no cached `main` reference for that repository.
- Repeated preparation takes approximately 0.50 s (27B) and 0.19 s (35B), including
  local draft-file hashing. These are preparation times, not full engine startup.
- These runs populated the native prepared-weight cache and took about 32–34 s
  to become ready. They are correctness smoke tests, not throughput benchmarks.
- Both test servers exited and released the GPU lock afterward.

Local regression checks: 548 engine Python tests passed (three skips); 117
installer, metadata, template, launcher and packaging tests passed. Metadata
coverage includes truncated files, duplicate keys, unknown types, nested/oversized
arrays, invalid UTF-8, incompatible tokenizer metadata, concurrent preparation,
interrupted publication, corrupted cache entries, source changes, no cross-repo
resolution, sidecar conflicts and language-only exclusion of mmproj. Formatter,
linter and architecture checks passed.

No native engine, weight preparation or inference kernel changed in this work.
Existing vision precision tests are documented in `upstream-vision.md`; this
change additionally checks the real mmproj metadata conversion and the source
assembly, but does not repeat a full image inference run.

Evidence is retained in
`/Users/jianchen/dev/flashmlx_dev/splash-gguf-tokenizer-20260923/`:
`verify.py`, `verification.jsonl`, both metadata headers, `smoke*.py`, `smoke*.log`,
`server*.log`, and test logs. Committed synthetic metadata tests are runnable with
`python -m unittest dev.tests.test_gguf_metadata`; template fixtures record their
source identities in `dev/tests/fixtures/chat_templates/README.md`.
