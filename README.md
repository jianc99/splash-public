# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash serves a small set of models to coding agents and to any OpenAI or
Anthropic compatible client, on one Mac. On a 48 GB M5 Pro, Splash 1.0 decoded
Qwen3.8-27B at 2× the speed of the next-fastest engine we measured and, with a
32K context cached, returned the first token in 282 ms
([Performance](#performance)). Its kernels, draft model, and memory
plan are specialized for each model it serves. That is why it is fast, and why
there is nothing to configure.

## Quick start

Apple M3 or newer, macOS 26.4 or later, [Homebrew](https://brew.sh), 36 GB
of unified memory (48 GB or more recommended), and free disk for the model and
a prepared copy of its weights (about 35 GB in total for Qwen3.8-27B).

```bash
brew install incoai/tap/splash
splash serve --model mlx-community/Qwen3.8-27B-4bit
```

The first run downloads the model and its matching DFlash2 draft, prepares
weights for the Metal kernels, checks available memory, and starts serving on
`127.0.0.1:8000`. Later starts reuse the prepared weights.

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes
```

Press Ctrl+C in the server terminal to stop Splash.

## Use the API

Splash speaks OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses
(`/v1/responses`), and Anthropic Messages (`/v1/messages`), all with streaming,
tool calls, JSON Schema output, images, and inline PDFs. `/tokenize` and
`/apply-template` return token IDs and the rendered prompt without running the
model.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

`model` is optional. If set, use the served model ID or a configured
[model alias](DEVELOPMENT.md#api-model-aliases).
Reasoning follows the model default unless a [server default](DEVELOPMENT.md#default-reasoning-effort)
is configured. `"reasoning_effort": "none"` turns it off, and
Qwen3.8-27B also takes `low`, `medium`, and `xhigh`.

`/v1/judgments` and `/v1/systemone` provide scoring without generation.
See [judgment contracts](DEVELOPMENT.md#judgment-contracts) for details.

## Models

| Base model | MLX affine example | GGUF example |
| --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` |

`--model` accepts the upstream repository directly: an MLX affine 4-bit,
group-64 checkpoint (such as the mlx-community `-4bit` conversions) or a GGUF,
selected with `OWNER/REPO:VARIANT` (for example `:UD-Q4_K_M`). Splash
identifies the model from its own metadata, its architecture and dimensions,
before downloading any weights, and pairs the DFlash2 draft trained for it;
`--draft-model` replaces that draft with another repository of the same layout
or a local draft directory. The tokenizer, configuration and chat template come
from the target repository for MLX and from the selected GGUF file itself for
GGUF, never from another repository: unsupported or incomplete tokenizer
metadata is an error. GGUF variants whose tensor types Splash cannot load are
rejected before download. Legacy Splash packages such as
`incoai/Qwen3.8-27B-Splash` remain loadable.

Vision comes from the same source: embedded vision tensors for MLX, or the
repository's companion BF16 or F32 `mmproj` GGUF. Both are prepared as
BF16; an F32 or F16 tensor loads only when every value is exactly a BF16, as in
Unsloth's mmproj files.
Use `--language-only` to skip vision loading and preparation. It also skips the
GGUF mmproj download; MLX vision tensors share the language model's shards, so
those shards still download in full. The server then rejects image and PDF input
and reports `vision: false` in `/status` and `/v1/models`.

The prepared weights live in `~/Library/Caches/Splash/weights`
(`SPLASH_WEIGHT_CACHE` relocates them); preparation uses bounded temporary
memory, and later starts reuse the result. Each start checks the upstream
revision with one Hub request of at most 5 seconds and installs a new commit
before serving it; without the Hub, or when the new commit cannot be installed,
the installed model starts. `--revision` selects an upstream branch, tag or
commit (a commit is never checked again); otherwise the default branch is
followed. Private repositories need `HF_TOKEN`. Downloads use the Hugging Face
cache, and `brew upgrade splash` preserves models and agent sessions.

For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine):
install the Splash runtime, then paste the full Hugging Face model link into its
model search. These integrations manage their own runtime and settings.

If a client’s model catalog does not list a model, the full model ID in
this table still works with `splash serve --model OWNER/REPO[:VARIANT]`. The browser chat
and the agent launchers connect to that server without a catalog search.

For a custom model download location, see [model cache](DEVELOPMENT.md#model-cache).

## Settings

There is no config file. The server binds `127.0.0.1:8000` by default.
Context supports up to the model’s native 256K window; usable capacity
depends on available memory. `splash serve --help` lists server options and examples.
The startup summary and `maximum_context_tokens` in `/status` show the effective
server limit. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model aliases.
Clients can impose a smaller limit. With enough memory,
request the full window using `--max-context 256K`. This is a capacity limit, not a guarantee
that a long uncached prompt will reach its first token quickly.

`splash serve` accepts these optional flags:

- `--host`: HTTP bind address. Default: `127.0.0.1`.
- `--port`: HTTP port. Defaults to `SPLASH_PORT` or `8000`.
- `--max-memory`: ceiling on Metal allocations, e.g. `28G`. Default: auto.
- `--max-context`: context limit, up to `256K`, e.g. `100K`. Default: auto.
- `--kv-format`: target KV cache storage, `int8` (default) or `bf16`.
- `--max-image-pixels`: maximum resized pixels per image. Default: 4,194,304.
- `--allowed-host`: extra HTTP `Host` name to accept, not a bind address. Repeatable.
- `--api-key`: require this key on API requests, as a bearer token or
  `x-api-key`. Defaults to `SPLASH_API_KEY`.
- `--no-webui`: turn off the chat page.

To use BF16 target KV, select it when starting the server:

```bash
splash serve --model mlx-community/Qwen3.8-27B-4bit --kv-format bf16
```

BF16 avoids target KV quantization, uses approximately twice the target KV
memory, and can be slower at long contexts. Model weights are unchanged.
Restart the server to switch formats. Omit `--kv-format` or use
`--kv-format int8` for the default INT8 cache.

If the model does not fit in the memory available, startup prints a memory
budget breakdown and stops.

Authentication is off by default. Set `SPLASH_API_KEY` in the shell that runs
`splash serve` and in the shell that runs an agent, and both sides use it.
Health and readiness probes stay public.

For LAN access and multiple servers, see
[server configuration](DEVELOPMENT.md#server-configuration).

Experimental cache offloading: [PR #3](https://github.com/incoai/splash/pull/3).

## Performance

Measured for the Splash 1.0 release (September 2026) on an M5 Pro (16-core
GPU, 48 GB), serving the Qwen3.8-27B and Qwen3.6-35B-A3B Splash packages:
selected SPEED-Bench coding prompts over HTTP, a 1,024-token output limit,
reasoning on (medium for the 27B). The ratio in each cell is against the
next-fastest engine we measured. The MLX 4-bit models prepare to the packages'
target weights, byte for byte but for the 27B's 48 per-layer GDN decay vectors,
each within a float ULP, and decode within 0.5% of them on this M5 Pro
([upstream loading](dev/benchmarks/upstream-loading.md)). GGUF targets run
other kernels and are not part of this comparison.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash 1.0 led on every measure at every prompt length we tested, and the lead
grew with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

For repeatable measurements on your Mac, see [local benchmarks](DEVELOPMENT.md#local-benchmarks).

## Design

The runtime, scheduler, cache, and API are shared. Everything else is rebuilt
per model:

- **A draft trained for the model.** Speculative decoding is the decode path in
  Splash, not an option. Each supported base model has a matching [DFlash
  2](https://inco.ai/blog/dflash2/) draft, and one pass of the target verifies a
  block of tokens in parallel.
- **Kernels for the model's shapes.** Fused Metal kernels for the models'
  attention, GDN and MoE dimensions, with dispatch policies measured offline
  per GPU family and core count. MLX weights are prepared once into layouts
  packed for these kernels. GGUF weights keep their llama.cpp quantization,
  repacked once into planes that kernels chosen by GPU family and core count
  decode directly, without per-shape tuning. Both are mapped zero-copy from
  disk. Everything ships precompiled: no Xcode, no compiler toolchain, nothing
  tuned on your machine.
- **A memory plan computed for this machine.** Context, KV capacity, and batch
  limits are worked out at startup from the memory Metal recommends, less the
  weights, the draft, and each request's state.

The [launch post](https://inco.ai/blog/splash/) covers the design in depth.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, model loading, tests
  and release packaging.
- Apache-2.0, see [LICENSE](LICENSE). Model weights keep their own licenses.
