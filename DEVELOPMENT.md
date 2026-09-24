# Development

Use Apple Silicon with macOS 26.4+, Xcode 26 or newer, Python 3.12–3.14,
and a Metal 4 compiler with `uint4b_format` tensor support.
The macOS 26.2 SDK can compile the host code, but Xcode 26.2's default Metal
component cannot compile the kernels; select a newer Metal toolchain when
using that SDK. Packaged users need none of these development tools.

## Build and run

```sh
git clone https://github.com/incoai/splash.git
cd splash
make -j4
./splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`--model` names an upstream Hugging Face model: an MLX affine 4-bit, group-64
repository such as `mlx-community/Qwen3.8-27B-4bit`, or a GGUF repository and
variant, `OWNER/REPO:VARIANT`, such as `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.
Splash identifies the model from its own metadata and pairs the DFlash2 draft
trained for it ([Upstream model loading](#upstream-model-loading)). The first
serve sets up Python dependencies, downloads the model and its draft, and
prepares the weights once, which takes about the model's size again on disk
([Model cache](#model-cache)). Each start checks the Hub for a newer revision
and follows it; without a network, the installed model starts as is. Existing
Splash packages such as `incoai/Qwen3.8-27B-Splash` remain loadable
([Splash packages](#splash-packages)). Public repositories need no login;
private or gated ones need `HF_TOKEN` or `hf auth login`. Ctrl+C stops serving;
stop before upgrading.

Use `--max-context 100K` or `--max-memory 28G` to set optional limits. Memory
limits cap Metal allocations, not combined process RSS. Agents must already be
installed; `./splash claude|opencode|codex|hermes` connects to the running server.
Arguments pass through, for example `./splash codex resume --last`.

Set `SPLASH_API_KEY` in the server and agent shells to require authentication;
`serve --api-key KEY` overrides the server's environment value. API requests
then require `Authorization: Bearer KEY` (or Anthropic's `x-api-key`). Health
and readiness probes and the chat page remain public; enter the key in the
chat page to send requests. The page does not persist the key. Use
`serve --no-webui` to disable the page. Authentication is off by default.

HTTP request bodies are limited to 128 MiB; `serve --max-request-size 256M`
overrides this. Concurrent input bytes share a budget of at least 512 MiB
(or twice the request limit), including retained generation inputs. This is
an input-byte budget, not a process RSS limit: large ASCII/base64 strings can
use roughly twice their encoded size during JSON parsing alone. Decoded images
and object-heavy JSON need additional memory. Oversized requests return 413;
exhausted ingress capacity returns 503. Image and model context limits apply
independently.
Stored Responses history is charged before decoding. Uploads allow 30 seconds
of inactivity; total upload time is limited to 30 seconds plus the body size
at 512 KiB/s (286 seconds for 128 MiB), capped by the overall request deadline.
Timed-out uploads return 408 and release their input reservation.
`/status` reports `http.request_body_bytes` and `http.max_request_bytes`.

Source `install/completions/splash.bash` for Bash or
`install/completions/_splash` for Zsh after `compinit`. Completion suggests
commands, bundled official model IDs and installed models without network access.

## Server configuration

The default listener is `127.0.0.1:8000`. To accept LAN connections:

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --host 0.0.0.0 --api-key YOUR_KEY
```

Connect to the server's LAN IP. `--host` selects the IPv4 bind address;
`--allowed-host NAME` accepts an additional HTTP Host name, such as a custom DNS
name or proxy hostname. It does not change the listener or allowlist client IPs.

Use `--port 8001` or set `SPLASH_PORT=8001` to select another port. Set the same
`SPLASH_PORT` in the local agent shell. Separate ports allow separate servers;
their memory limits are independent. The packaged agent launchers connect to
loopback, so use a listener that includes loopback when launching agents locally.

## API model aliases

Repeat `--served-model-name NAME` to accept additional API model IDs. The full
`--model` ID still selects the model. `/v1/models` lists that ID first, followed
by unique aliases; each alias's `root` identifies the loaded model. Generation
and scoring responses always report the real model ID, even when requested
through an alias. The model list and lookup support both names.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --served-model-name local-qwen
```

Aliases cannot contain whitespace, control characters, `\`, `%`, `?`, `#`,
or empty, `.` or `..` path segments. This keeps model discovery URLs unambiguous.

## Default reasoning effort

`--default-reasoning-effort` (or `SPLASH_DEFAULT_REASONING_EFFORT`) sets the
fallback for Chat `reasoning_effort` and Responses `reasoning.effort` when absent
or null. Accepted values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`. An explicit request value wins; the CLI flag takes precedence over the
environment. Unset, the model's template default is unchanged. Effort names are
passed to the template using the same mapping as per-request values, not token
budgets.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --default-reasoning-effort none
```

`/apply-template` uses the same default. Anthropic `thinking` keeps its protocol
semantics (off when omitted); judgment endpoints always disable thinking.

## Model cache

To download new models to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`HF_HUB_CACHE` selects the Hugging Face download cache. Alternatively, set
`HF_HOME` to relocate the Hugging Face home directory, including its default
`hub` cache. Model links and agent sessions stay in Splash's data directory;
existing downloads are not moved. Prepared weights have their own cache
([Local weight preparation](#local-weight-preparation)), so moving downloads
does not move them.

## Splash packages

Splash packages are the prebuilt format that predates upstream loading. They
contain `manifest.json`, packed `target/`, `draft/`, `vision/` weights and
`tokenizer/`. The manifest lists artifact paths, sizes and SHA-256 hashes.
Dense packages use schema 3 / `splash-packed-q4`; MoE uses schema 4 /
`splash-packed-q4-moe`.

These formats encode Qwen3.8-27B and Qwen3.6-35B-A3B layouts. Compatible community
fine-tunes may use any nonempty manifest model name. Native loading validates
geometry, tensor sizes, binary headers, tokenizer and target/draft compatibility.
New architectures require engine support.

### Upstream model loading

`install/upstream.py` resolves the target repository and the draft paired with
it, then atomically publishes a local assembly of links to their Hub
snapshots. `model.json` describes those resolved sources and selected formats;
it is local installation metadata, not a file model publishers must supply.
A target is identified by its own metadata: an MLX config's `text_config`, or
the one `gguf.model_config` derives from the selected GGUF's header, read with a
few HTTP range requests before any weight download. The registry (`FAMILIES`)
states each supported architecture's signature and the draft trained for it;
repository names and model-card `base_model` fields play no part. The same
header rejects a GGUF whose tensor types the native loader cannot read
(`gguf.LOADABLE_TYPES`, checked against `metal/abi/QuantFormat.h`) before it is
downloaded. Native source adapters validate model geometry, quantization, tensor
shapes and draft compatibility again before execution.

Examples:

```bash
splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only
```

Installation resolves each source's revision once to a commit, downloads by
that commit and records it in `model.json`, so a repository update cannot mix
files from different revisions. It pins those snapshots in the Hub cache
(`refs/splash/<installation>/<commit>`), so pruning the cache cannot remove files
an installed model links. Every start resolves the target's revision (the
default branch, or `--revision`) with one Hub request of at most 5 seconds; if it
is the installed commit, `prepare` checks the assembly's links, sizes and times
and returns. A new commit is followed: only changed files are downloaded, and
its assembly replaces the installed one atomically once published. When the Hub
cannot answer, or the new commit cannot be installed, the installed assembly
that verifies starts instead, with a message naming the reason. A commit
`--revision` never moves and `HF_HUB_OFFLINE=1` forbids the Hub, so both start a
verified installation without a request. An assembly that no longer verifies is
rebuilt; without the Hub it is rebuilt from a cached snapshot, of the commit the
`--revision` names, or else the one the installation recorded or pinned, or one
the Hub cache records for the branch, never of another revision. Only files
downloaded before are available, which is enough to rebuild a damaged or
deleted assembly or to add a selection of a cached commit. When a release pins
another draft for a family, the next start downloads it and re-assembles the
installed target commit; if the draft cannot be fetched, the installed one is
kept.
The installer never rewrites upstream files. Older manifest-based packages use
the legacy installer.

MLX configuration, tokenizer, chat template and processor come from the same
resolved target snapshot. For GGUF, `install/gguf.py` reads the selected file's
metadata without mapping or decoding weight tensors. Vocabulary IDs, BPE merge
ranks, control/user-defined token types, BOS/EOS/padding IDs and template text
come from that file. The supported `gpt2/qwen35` profile supplies the NFC and
byte-level pre-tokenization algorithms. Unknown profiles and malformed metadata
are rejected, with no cross-repository fallback. GGUF sidecar tokenizer/config
files do not override embedded metadata.

Model geometry is translated from GGUF metadata, subtracting any declared MTP
layers from the transformer layer count. Vision configuration and preprocessing
come from the same snapshot's mmproj. Native loaders independently validate the
model geometry and all tensor shapes. Derived metadata is cached under
`models/.metadata`, keyed by source identity, adapter code and tokenizer-library
version; publication is atomic and cache contents are hash-checked. Only the
header is read before the download; the tokenizer and configuration are derived
from the downloaded file.
Remote Python code is not loaded. Vision uses MLX's `vision_tower.*` tensors,
linking only `config.json` and the shards holding them, or the same GGUF
repository's `mmproj-BF16.gguf` or `mmproj-F32.gguf`; F16 has a narrower exponent
than BF16, so an F16 projector has already rounded small weights. Both source
adapters prepare the packed `vision/model.bin` layout, which the one BF16 vision
operator reads: BF16 tensors are copied, and F32 or F16 tensors are converted only
when every value is exactly a BF16. Otherwise preparation fails, naming the tensor
and file. Unsloth's mmproj stores its 1-D tensors, patch embedding and position
table as F32, all of them BF16-exact, and prepares byte-identical to the packed
file. Quantized MLX towers, deepstack projectors and mmproj tensors the tower does
not use are rejected. `--language-only` removes vision weights from startup and
memory accounting. The native Ready event announces vision only when the model
loaded it. Without it, image and PDF input fails with a 400 naming the modality.
Every API shape converts its media to image and file parts, and message
normalization, the one place that accepts or rejects them, checks before any
image is decoded or PDF rendered, in user turns, tool results and stored
Responses history alike. `/status` and `/v1/models` report `vision: false` and
`input_modalities: ["text"]`, and the launchers configure OpenCode and Hermes
without attachments.

### Draft assets

Splash's DFlash2 drafts share one Hub repository, `upstream.DRAFTS`
(`incoai-internal/Splash-DFlash2`), with a folder per base model named after it:
`Qwen3.8-27B/` and `Qwen3.6-35B-A3B/`. Each folder holds `config.json`,
`model.bin` and `layer-N.bin`. The configuration is the original DFlash2
configuration with `splash.format = "MDFD0004"` and `splash.source`, the DFlash2
checkpoint the weights came from; native loading validates it against the
target. These are the existing verified Q4 draft weights, not a new
quantization of the draft at startup. Each family pins the commit that published
its folder (`Draft.revision` in `FAMILIES`), and installation downloads only that
folder.

To prepare a folder from a verified existing package:

```bash
python dev/tools/export_draft.py PACKAGE ORIGINAL_DRAFT_CONFIG DRAFTS/Qwen3.6-35B-A3B
```

The exporter verifies existing artifact hashes and copies only draft files.
`--draft-model` accepts such a folder, a local copy of the whole repository, or
another Hub repository with the same layout.

### Upstream tokenizer and chat templates

The server uses the upstream tokenizer and chat template. Request preparation
merges the leading system and developer messages into one system message, joined
by a blank line: Responses instructions and developer items, or an Anthropic
`system` and a leading system message. A system message after that, as agent
clients send when they change instructions during a conversation, renders where
it occurs as a system turn in the template's own markup.

`server/chat_templates.py` probes each of the tokenizer's templates, including
each named variant such as `tool_use`, once at startup: it renders a canary
conversation whose later system message carries a marker. A template that renders
it in place is used unchanged (`native`). The official Qwen templates raise for
it, and Unsloth's Qwen3.6 GGUF template skips it. Such a template is patched at
the construct responsible, found by parsing its tags: the `raise_exception` in
the message loop's system branch, or the loop condition that excludes system
messages. The patch renders the message with the block the template gives a
leading system message (`patched`), and is kept only if ordinary conversations
(with and without tools, every reasoning effort, preserved thinking, tool calls
and results, images) still render byte-identically and the canary renders in
place. Otherwise a request with a later system message fails with a 400 instead
of losing it (`unsupported`). Startup logs the outcome once and `/status` reports
it as `chat_template.later_system`. Every request, including image placeholder
and token-count rendering, uses the template chosen at startup; tokenizer files
and the tokenizer object are unchanged.

`response_format` constrains generation and validates final output; it does not
inject formatting instructions into the prompt. Clients should describe their
output requirements in their own messages.

### Local weight preparation

`AffineTarget` reorders codes, scales and biases into the existing affine ABI
without requantization. GDN decay is computed as `float(-exp(double(A_log)))`;
older packages produced using MLX's float exponential may differ by a few float
ULPs in this small vector. The existing inference kernels are unchanged.

Target and vision source adapters use `PreparedWeights`. Its default cache is
`~/Library/Caches/Splash/weights`; `SPLASH_WEIGHT_CACHE` overrides that location.
Preparation costs an additional on-disk copy of the prepared target and vision tensors. Existing
packed artifacts are used directly. Source-content hashes, a build-generated fingerprint of preparation code and its
storage ABI, and transformation parameters identify the cache. An MLX vision artifact hashes
`config.json` and only the shards holding `vision_tower.*`. Changing that
code invalidates its artifacts automatically; unrelated app releases, core
counts and support-asset updates do not. Completed files are read-only.
One writer per cache serializes conversion; complete cache hits bypass this lock.
Interruption, disk-full errors and memory-pressure rejection cannot publish partial
files. Before conversion, the adapters validate the whole source and budget every
missing target and vision artifact plus a 2 GiB disk reserve. Each output's disk space is preallocated
before writing. Concurrent external disk activity can still exhaust the volume;
write errors leave no published partial artifact. Retrying removes abandoned writes
under the converter lock and reuses previously completed layers.

Cold preparation reports each artifact's progress. Each new cache entry records
its source path and artifact name in `source`. There is no automatic eviction:
completed entries can be shared by installations and pinned source revisions.
With Splash stopped, unused entry directories can be deleted; deleting the whole
cache causes preparation at the next load. Uninstalling one model does
not delete possibly shared prepared weights.

Cold source hashing and output validation stream bounded buffers. Unchanged
files reuse a digest proof tied to device, inode, size, birth time, mtime and ctime;
a write or replacement invalidates it. This is not a full disk scrub on every
startup. Preparation uses uncached destination I/O and bounded tensor tiles,
with a 64 MiB admission reserve for staging and capped metadata. The input/output
staging buffers together stay within 32 MiB. Complete rows and multiple row tiles
are processed together where possible, avoiding per-row I/O and small GPU waits.
Warning/critical memory pressure or inadequate host headroom stops conversion.
Cache hits and bounded source verification use normal startup admission instead;
cancellation and critical pressure still abort loading. Runtime admission
counts prepared weights, draft and vision exactly once. File backing removes the
whole-model anonymous repack allocation. It does not make Metal-resident pages
reclaimable: residency can wire them until released. macOS page cache, driver
allocations and other applications still affect memory pressure and swap.

Operator plans use each projection's physical layout, independently of the source
container. `Projection`, `MoeWeights` and `EmbeddingWeights` represent different
operator contracts. Arena sizing collects the actual layer layouts, including
mixed affine/block layers, and reserves the vocabulary head only for decode.

### GGUF targets

Qwen3.8-27B and Qwen3.6-35B-A3B can be served straight from a llama.cpp GGUF:
`splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` selects the root-level file whose
name ends in `-UD-Q4_K_M` and installs it as described in
[Upstream model loading](#upstream-model-loading).

Prebuilt `gguf` Splash packages ship only the shared `draft/`, `vision/` and `tokenizer/`;
`schema_version` 3 describes a dense Qwen3.8 target and 4 a Qwen3.6 MoE target, and the loader
checks the GGUF's architecture against it. Their manifest names the source repository and the
files a model ID may select:

```json
"format": {"name": "gguf", "target_layer_magic": "MDGG0001", ...},
"target": {"gguf": {"repo_id": "unsloth/Qwen3.8-27B-GGUF", "revision": "<commit>",
                    "variants": {"UD-Q4_K_M": {"file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                                               "size": 16464440224, "sha256": "..."}, ...}}}
```

For such a package, `--model OWNER/REPO:VARIANT` downloads the shared files and that one GGUF
into the Hub cache, checks them against the manifest, and installs
`models/<owner>/<repo>:<variant>/` as a real directory of per-file symlinks whose
`target/<file>.gguf` links the cached GGUF (the engine requires `target/` and `draft/` to be
subdirectories of one root). The original download remains unchanged.

At load time the engine validates the GGUF metadata and plans the existing
`MDGG0001` layout. `GgufPreparation` sizes row batches to a fixed 32 MiB total
staging budget, uses contiguous reads where possible, runs the existing repack
kernel, and writes its planes into a prepared file. Rows wider than this budget
are split into column chunks. Embeddings and F32 sections use bounded direct copies. `PreparedWeights`
publishes only completed files; `WeightFile` maps them read-only without copying
into a model-sized Metal allocation. Later starts reuse these files.

Every tensor keeps its stored format: the F32 norm multipliers,
the MoE router and shared-expert gate, and GDN alpha/beta when a file stores them as F32 stay F32
and run in fp32, as llama.cpp keeps them (Apple10 prefill chunks multiply the router and
alpha/beta on the neural accelerator as three bf16 parts per weight that sum to it exactly, so
only fp32 accumulation rounds); the other small F32 tensors (the GDN convolution and
time-step bias) become bf16 only when every value converts exactly, and loading fails otherwise.
Prepared weight files are file-backed like existing packed packages; while Metal holds them
resident, their pages stay wired. Supported tensor types are Q4_K,
Q5_K, Q6_K, Q3_K, IQ4_XS, IQ4_NL, Q8_0 and IQ3_S for linears and experts, F32 for the tensors
above, and Q4_K, Q6_K or Q8_0 token embeddings; the loader lists every unsupported tensor in one
error. Of Unsloth's files that covers, for Qwen3.8-27B, UD-Q4_K_M and every larger file but
Q4_1, UD-Q8_K_XL and BF16, and for Qwen3.6-35B-A3B, UD-IQ4_XS and every larger file but
MXFP4_MOE, UD-Q8_K_XL and BF16. The smaller files need IQ3_XXS, IQ2, IQ1 or Q2_K kernels and the
others Q4_0/Q4_1, MXFP4 or BF16 ones, which do not exist yet.

Decode runs one of two kernel families, chosen by GPU family in `runtime/ops/LinearGguf.cpp`. On
Apple9 (M3, M4) the register kernels of `runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal`
feed the codes themselves to bf16 matrix operations with one fp32 epilogue per coefficient
group, so every output is the bf16 rounding of its fp32-accumulated sum. On Apple10 (M5) the
staged kernels of `runtime/metal/kernels/shared/gguf_linear.metal` dequantize each weight once
to half in threadgroup memory for MPP `matmul2d`, the neural accelerator's path; a step of three
request lanes runs the 32-row tile over four lanes of storage. Prefill runs the staged kernels
on both families, chunks of up to 32 rows on the decode tiles. Every projection splits its K
across threadgroups by one rule (`decodeSplits`: each tile's tiers of threadgroups per core and
inputs per partition, from measured occupancy) that does not depend on the batch width. The MoE
experts (`runtime/ops/MoE.cpp`) run the same numerics per family over the grouped rows. The ABIs
are in `runtime/metal/abi/Gguf.h` and `MoE.h`, the image formats in
`runtime/metal/abi/QuantFormat.h`, their decoding in `runtime/metal/kernels/common/quant_formats.h`.

The tests' CPU reference (`dev/tests/engine/GgufFormatReference.hpp`) must reproduce checked-in
hashes of upstream GGML's dequantization (llama.cpp 7ab4ee7) in `make test-engine-cpu`, which
also checks the planner's CPU-built tensors against it; `make test-engine-metal` checks the
production `gguf_repack` and `gguf_copy` kernels bitwise against it, the projection kernels
(`gguf-projection full`) and the MoE layer in every format (`gguf-moe`) against fp64, and the
F32 norm, router and alpha/beta paths. With `SPLASH_GGML_ORACLE=<libggml-base.dylib>` the
reference is also compared with GGML directly and `gguf-repack --cpu` prints GGML's hashes.

## Code and API boundaries

- `server/`: OpenAI Chat/Responses, Anthropic Messages/count_tokens, typed
  judgments, templates, streaming and input processing. No client-version branches.
- `runtime/engine/`: scheduling, memory admission and reusable request state.
- `runtime/model/`: target/draft execution and vision.
- `runtime/ops/` and `runtime/metal/`: operators and Metal kernels.
- `install/`: launcher, client configuration and model installation.
- `dev/`: maintained tests, benchmarks and build/release tools.

Within `server/`, `server.py` owns HTTP and startup; `frontend.py` prepares
requests and history; `backend.py` owns native request lifecycles. `judgments.py`
owns finite-choice prompts, validation and typed answer math. `output.py` parses
generated text for both streaming and complete responses, and `constraints.py`
compiles token constraints. `make architecture-check` prevents lower layers from
importing the HTTP entry module.

Tools can be combined with structured answers. Tool argument framing resolves
local references and projects object fields through schema composition. The
original schema validates complete arguments, including cross-field conditions,
dependencies and property-count rules that framing alone cannot enforce. Extra
properties use JSON-encoded values; statically typed strings retain raw text.
Remote schema references and parameter names containing XML delimiters are
unsupported. Hosted search is unsupported; configure client-owned tools such as
MCP. Omitted effort uses the model default.
Hidden thinking signatures use a persistent user key; imported encrypted thinking
preserves visible history without recovering the private reasoning.

`/status.admission` distinguishes memory and concurrency waits, reports suspended
requests, recovery draining and the oldest current wait age. Memory transitions
also appear in the console. Warning pressure can pause growth while `/ready`
remains healthy for work that fits existing allocations.

PDF input supports base64 documents within a shared 64 MiB source/rendering
budget and the native 64-image limit (one image per page). Model context and
isolated rendering limits also apply. URL inputs, opening passwords and citations
are unsupported.
Responses automatic truncation and unsupported history edits return errors.

`POST /tokenize` accepts `{"content":"hello","add_special":false}` and returns
`{"tokens":[...]}` using the loaded tokenizer. Special-token strings are recognized;
`parse_special:false` and `with_pieces:true` are unsupported.
`POST /apply-template` accepts Chat-style `messages`, `tools` and reasoning options,
and returns `{"prompt":"..."}` using the same template as generation.
`add_generation_prompt` defaults to true. Image prompts retain textual placeholders;
raw tokenization does not account for image embeddings (use `count_tokens` for that).
Both endpoints run without inference and share bounded preparation capacity with
`count_tokens`; they can inspect prompts larger than the serving context limit.

Streaming requests accept `"return_progress":true` (default false). Before output,
`prompt_progress` reports `{total, cache, processed, time_ms}`: prompt tokens,
initial cached tokens, completed tokens including cache, and elapsed milliseconds
since prefill admission. Updates follow completed chunks and never regress during
recovery; they are not a time estimate. Chat uses empty-delta chunks, Responses
uses `response.in_progress`, and Messages uses `ping`. Queueing and prompt
preparation do not advance this counter. Non-streaming requests cannot enable it.

`GET /status` returns instance identity and the effective context limit as JSON.
Proxy consumers can use these fields; additional fields may be added:

| Field | Meaning |
| --- | --- |
| `requests.submitted`, `completed`, `cancelled`, `failed` | Native request counters since engine start |
| `memory_actual.current_bytes`, `peak_bytes` | Metal allocations, not process RSS |
| `metrics.decode_tokens_per_second` | Aggregate native decode throughput, not a request's end-to-end rate |
| `maximum_context_tokens` | Declared context limit; available memory may limit admission |
| `vision`, `input_modalities` | Whether image and PDF input is accepted; `false` and `["text"]` after `--language-only` |
| `chat_template.later_system` | `native`, `patched` or `unsupported`: how system messages after the first render (per name for named templates) |

`GET /metrics` exposes the same counters in Prometheus text format. Both endpoints
require the API key when authentication is enabled. Consumers should tolerate
missing native fields while the engine is unavailable, and counter resets after
an engine restart. Chat streams include token usage when the request sets
`"stream_options":{"include_usage":true}`; non-streaming Chat responses always
include usage. A proxy must consume these fields to display statistics.

Chat completions also include a llama-server-style `timings` object, both in
non-streaming responses and in the final finish-reason chunk of a stream,
even without `include_usage`. `prompt_n` and `predicted_n` are the full prompt
and output counts; `cache_n` is the cached prompt count. `prompt_ms` measures
native start to first emission, and `predicted_ms` measures first emission to
completion. These elapsed intervals exclude the initial admission queue and
are not isolated GPU timings. `prompt_per_second` uses only uncached prompt
tokens; `predicted_per_second` excludes the entire first emission (which can
contain multiple speculative tokens). Thus rates use tokens processed in the
measured interval, not the full counts. An unavailable rate is zero, including
responses completed in one emission. Per-request draft counters are omitted
because the native runtime only reports them at batch level.

`/metrics` also exports fixed latency histograms in seconds, with a bounded
set of stages in `/status.latency`. HTTP duration includes body upload and
response writing for admitted API requests. Preparation, queue, template,
tokenization, output grammar preparation and image preparation are measured
separately; preparation includes its nested stages. Tokenization covers the encoding call, including reuse when
available. Histogram buckets are cumulative and labeled by upper bound.
TTFT starts before upload and ends at the first native token
event. Output intervals are between native token events, which can contain
multiple speculative tokens; they are not per-token latency. Native queue timing
is recorded from successful completions. These histograms live with the HTTP
process and survive a native engine restart.

HTTP bodies require Content-Length, and browser
Origin must match Host. `--allowed-host` permits additional hostnames. Request
logs omit bodies; full crash traces require explicit `SPLASH_CRASH_TRACE=1` and
can contain private conversation data.

Requests sharing a cold prefix can wait for a resident request's planned recovery
point, then enter through the ordinary cache restore path. Waiting requests hold
no active state cell or KV pages and return to ordinary admission when no useful
producer remains. Late arrivals can extend the plan at complete state boundaries.
Higher-priority work does not wait for a lower-priority producer. `/status` exposes
`scheduler.waiting_prefix` separately from resource waits.

Greedy and sampled requests can share an unconstrained decode batch; each lane
keeps its own sampling policy and RNG. Pure greedy batches retain their argmax
path. Constrained requests use a separate batch for the host mask exchange.

Long prefill uses disposable rolling checkpoints every 4096 tokens. Contended
prefill adapts toward a 500 ms slice, keeping 2048-token chunks for long unopposed
work. These policies do not extend client deadlines. Memory recovery waits are
bounded, but readiness does not guarantee that a request-sized allocation fits.

### Judgment contracts

`POST /v1/systemone` accepts the [TypeSafe System One](https://docs.typesafe.ai/)
request and response shapes: `noul`, `choice` and `score` questions over a shared
state. It works with the official `typesafe-sdk` (verified with 0.7.0). Use the
actual served model ID, not a hosted Jev model name; `/v1/models` answers both
OpenAI model discovery and the SDK's `models.list()`.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="local",  # Use SPLASH_API_KEY's value if server authentication is on.
    model="mlx-community/Qwen3.8-27B-4bit",
) as client:
    result = client.system_one(
        state={"message": "I was charged twice. Please fix this today."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": None, "technical": None, "sales": None},
            ),
            "urgency": Score(
                instructions="How urgent is the request?",
                criteria=["No urgency", "This week", "Today"],
            ),
        },
    )
    print(result.choices["department"].choice)
```

`POST /v1/judgments` scores one [SemIf](https://github.com/TheoLeeCJ/SemIf) row of
2–16 options and returns raw option logits:

```bash
curl http://127.0.0.1:8000/v1/judgments \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "approval",
    "state": "The proposal is awaiting approval.",
    "question": "What is the current approval status?",
    "options": [
      {"id": "approved", "description": "Approval was explicitly given."},
      {"id": "pending", "description": "Approval has not been given."}
    ]
  }'
```

`POST /v1/judgments` preserves SemIf's `direct-options-v1` JSON serialization,
system prompt and A–P option order. It returns the exact rendered prompt's SHA-256,
answer token IDs, raw option logits, normalized probabilities and zero completion
tokens. Every answer label must round-trip as one token, including at the actual
assistant prompt boundary. Unsupported generation controls return errors rather
than silently changing the scoring protocol. SemIf-derived code retains its MIT
notice in `server/judgments.py`.

`POST /v1/systemone` requires the served `model`, a string/object/array `state`,
and a nonempty `questions` map. Instructions may be omitted, null or structured;
criteria descriptions may also be structured. Noul criteria may be omitted.
Choice and score domains contain 1–255 entries. Singletons return their sole
answer without inference. Other domains use deterministic, distinct single-token
slots selected from the tokenizer. All questions are validated before any inference.
A request holds at most 64 questions and 1M total prepared prompt tokens;
larger batches are rejected before any inference.
Questions run sequentially within a request under one shared deadline, allowing
prefix reuse without filling the admission queue; independent HTTP requests still
share the scheduler. Disconnects and timeouts cancel the current question.

Preparation renders each prompt once, then enforces the context limit and the
batch token budget before the per-slot boundary checks, which re-tokenize the
prompt once per option. Those checks also observe the request deadline, so an
oversized or expired request is rejected without paying for every option.
Prompts that exceed the context limit are rejected, not truncated.

System One validation uses 422 `detail` arrays; successful responses contain
`model`, `answers`, and `usage`, plus an `x-typesafe-request-id` header. SDK model
discovery reports an empty `release_date` because packages do not record one.
The official SDK is a client only, not a server dependency. API compatibility does
not imply Jev weights, accuracy, proprietary confidence semantics or calibration.

These are local model scores, not calibrated confidence. Probabilities are a
softmax over the declared answer slots. Choice/score `confidence` is normalized
entropy concentration, `1 - H(p) / log(K)`, not an estimate of correctness.
Score answers are probability-weighted level indices. Measure accuracy and
calibrate on representative held-out data before using decision thresholds.

Native wire version 6 appends score-token IDs to requests and selected f32 logits
to Done events; a version mismatch is fatal. Scoring requires 2–255 distinct,
in-vocabulary tokens, no images or generation constraints, and a zero output budget.
It may use the full context window because no generated token needs a reserved
position. The final prefill chunk runs the target head but no sampling policy or
DFlash decode. Successful scoring emits no Tokens event, finishes with Stop, and
reports zero decode time. Cancelled requests carry no logits.

A non-finite score logit is a per-request failure, not an engine fault: the
engine reports `model_result_invalid` for that request alone, before it
publishes the failing step's cache state or any output, and the rest of the
batch finishes normally. Prompt chunks that already succeeded keep the blocks
they committed, exactly as they do for a cancelled request. GPU faults and
broken engine invariants stay fatal and still mark the runtime unhealthy.

## Validate

```sh
make check
make install test-real test-http-real MODEL=incoai/Qwen3.8-27B-Splash
```

`make check-native-cpu` builds production and runs native CPU tests without a
GPU. `make check-native-metal` requires a supported Metal device; `make check`
includes both. Hosted CI runs CPU checks and sanitizers; the hardware release
gate runs the full suite.

Repeat model tests with `MODEL=incoai/Qwen3.6-35B-A3B-Splash`.
Before release, install all four agents and run `make release-check MODEL=...`
for both models from a clean checkout. It includes correctness, sanitizers,
real HTTP/client behavior and performance checks.

Compare performance on the same idle Mac with the same model and workload.
`make tune-kernels MODEL=...` measures kernel policies. Keep generated reports,
profiles, local paths and experiment notes out of the source tree and commits.

### Local benchmarks

From a source checkout with the model installed, use the existing native
benchmark for prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=incoai/Qwen3.8-27B-Splash
```

It writes `build/release/backend-benchmark.json`. Repeat with the 35B package
for that model. This characterizes one build; it is not a comparison with
another engine or a test of agent task quality.

For a same-machine HTTP regression check, retain the previous `splash` binary
**and its adjacent `splash.metallib`**, then run from the candidate checkout:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model incoai/Qwen3.8-27B-Splash \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

This starts isolated servers in alternating order, compares matched cold,
exact-prefix and decode requests, and saves `build/release/http-regression.json`.
It does not contact your running server. Use the same power mode and charger,
stop other GPU workloads, and report chip/GPU cores, memory, Splash version,
model revision, actual input/output token counts, and cache hits with results.
Keep cold prefill, cached TTFT and sustained decode separate; a UI token rate
alone does not measure end-to-end agent performance.

For slow tool-bearing requests, the `latency` section of `/status` separates
preparation, tokenization, grammar preparation, native queueing and TTFT. Grammar preparation
includes construction, compilation/cache lookup and per-request cloning;
it does not include generation-time masks. The `grammar_cache` counters show
whether compiled output grammars are reused. Tool definitions still contribute
tokens to the prompt; saving their JSON alone cannot avoid model prefill.
Existing exact-prefix caching reuses model work while the server remains alive.
Text requests also reuse tokenized history at literal message-end boundaries
when the tokenizer supports independent encoding there. This process-local
cache retains at most four prefixes and 8 MiB of text/token storage; it falls
back to full encoding for other tokenizer pipelines. `/status.tokenizer_cache`
reports its usage. It does not alter prompt text, token IDs or the GPU KV cache.
Server restarts require recomputation. The separate
[SSD cache proposal](https://github.com/incoai/splash/pull/3) preserves evicted
model state during a server session; its temporary files do not survive shutdown.

## Package

Release archives contain no Hugging Face credentials and use the official model
list committed with the source. The model-catalog workflow updates that list
from the official collection independently of packaging.
Users accessing private models supply their own `HF_TOKEN` or Hugging Face login.

Release versions are three-part, `x.y.z`, with no `v` prefix: `1.0.0`, then
`1.0.1` for a fix and `1.1.0` for a feature. Use the same version in all three
commands:

```sh
make package RELEASE_VERSION=1.0.0
make package-bottle RELEASE_VERSION=1.0.0
make package-check RELEASE_VERSION=1.0.0
```

The archive, checksum, formula and bottle go to `dist/`; these commands do not
publish. Build bottles on the oldest supported macOS. Bottle/check commands use
a temporary tap and remove their installation; they refuse to replace an existing
Splash installation. The install check requires a poured bottle and runs the
bundled launcher without a compiler or separate Python installation.

To publish: create a GitHub Release on the public mirror `incoai/splash` with
the archive, bottle, checksum files and `SHA256SUMS`, then copy `dist/splash.rb`
over `Formula/splash.rb` in `incoai/homebrew-tap`. Both public repositories
hold exactly one squashed commit of this repository's `main`, authored by Jian
Chen with Zhijian Liu as co-author, and are updated by force-push, never by
pull request. Before re-squashing, `main` must contain no references to the internal
or academic mirrors of the model repositories. Then, on a clean machine:
`brew install incoai/tap/splash && splash --help`, and
`brew audit --strict --online incoai/tap/splash`.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links and Hermes
sessions survive upgrades; downloads remain in the Hugging Face cache.
