# Validate

```sh
make check
make install test-real test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit
```

`make check` needs no model weights, but it runs the Metal tests, so it needs a
Mac that Splash supports: it runs the source checks, `check-native-cpu`,
`check-native-metal`, `check-native-build`, the Python checks and `check-app`,
the desktop app's tests (`swift test --package-path app`). `make
check-native-cpu` builds production and runs native CPU tests without a GPU;
`make check-native-metal` requires a supported Metal device and runs the kernel
tests under shader validation, and the Linear pipeline resource check without
it, then `test-sanitizers-metal`. They include loading small synthetic MLX,
GGUF and vision sources and the GGUF kernels on synthetic tensors. `make
check-native-build` builds every native test, benchmark and tool and runs
none. Hosted CI runs the CPU checks, `check-native-build`, the sanitizers and
`check-app`. `make test-sanitizers` runs the CPU tests that start threads,
the control plane's (`engine`, `native-engine-loop`, the cache, protocol and
transport tests) among them, under AddressSanitizer with UndefinedBehavior
and under ThreadSanitizer. `make test-sanitizers-metal` runs the Metal tests
of the backend (`metal-backend`, whose host-step runner shares each
command's steps with the thread that encodes it) and of the components whose
host steps share their state with the engine thread and the SSD readers (the
hot block cache of spilled KV, the history stream and the expert store) under
ThreadSanitizer, built with it in `build/metal-tsan`; only orders that lie
inside Metal are suppressed (`dev/tests/engine/metal_tsan_suppressions.txt`).
`make architecture-check`, one of the source checks,
also fails when a workflow under `.github/workflows` names a Hugging Face token
(`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or a `secrets.HF_*` secret): CI installs
public models alone and never receives one.

The native tests are in `dev/tests/engine` and the desktop app's in
`app/Tests`; each Python test is in the directory of what it tests:
`dev/tests/server`, `dev/tests/install` (launcher, client configuration and
model installation) or `dev/tests/tools` (`dev/tools`, `dev/benchmarks`, the
Makefiles and the real-model harnesses).

The real-model targets run against an installation in this checkout's
`install/models`: run `make install` with the same options first, or in the
same command, as above. They take `MODEL` exactly as `splash serve --model`
does, and `REVISION`, `DRAFT_MODEL` and `LANGUAGE_ONLY=1` as its `--revision`,
`--draft-model` and `--language-only`:

| Target | Runs |
| --- | --- |
| `verify-models` | the installer's restarts without the Hub, `verify --full`, and the record of the weight images (`dev/tools/installer_restarts.py`, `weight-digests` with `WEIGHT_DIGESTS_ARGS` added, such as `WEIGHT_DIGESTS_ARGS="--expert-memory 7G --ple-memory 2G"` to stream the routed experts and the PLE table, [Release check](#release-check)) |
| `test-real` | vision parity with the family's fixture in `dev/tests/fixtures/vision-parity/` when the installation serves vision, and the native model runtime oracle, each of its three runs with `ORACLE_ARGS` added (`ORACLE_ARGS="--kv-format bf16"`, `"--expert-memory 16G"` or `"--max-context 1010000"`); for a block-sparse target their spill checks take the default prompts, two staging buffers, and 32K tokens with the default window |
| `test-http-real` | the HTTP frontend on an isolated server (`dev/tests/smoke_real.py`), whose greedy comparisons (a penalized greedy request repeats once its prompt is cached, `min_p` 1 answers as greedy) run where `/status` reports `verify_rows` `all`, which on any MoE target takes `HTTP_SMOKE_ARGS="--verify-rows all"`; where `/status` reports `reasoning_switch` `false` (Flint), it gives each request whose answer it checks room for the model's reasoning and leaves out the thinking budget's check; with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`, instead of that smoke, the [persistent cache](server.md#persistent-cache) across a restart in a fresh cache directory: a clean stop, a restart that takes the restore point back, and a next turn that restores the prompt from disk; `release-check` runs both |
| `test-agent-real` | the five official clients through `splash serve` (`dev/tests/agent_real.py`), in `AGENT_SCENARIO` `complete` (the default) or `smoke` |
| `test-release-real` | the HTTP smoke and all five clients on one `splash serve` |
| `test-performance-real` | the native decode, partial-prefix and short-prompt benchmark, or with `BASELINE` its ABBA comparison with that build, in which both builds' `weight-digests` take `WEIGHT_DIGESTS_ARGS` (`dev/benchmarks/backend_regression.py`) |
| `test-pressure-real` | the installation on this checkout's own server under `PRESSURE_ARGS` (default `--max-memory 18G`), which must make it stream its routed experts, with the [serving benchmark](#serving-benchmark)'s three churn clients for `PRESSURE_MINUTES` (default 5), judged on its own by `matrix.py check`: every client served, no engine restart, HTTP error (but `context_length_exceeded`) or Metal failure, no tick of the command loop past the GPU wait limit, no client decode gap at or past 6.2 s, and with a schema-7 `/status` the run's own gap bound; results in `build/release/<model>/pressure` |
| `release-check` | one model on this Mac ([Release check](#release-check)) |

`test-agent-real` and `test-release-real` first run the tests
`dev/tests/install/test_clients.py` has of the installed OpenCode, Codex and Pi
among the selected clients, which need no model.

`test-agent-real` runs Hermes in a profile of its own in the developer's Hermes
root, `splash-test-<id>`, which moves into the run's folder under
`build/release` when Hermes finishes.

Each phase's record keeps what its requests reused of the cache (`reuse`).
Every request after a phase's first resends the conversation, so a phase that
completed two or more requests and reused no cached prompt token fails. Two
phases are exempt: the cancellation phase (`cancel`), and the reference wave
that compacts the conversation, whose summary request and the request after it
share only the system prompt and tools. Replay points of unfinished requests
evicted during a phase only print a warning.

`benchmark-backend` and `benchmark-decode-profile` take `MODEL` the same way.
The models they are run with, one per family and source format:

| Family | MLX | GGUF |
| --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` |

Both source formats load into the block layout of the GGUF projection and MoE
kernels, an MLX target in its MLX formats, a GGUF target in its GGUF ones.

On a 24 GB Mac, `test-agent-real` stops a client's workflow at macOS's warning
memory pressure, which the smaller GGUF variants such a Mac uses can reach under
an agent's load; `SPLASH_TEST_PRESSURE_STOP=4` stops only at critical pressure,
to observe how the engine sheds its cache. The runtime oracle in `test-real` has
no production memory guard: the weights must fit in what macOS has available
above its reserve. When they do not, it measures again every second for up to
30 s before it stops, as a start waits for memory macOS refuses it: macOS can
return the memory of a process that has just exited a moment later (on a 36 GB
M5 Max, a server engine's memory came back within a second of its exit).
Once it has loaded, its checks can neither wait for memory nor yield it, so the
governor a start builds admits all the runtime then allocates, its arenas and
the lanes, KV and cached states of the requests the oracle runs together, as
memory a request in service needs: only the memory plan's budget and macOS's
critical pressure refuse it, and the run stops at critical pressure, naming
what it asked for. Production grants that only to its start, a request it
serves alone and the KV of the requests it serves; a start beside a resident
request, and the cache, are held to the spare reserve, the 1 GiB margin above
it and macOS's warning. A Neural Engine split leaves the plan room for what the
checks hold at once (every lane's state, six cached states and the oracle's KV
pages), as a start's split leaves `--max-context`, so the oracle's split may
take fewer of the Neural Engine's units than a server's does. With only desktop
applications open, a 24 GB Mac runs it for those variants.

The engine's `--kv-spill always`, which `splash serve` never passes, keeps
the KV of every request of a block-sparse target on the SSD cache
([KV on the SSD](server.md#kv-on-the-ssd)), so that a run of `build/splash
serve-native` with it compares spilled requests' outputs with those of a run
without it, which must match bit for bit.

Compare performance on the same idle Mac with the same model and workload.
Keep generated reports, profiles, local paths and experiment notes out of the
source tree and commits.

## Release check

A release is checked once per source identity, and then on each Apple GPU
family (an Apple9 M3 and an Apple10 M5) against a retained baseline build,
`BASELINE`: a checkout whose `build/` holds `splash`, `splash.metallib`,
`engine-tests/backend-benchmark` and `engine-tests/weight-digests`.
`release-check` fails without it. The baseline must load the model: it is the previous release's
build when that loads the model. From a clean checkout:

```sh
make check test-sanitizers                      # once, model-free, on one Mac
make check-native-metal                         # on the other Mac
make install release-check MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit> BASELINE=../splash-baseline
make install release-check MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M LANGUAGE_ONLY=1 REVISION=<commit> BASELINE=...
make install release-check MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> BASELINE=...
make install release-check MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> BASELINE=...
make install test-pressure-real MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> PRESSURE_ARGS="--max-memory 18G"
make install verify-models MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
make test-agent-real MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> AGENT_SCENARIO=smoke AGENT_CLIENTS=...
```

Pin each upstream model to one commit, the same on both Macs. Without a Hub
token for a private draft repository, set `DRAFT_MODEL` to a local copy of the
draft's checkpoint ([Drafts](models.md#drafts)). The Metal suite depends on the GPU
family, so it runs once on each Mac: within `make check` on the first, alone on
the other. Per model, `release-check`:

- runs the runtime oracle and, when the installation serves vision, vision
  parity (`test-real`);
- restarts the installer offline, with the Hub unreachable
  (`HF_ENDPOINT=http://127.0.0.1:9`) and with an empty `HF_HUB_CACHE`: each
  restart must start the same assembly within 10 seconds, name the Hub's
  reason on one line when it asked the Hub ([Revisions](server.md#revisions)), and
  download nothing. It then hashes the
  sources and records in `weights.json` the component, size and SHA-256 of
  every weight image the installation loads (`verify-models`);
- runs the HTTP smoke, which for a text-only installation checks the 400s
  instead of images (`test-http-real`), on the default verify rows: on a MoE
  target it leaves out the greedy comparisons, which run only with
  `HTTP_SMOKE_ARGS="--verify-rows all"`, and says so;
- stops and restarts a server with a [persistent cache](server.md#persistent-cache),
  and requires the conversation's next turn to restore its prompt from disk
  (`test-http-real` with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`);
- compares this build with `BASELINE`, which must have another build
  identity, in ABBA order (`test-performance-real`), every round after the
  first to report a [Neural Engine share](internals.md#neural-engine-prefill) running that
  share: output tokens and acceptance must be
  identical (`EXPECT_OUTPUT_CHANGE=1` allows changed outputs with acceptance
  within 0.02; the benchmark verifies every row (`--verify-rows all`), so a
  MoE target's outputs repeat and its times measure the backend, not the
  verify budget of [Drafts](models.md#drafts)), and so must the bytes of the weight images
  both load (`dev/benchmarks/weights.py`); decode and prefill GPU time, with the cold
  prefills of short prompts (480 to 2016 tokens, one chunk each), and their wall
  times (decode milliseconds per output token, the cold 14K prefill's wall
  time and the short prefills' TTFT), which hold the host work between
  commands, may regress by at most the larger of 2% and
  twice the run's own ABBA spread, a spread above 5% fails as inconclusive,
  and this build's
  memory plan must hold, without the Neural Engine split, at least the
  context the baseline's holds without it. `ANE_FFN_SHARE=SHARE` runs
  every round at that share instead (0: the GPU alone).

Where the Mac's memory cannot hold a model's weight images, or its largest
Metal buffer cannot hold one of them, as for Qwen3.8-Flash-Next,
`WEIGHT_DIGESTS_ARGS` streams its routed experts and PLE table in the loads
whose images are recorded and compared ([streamed experts](server.md#streamed-experts),
[PLE table streaming](server.md#ple-table-streaming)): it gives `weight-digests` the
`--expert-memory SIZE` and `--ple-memory SIZE` that `ORACLE_ARGS` gives the
runtime oracle, each SIZE a whole number of bytes or of `K`, `M` or `G`, such
as `7G`, in `verify-models` and in both builds of the comparison with
`BASELINE`. The layers' images then
leave out their routed experts, and a PLE table that streams has no image
(`ple.bin`): the engine reads those bytes from the source files, which
`verify --full` hashes, and the digests do not cover them. Any size a store
takes records the same images. Where the images and stores of a load exceed
the engine's Metal memory budget, `weight-digests` refuses it before loading
any image and names the options. A tool's expert store is SIZE and stays so,
its chunks laid out as `splash serve`'s, where `splash serve` takes SIZE as
the floor of a store that may be larger (`dev/tools/LoadOptions.hpp`), and
loads the experts whole where memory holds every one: an HTTP smoke that
streams them there takes a smaller `--max-memory` beside `--expert-memory` in
`HTTP_SMOKE_ARGS`.

Results go to `build/release/<owner>--<repo>[--VARIANT]/`. The weight images
do not depend on the GPU, so each model's `weights.json`, recorded with the
same `WEIGHT_DIGESTS_ARGS`, must be identical on the two Macs. The unpinned
`verify-models`, run after the pinned ones while the default branch still
names the pinned commit, resolves the branch online, and its unreachable-Hub
restart must fall back with the Hub's reason. The agent
clients depend on neither the model's format nor the GPU: run the smoke
scenario once per Mac, with the five clients split between the Macs, and
`AGENT_SCENARIO=complete` for one model when the client integration changed.
Expect about 1.5 hours on an M5 Pro and 2.5 hours on an M3 Max, most of it in
the three 27B comparisons. A laptop can cap its GPU power during a long
comparison and so make it inconclusive; rerun `make test-performance-real` for
that model alone once the Mac has cooled.

`.venv/bin/python dev/tools/kernel_identity.py BASELINE/build build` compares
the AIR of each kernel the decode path may run in the two builds'
`splash.metallib` (`--match` for others) and fails if any differs. The helpers
a kernel's module defines are compared in the order they appear there, not by
name: C++ names them for their template values, which a kernel template that
gains a value renames without changing their code. So are the numbers of its
lambdas' anonymous classes and its classes' names in the alias metadata, which
the same change renames.

Two pairs of constants in `runtime/engine/Protocol.hpp` and `server/protocol.py`
version what the server and the engine exchange. When the native wire layout
changed since the last release, bump `kProtocolVersion` and `PROTOCOL_VERSION`
together; each side refuses frames of another version. When a field of the
status document the engine writes was removed or changed meaning, bump
`kStatusSchemaVersion` and `STATUS_SCHEMA_VERSION` together; the server refuses
a status document of another schema. An added field keeps the version, as
`weights` did in 1.2.1 and `ane_ffn` in 1.3.0. Builds between releases share a
version while its layout changes.

## Local benchmarks

From a source checkout with the model installed, use the native benchmark for
prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit BASELINE=/path/to/baseline
```

The first characterizes this build: the decode widths B1-B4, a 14,096-token
partial-prefix request and the cold prefills of short prompts, three samples
each, in
`build/release/<owner>--<repo>[--VARIANT]/backend-benchmark.json`;
`make benchmark-backend MODEL=...` adds the 2K to 128K contexts the memory
plan holds. The second compares this build with a retained checkout's in ABBA
order, as the release check does ([Release check](#release-check)), and writes
`backend-regression.json` there. Neither is a comparison with another engine
or a test of agent task quality.

A change of arithmetic that moves outputs (`EXPECT_OUTPUT_CHANGE=1`) is
measured by its next-token distributions instead. The benchmark's `logits`
scenario dumps the full-vocabulary logits after 16 prefixes of each of the six
transcript requests' 256-token greedy answers, the score-only requests cold and
`--logits-width` at once so that they prefill beside each other, and
`dev/benchmarks/logit_compare.py` compares a dump with one whose prefixes it
scored (`--logits-from`): top-1 agreement, mean and p99 KL and the next-token
NLL change, failing past `--min-top1`, `--max-mean-kl` and `--max-p99-kl`. A
dump is three files, `sequences.txt`, `records.txt` and `logits.f32` (fp32
rows, little-endian as numpy's tofile writes them), so that another
engine's logits after the same prefixes compare the same way:

```sh
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/base
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/b4 --logits-from /tmp/base --logits-width 4
.venv/bin/python -m dev.benchmarks.logit_compare /tmp/base /tmp/b4
```

The second line also measures this build's noise floor: the same prefixes
prefilled four at a time.

A change of the weight images' layout (`EXPECT_IMAGE_CHANGE=1`) makes
`backend_regression` report the images it changed instead of failing on them.
`decode-profile --dispatch-list FILE` writes the profiled B1 cycles'
dispatches in order with their GPU time.

To compare the model's numerics with another engine on the same weights,
`make teacher-logits MODEL=... TEACHER_LOGITS_ARGS='TOKENS OUTPUT --window END:ROWS ...'`
writes the installed model's teacher-forced next-token logits over windows of
a file of uint32 token ids (`dev/benchmarks/teacher_logits.mm`): for each
window, the fp32 logits of the prefill's final position and of ROWS decode
positions after it, each computed by a verify row forced to the true next
token, in `OUTPUT-END.f32`, and prints how well they predict the true ids. It
also takes `--kv-format`, `--state-format`, `--expert-memory` and
`--ple-memory`, and, with `--captures yes`, writes the output rows of the
target layers the model's draft reads (its `target_layer_ids`) at the
positions a window's prefill computed, in one command, to
`OUTPUT-END.captures.f32` (fp32, a row per position of each layer's hidden
values in the draft's layer order), which a reference's residuals after those
layers compare with (`dev/tools/flint_reference.py` writes Flint's,
`residuals.npy`): a draft of the layers to compare, such as a test draft whose
`fc` reads them, makes them its captures.

`benchmark-backend` lists the contexts the memory plan cannot hold in its
report. Its cache checks reuse each context's cached prefix, so they need that
memory free: when other programs leave too little, the engine evicts cached
prefixes and the checks fail, naming what each lookup found.

For a same-machine HTTP regression check, retain a `splash` binary **and its
adjacent `splash.metallib`** built from a checkout with the same native wire
version and status schema as this one (the server refuses any other), with
`engine-tests/weight-digests` beside them, then run from the candidate
checkout, after
`make build/engine-tests/weight-digests`:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

It takes any installed model and, for an upstream one, holds its assembly for
the whole run, so every round serves the same model. For a MoE target, pass
`--verify-rows all`: the check requires the two builds' transcripts to match,
and by default the verify budget chooses a MoE target's rows from cycle wall
times, so its greedy output can differ between runs (see [drafts](models.md#drafts)).
Both rounds pass the option to their engine, so a MoE comparison needs a
baseline built from a checkout that takes it; an older baseline cannot run one.
It starts isolated servers in ABBA order, compares matched cold, exact-prefix
and decode requests by the release check's speed rule and weight bytes (decode
by `metrics.decode_cycle_ms` per output token, so host work between commands
counts), and saves
`build/release/http-regression.json`. With `--burst N` it sends N requests of
each context at once instead, 16 output tokens each, and reports each build's
replay points lost to the burst (`replay_state_publication_failures`), its
decode rate and advertised context. `--follow-up`, which needs `--burst`, then
sends each conversation's next turn and compares their time to first token and
how many resumed at their replay point; a burst run passes only with it. It does
not contact your running server. Use the same power mode and charger, stop other
GPU workloads, and report chip/GPU cores, memory, Splash version, model
revision, actual input/output token counts, and cache hits with results. Keep
cold prefill, cached TTFT and sustained decode separate; a UI token rate alone
does not measure end-to-end agent performance.

## Serving benchmark

`dev/benchmarks/serving` measures a server under concurrent requests and
compares builds of different trees, such as a release candidate with the last
release, on one Mac. Each run starts the tree's own server, so every tree is
measured through the same OpenAI API whatever its native wire and status
schema, and the client times what a user sees.

| Scenario | Requests |
| --- | --- |
| `churn` | three clients, each sending 1500-token answers to prompts that grow every round (about 16K tokens, then more; `CHURN_BASE`/`CHURN_STEP` words), every fourth round resending its conversation, the first requests in a fixed arrival order (`--order 2,0,1` or `0,1,2`, 0.6 s apart), for `--minutes`; `/status` every 2 s |
| `ttft` | for each N of `--n`, on a fresh server: N conversations cache an ~8K-token prefix one at a time, then send it with distinct ~300-token suffixes at the same instant, `--rounds` times; `/status` every 0.5 s |
| `serve1` | one request at a time on an idle server, cold prompts of about `--sizes` tokens (300, 2K and 14K): time to first token and decode milliseconds per token; the `/status` each request waited idle for, and one after the last |
| `warm` | per cached depth of `--sizes`, a fresh cold prefix, then 2 discarded and 8 measured ~300-token suffixes after it: the server's prompt time of each (`--trial` picks the texts, the same for every tree); `/status` as `serve1` |

Every request timestamps each content chunk of its stream; its gaps are the
intervals between consecutive chunks after the first. Each request may take
`--request-seconds` from its send (`plan` gives twice the longest request of
the machine's recorded runs, `longest_request_s` in the spec; `run.py`'s
default is twice the longest on any machine): one the server keeps streaming
past it, though its keepalives hold the connection open, ends the scenario as
the server's failure; in `churn` its client stops, and the end record lists
every request still open, with its open gap or, before its first token, as
unanswered.

```sh
.venv/bin/python -m dev.benchmarks.serving.run --tree TREE --machine m6 \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision <commit> \
  --draft-model DRAFT --scenario churn --order 2,0,1 --minutes 15 --label LABEL --out RESULTS
```

`run.py` installs the model with the tree's own installer, offline, or takes
`--model-root`; refuses a port that already answers; checks that `/status`
names the server it started (`instance.pid`) and the build of the tree's
sources (`identity.cache.build_id` against the tree's own
`dev/tools/build_identity.py check` with its Makefile's constants); runs the
scenario; and stops the server with SIGINT, then SIGTERM. Beside the client's
`LABEL.events.jsonl` and `LABEL.status.jsonl` it writes `LABEL.out`, whose last
line is `DONE` only once everything completed, `LABEL.meta.json` (binary
SHA-256, build id, model, spawn-to-ready seconds and whether the binary had
started on that Mac before, host memory and pressure, process snapshots at
the start, middle and end of the scenario while the server serves, the pids
the run started, and for a run that did not complete why and whose failure
that was: the server's, which exited, was not ready in time, failed a
request or, once ready, did not let the scenario end, or `run.py`'s refusal
to measure) and on the M5 Pro
`LABEL.idle.jsonl`, its idle time and sleep assertions every 10 s. It refuses any output directory that resolves,
through whatever symlinks, into `~/.cache/huggingface` or an `install/models`
directory.

`matrix.py` reads the matrix, `matrix-1.6.0.json` for 1.6.0: the trees and
their commits, the models at pinned revisions, each machine's GPU lock, port
and measured noise, and the cells (machine, model, scenario, server options,
runs per tree and order, recorded reference runs, gates).

- `plan --machine M --tree NAME=DIR ...` writes one job per run, holding the
  machine's GPU lock around `run.py`, interleaved so that no tree owns the
  start or end of the queue; `--references` plans the reference runs a cell
  still lacks.
- `check RESULTS LABEL --minutes N` judges one churn run on its own (`DONE`,
  the load window, every client served, health and gaps); `check --running
  GPUQ` lists queued jobs running past twice the time the first finished job
  of their type took on that Mac (20 minutes per churn run without one).
- `analyze --results DIR... --build cand=BUILD_ID --out DIR` applies the
  gates to every cell with results, on the runs of each tree's build under
  test (`--build NAME=BUILD_ID`, cand's at least, as its
  `build/engine/build-identity.json` names it): a run whose `run.py` build id
  is another, a single cell's comparison whose candidate rounds ran another,
  or a second copy of a label in another results directory is not used. It
  reads every run of a cell, planned or a rerun (a rerun takes the next
  repeat number); an ABBA set whose cand runs are all of another build is
  passed over. It writes `verdict.json` and `verdict.md`: PASS, FAIL or
  INCONCLUSIVE with the numbers (a candidate none of whose runs is usable
  gets its gates INCONCLUSIVE, naming its runs), the trades for the owner,
  every run it did not use and why, a run a cell plans that left no record
  as not run, and per fixed-order set of each churn cell the per-run numbers
  of `bin/cmp.py` and `pertag.py`.

A run of any scenario counts only if it ended with `DONE`, its server never
restarted and failed no request (an HTTP error but `context_length_exceeded`,
an error event in a stream it had begun, a request of a failed TTFT round, or
one `/status` counts in `requests.failed`), it saw no `serious` or `critical`
thermal sample, it served the cell's model, no process it did not start was
among the top five by CPU at 24 % or more (`foreign_cpu_percent` in the spec,
whose note gives its source), and, on the M5 Pro, its owner neither touched
it (HIDIdleTime under 10 s) nor had a process the run did not start hold a
display or system sleep assertion. A TTFT run's concurrency counts only where
every round it was to run completed its N requests. An ample-memory run counts
only if the experts were at their cap at Ready and macOS's memory pressure
level was normal before `run.py` started the server, where it recorded one
(the engine's `memory_pressure` is its band state, and right after the load
macOS's own level can read warning for tens of seconds). A churn run of a
tree under test counts only if its SSD cache is comparable with that of every
run of the trees its cell's gates compare it with: `--max-cache-disk auto` sets
the limit from the volume's free space that day, so some limit must hold
each run within the gates' 2 % of what it held, from 98 % of the most it held
within its limit up to, where it came within 2 % of its limit, that limit
over 0.98 (the reason gives both runs' caches and this run's free space). G8 is
judged on every candidate run of every scenario: a run whose server
restarted, failed a request, had a Metal failure, or exited, was not ready in
time or, once ready, did not let its scenario end (a request or `/status` it
did not answer in time, a request still streaming past `--request-seconds`,
no idle within 10 minutes) fails G8. A run that
`run.py` refused to measure (a port that already answers, a build that is not
the tree's, an installation that failed) or that its surroundings or the
cell's void list voided is listed, to be repeated, and fails G8 only on those
failures: its loop's long tick, which its surroundings can cause, is not
judged. Copy a tree to another Mac by content
(`rsync --checksum`): `git archive` leaves `dev/tests` and `dev/benchmarks` out
(`.gitattributes`), and a copy that keeps the files' old times leaves make
running stale binaries. For a gate, delete the test binary before building it
and record the commit, an empty `git status --porcelain`, the binary's SHA-256
and its test count.

The gates compare per-run metrics on a log scale: work at the end of the
cell's load window, at most 900 s (prefill rows and output tokens at each
model's single-request prices), and over the tags every compared run
completed the geometric means of latency, TTFT and decode rate. The interval
of the candidate/reference ratio is one-sided 90 %, its half-width t(0.9, df)
times the machine's per-run noise for that metric (pooled from the cell's own
same-tree runs, else the spec's recorded value) times sqrt(1/n + 1/m), per
arrival order and pooled. A gate PASSes when the interval's good bound clears
its tolerance and FAILs when its bad bound is past equality (or past the loss
the gate allows), else it is INCONCLUSIVE; noise only decides how many runs
resolve it.

| Gate | Rule |
| --- | --- |
| G1, G2 | work and decode rate against 1.5.0: within 5 % only with a clear gain on TTFT, completion latency, freezes (1.5.0's own runs with a decode gap at or past the machine's G4 floor while the candidate holds G4) or concurrent TTFT, else no loss at all; completion latency: no loss at all |
| G3 | TTFT against main within 2 %, failing once surely worse; under a memory cap also when G2 passes, no decode froze, and the TTFT is not surely worse than the cell's best tree that kept decoding |
| G4 | per candidate run: contended prefill p95 and the p99 of the client gaps between token events (chunks less than half the shortest decode command apart are one event's) within the run's own bound (`max(decode_gap_target_ms, u/0.1)` with its estimate's p95 error), every client gap under the smallest stall rejected on that machine |
| G5 | concurrent TTFT: per N the median of each round's worst, at most the reference's times one plus its own spread |
| G6 | ample memory: work, latency and TTFT within 2 % of the best measured tree, failing once surely worse |
| G7 | single requests (`test-performance-real` against the base, `serve1`, `warm`): `abba.compare` on every metric, in session; a single cell's job removes an earlier comparison before its make and copies only the one its make wrote |
| G8, G9 | no restart, failed request, Metal failure or tick past 5 s; warm start within the base's plus the warm-up; memory accounts summing to the charged bytes, nothing owed at the end, no claim waiting past the resource wait |
| SSD | disk and expert bytes per unit of work not surely above both release references |
| Tags | no tag whose every candidate run is past the references' worst by their own spread |
| G10 | dense, resident, Flint and Apple9 cells against the base: work and TTFT within 2 % by the interval rule (TTFT cells on each round's worst), failing once surely worse, and no G4 failure; undecided goes to the owner |
| Floor | the Flash-Next store held at its cap becomes the floor default only when no metric is worse than the value default's and one is better; mixed results go to the owner |

Each candidate churn run also reports, with nothing judged on them, its
prefill hole share (the streamed experts' prefill host steps, pickup, work
and resume, over `metrics.prefill_wall_ms`), its SSD-busy share (its
prefill expert reads at the seeded `weight_read_bytes` rate over the same
wall time), its `scheduler.bound_cuts`, the rows of its prefill units (each
sample interval's `prefill_rows` per committed command, p10, p50, p90 and
max, with its pass commands and units) and its
`scheduler.index_page_conflicts` per 15 minutes; a run past 5 per 15
minutes (`index_page_conflicts_per_15min`, the scheduler's bound) is
reported to the owner.
