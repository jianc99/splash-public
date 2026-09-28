# Device policy and performance qualification

The convergence pass starts at `2a43027`. It preserves the measured default
execution paths while making offline Q4 calibration independent of fixed
threadgroup counts from individual GPUs. It adds no kernels, weight formats,
production allocations, startup benchmarks or per-model/per-SKU tables.

## Policy ownership

`runtime/ops/KernelPolicy.hpp` holds what the projection policies read of the
device, the primitive its GPU family runs (register tiles on Apple9, tensor
tiles on Apple10 and later) and its core count, and the laws the kernel
families share: the row quanta of each family's tile and one split-K law,
whose tiers, comparator and partition steps each family records with the
machine and date they were measured on.

`runtime/ops/Linear.cpp` owns Q4 selection. Apple9 decode, and prefill chunks
of up to 32 rows, use bfloat simdgroup matrices with 1/2/4/8 K partitions
([apple9-simdgroup.md](apple9-simdgroup.md)). Apple10 uses MPP tiles, with
shape and core count selecting grids. A projection whose grid of N128 tiles
is too narrow to fill four 256-thread threadgroups per core splits K instead:
`LinearTile::Split128` runs the N128 tile over the largest power of two, up to
eight, of K partitions whose grid still fits four threadgroups per core, each
partition at least one 256-input block. Steps of three and four lanes, whose
tiles compute two 16-row MPP fragments, keep 512 inputs per partition and
split long partitions (2048 inputs) on to six threadgroups per core. The
split reads the rows only through that fragment count: one and two lanes take
the same plan, and so do three and four. Every other projection runs a
sequential tile by the tile law of `kAffineTensorTiles`: paired N256 at one
lane and N256 at three and four lanes where the N256 grid holds enough tiles
per core, paired N128 at one lane but where its full grid needs a second
paired wave (k = 3) and the unpaired tile still one (k = 4), four simdgroups
at three lanes. The one-lane Split32/Split64
tiles, withdrawn as defaults after speculative-acceptance reductions on some
M5 prompts, remain offline candidates: they beat Split128 or the sequential
tile on a few 35B one-lane shapes, together 0.55% of a one-lane step on a
20-core M5 Pro. The obsolete Apple9 one-lane MPP branches have been removed;
those kernels remain useful as qualification references and offline
candidates. Apple10 prefill chunks of up to 32 rows run the decode plan of
their rows in whole lanes, an up-with-gate projection the up pass of the decode
gate/up plan, where that plan's tiles compute one 16-row MPP fragment (up to
16 rows) or it splits K; at 17-32 rows both it and the 32-row prefill tile
compute two fragments, and only a split grid gains. The measurements behind
the tensor rules are in "Tensor decode law" and "Tensor short prefill" below.

Core count comes from the Metal device's IORegistry property. Missing metadata
uses one 32-core estimate for every kernel (`kAssumedGpuCores`), an
intermediate value in the 16–40-core range of our reference machines. This is
not a calibrated optimum or a performance guarantee for unidentified GPUs. A
nonzero reported count always overrides it. Family 11 (the M6) runs the
family 10 policy, and a CPU test holds every family 11 plan equal to family
10's; the measurements behind the policy come from families 9 and 10. Core
count alone cannot describe memory bandwidth, cache capacity, power state or
compiler behavior.

MoE routing scales its row threshold with core count. The expert tile's
four-SIMD-group Apple9 decode default remains a family rule, measured on the
40-core M3 Max. Smaller Apple9 devices need an expert-kernel comparison before
claiming that rule is optimal. Prefill and Apple10 retain eight SIMD groups.

## Bounded offline calibration

Q4 candidates always start with the shipped baseline. Persistent grids now
include two, three and four threadgroups per reported core plus the full grid,
instead of fixed counts 36/60/80. One lane lists the Split32/Split64 tiles
when K % 1024 == 0. Apple9 additionally exposes every valid simdgroup split in
1/2/4/8, and Apple10 every Split128 split of two to eight its blocks allow at
every lane count. The maximum candidate count is 20, derived beside
`Linear::kMaximumCandidates`; deduplication handles small grids. Apple9 prefill
chunks of up to 32 rows list only the simdgroup splits, whose fixtures differ
from the MPP prefill tile's, and Apple10 chunks that run a decode plan list
only it; the tuner does not probe them (`kPrefillProbeRows`).
Other prefill candidates are unchanged. New candidates do not automatically
change serving.

The existing offline tuner qualifies numerical results, admits the maximum
candidate workspace, alternates baseline/candidate timing and requires both
GPU and wall-time evidence. Interrupted or inconclusive runs keep the default.
Its scratch admission already covered candidate maxima; the independent batch
reuse test was corrected to do the same after the extra K splits exposed its
baseline-only allocation assumption.

On a new device, run the tuning tool on an installed model:

```sh
make tune-kernels MODEL=mlx-community/Qwen3.8-27B-4bit \
  TUNE_ARGS='--seconds 30 --pairs 31 --candidates --confirm 12'
```

Record GPU family/core count, power mode, OS/toolchain and source identity.
Keep other GPU work idle. The tool prints measurements and confirms complete
prefill/decode graphs; it does not persist a serving profile. Operator timings
include standalone preparation and do not substitute for fused-producer or
cold full-model measurements. Promote a default change only after repeatable
whole-model A/B results, unchanged correctness/state-restoration behavior,
and acceptable speculative acceptance and memory use. Check both models,
short/long prompts and batch widths 1–4. Preserve the baseline when evidence
is mixed. For the family-only MoE rule, separately compare four/eight groups
on the missing hardware; the existing MoE tuner does not vary that rule.

## Convergence validation (2026-09-21)

- An independent before/after snapshot compares 1,591,200 default plans across
  families 9/10/11, every core count 1–128, unknown count, the IORegistry reader's
  upper bound of 4096, production-like shapes and dispatch boundaries. Configs,
  pipeline names and workspace sizes are byte-identical. These simulated core
  counts establish policy consistency, not measured performance on those GPUs.
- Permanent CPU tests cover 84,240 decode workload/device combinations: valid
  grids, bounded unique candidates, retained defaults and workspace admission
  after installing each candidate. `test-engine-cpu` now runs `linear-plan
  --cpu`, so these checks do not depend on a Metal test run.
- M3 Max 40, M5 Pro 16 and M5 Pro 20 pass full builds and CPU suites, Linear
  candidate numerical tests, tuning controls/batch reuse, and the 168-case
  independent fp64 simdgroup test with GPU shader validation. Both remote
  machines are on AC. All build the same production source identity:
  `src-a996ac63153606d2ab3534be64d2ca804dc396d44e06ab222cfc42922b2a3740`.
- On each machine, the production metallib is byte-identical to its pre-cleanup
  library. This pass establishes unchanged default GPU work and valid expanded
  calibration; it does not claim a new serving speedup or repeat the previous
  whole-model ABBA measurements.

The policy snapshot harness of this pass is not in the repository.
`plan-census` (`dev/tuning/plan_census.cc`, built by `test-engine-cpu`)
prints every production plan, with its tuning candidates and arena bounds, of
the 27B and 35B targets and drafts, affine and GGUF, and of the 35B MoE layer,
on families 9–11 at an unknown and 14 known core counts from 10 to 80. Run in
two builds, the diff of its outputs lists exactly the plans a change moves.
`policy-bench` (`make benchmark-policy`, `dev/benchmarks/policy_bench.mm`)
times those plans DRAM-cold beside every configuration a law could pick, at
the native core count and others emulated by width; runs of the two builds,
alternated and summarized by `dev/benchmarks/policy_summary.py --new`, give a
change's projection time per decode step against its parent's.
Earlier serving results remain in
[remaining-decode-optimizations.md](remaining-decode-optimizations.md) and
[apple9-simdgroup.md](apple9-simdgroup.md).

## Unknown-core fallback follow-up

The unknown-core path now uses a single 32-core estimate. The convergence
snapshot reported above precedes this fallback change; unknown-core plans
intentionally differ. Existing CPU policy tests cover unknown counts in both
prefill and decode, including equivalence to an explicitly reported 32-core
GPU. Known-core plans remain byte-identical in a separate before/after
comparison. This fallback does not require startup or user-run calibration.
The MoE router kept its own 512-row wide-tile threshold for unknown counts
(about 20 cores) until it joined this estimate: it now takes 32 cores' 832
rows, so on an unknown GPU a prefill chunk of 512–831 rows scores its routes
on the 8-row tile instead of the 32-row one. Both tiles give the same scores.

## Tensor decode law (2026-09-27)

The split tiers and tile law of Apple10 and Apple11 affine MPP decode
(`kAffineSplitTiers`, `kAffineTensorTiles` in `runtime/ops/KernelPolicy.hpp`)
come from `policy-bench` runs of the b6752e1 kernels and policy:

- the 20-core M5 Pro (Apple10, macOS 26.5.2) and the 12-core M6 (Apple11,
  macOS 27.0), eight alternating DRAM-cold runs each, after `--check` held all
  13,838 and 10,757 configurations to fp64 under `MTL_SHADER_VALIDATION=1`;
- the 27B and 35B MLX 4-bit targets and drafts at 8-32 rows with their
  counts per decode step, at the native core count and 10-80 (M5 Pro) or
  10-20 (M6) cores emulated by width, and a grid of N 256-16384 by K
  2048-25600;
- per row count, an exhaustive search over the split target, floor and a
  second tier, the N256 threshold, the pairing window, the four-simdgroup
  rule and the persistent-group constants. The law has the lowest RMS log
  regret against each shape's fastest measured configuration among those
  that leave no (machine, cores, model, width) group more than 0.5% slower
  than the previous rules.

Q4 projection time per decode step (targets and drafts, vocabulary heads and
MoE excluded), law over the previous rules, and law over each shape's fastest
configuration:

| host | cores | model | B1 | B2 | B3 | B4 | law / fastest, B1-B4 |
|---|---:|---|---:|---:|---:|---:|---|
| M5 Pro | 10 (emulated) | 27B | 0.940 | 1.000 | 0.958 | 1.000 | 1.004, 1.027, 1.004, 1.039 |
| M5 Pro | 10 (emulated) | 35B | 1.000 | 1.000 | 0.965 | 0.994 | 1.063, 1.006, 1.035, 1.038 |
| M5 Pro | 12 (emulated) | 27B | 0.957 | 1.000 | 0.983 | 0.991 | 1.117, 1.116, 1.009, 1.133 |
| M5 Pro | 12 (emulated) | 35B | 0.987 | 1.000 | 0.991 | 0.999 | 1.054, 1.022, 1.017, 1.000 |
| M5 Pro | 16 (emulated) | 27B | 1.000 | 1.000 | 0.961 | 0.949 | 1.060, 1.070, 1.029, 1.018 |
| M5 Pro | 16 (emulated) | 35B | 1.000 | 1.000 | 1.000 | 1.000 | 1.005, 1.007, 1.000, 1.004 |
| M5 Pro | 20 | 27B | 1.000 | 1.000 | 0.999 | 0.998 | 1.014, 1.000, 1.026, 1.017 |
| M5 Pro | 20 | 35B | 0.983 | 1.000 | 0.999 | 0.978 | 1.008, 1.007, 1.004, 1.017 |
| M5 Pro | 32 (emulated) | 27B | 0.994 | 1.000 | 0.983 | 0.981 | 1.145, 1.133, 1.010, 1.069 |
| M5 Pro | 32 (emulated) | 35B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000, 1.000, 1.018, 1.155 |
| M5 Pro | 40 (emulated) | 27B | 0.980 | 1.000 | 1.003 | 0.959 | 1.001, 1.003, 1.012, 1.001 |
| M5 Pro | 40 (emulated) | 35B | 1.000 | 1.000 | 1.001 | 0.999 | 1.000, 1.000, 1.020, 1.083 |
| M5 Pro | 60 (emulated) | 27B | 1.000 | 1.000 | 0.989 | 0.939 | 1.018, 1.017, 1.050, 1.039 |
| M5 Pro | 60 (emulated) | 35B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000, 1.000, 1.000, 1.030 |
| M5 Pro | 80 (emulated) | 27B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000, 1.000, 1.056, 1.077 |
| M5 Pro | 80 (emulated) | 35B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000, 1.000, 1.000, 1.009 |
| M6 | 10 (emulated) | 27B | 0.958 | 1.000 | 0.962 | 1.000 | 1.003, 1.005, 1.003, 1.015 |
| M6 | 10 (emulated) | 35B | 1.000 | 1.000 | 0.941 | 1.000 | 1.051, 1.005, 1.014, 1.023 |
| M6 | 12 | 27B | 0.938 | 1.000 | 0.978 | 0.995 | 1.069, 1.061, 1.009, 1.088 |
| M6 | 12 | 35B | 0.993 | 1.000 | 0.995 | 1.000 | 1.048, 1.043, 1.012, 1.002 |
| M6 | 16 (emulated) | 27B | 1.000 | 1.000 | 0.952 | 0.986 | 1.064, 1.072, 1.020, 1.015 |
| M6 | 16 (emulated) | 35B | 1.000 | 1.000 | 1.000 | 1.000 | 1.009, 1.008, 1.002, 1.019 |
| M6 | 20 (emulated) | 27B | 1.000 | 1.000 | 0.999 | 1.000 | 1.007, 1.002, 1.013, 1.007 |
| M6 | 20 (emulated) | 35B | 0.981 | 1.000 | 1.000 | 0.986 | 1.000, 1.022, 1.012, 1.012 |

Not adopted, measured the same way: split targets per GB/s of stream
bandwidth instead of per core (equal on both machines, up to 7% slower in the
M6's emulations); an Apple11 column fitted on the M6 alone (0.03% apart);
Split128 at three lanes on four lanes of storage (neutral, slower at eight
splits); any refit of the persistent-group constants (none passed the 0.5%
bound). Splish's 40-core M5 Max runs the 27B attention input at four lanes
(1.4 N256 tiles per core) in 0.78 of the N128 time on N256, where 40 cores
emulated on the M5 Pro measured 1.03; the four-lane threshold stays at 1.6
until a real 40-core machine can be measured with policy-bench.

### Effective concurrency (2026-09-27)

The hardware characterization of the M5 Pro and M6 measured each tile's
effective concurrency k: a core launches six to eight of its threadgroups,
but once it saturates only k progress, so a grid runs
ceil(ceil(G / cores) / k) waves on its busiest core. k is the same on both
families: N128 4 at 8 and 16 rows and 3 at 32, the paired N128 3, N256 2,
the paired N256 4, four-simdgroup N128 at 24 rows 5
(`kTensorConcurrency`). The laws it reproduces are now written in it: the
N128 persistent grids (full grid and wave of k = 4), the paired N256 grid
(one wave of 4), and the one-lane pairing window, from a fifth of a tile
per core past the paired tile's k to the unpaired tile's k (3.2-4). The
forms that would replace measured constants fail the gate on the policy-bench
data of both machines:

- pairing by fewer waves (unpaired from any fourth threadgroup per core):
  the 35B GDN input at 32 emulated cores (3.1 per core) runs 2.4-3.8% faster
  paired in every run of two sets, 0.95% of that step; past the full grid the
  same comparison costs up to 13%;
- persistent grids of one k-wave: two N256 groups per core cut the M6 27B
  B1 and B2 Q4 time 5.3% and 5.8% (gate/up, 5.7 tiles per core) but cost up to
  16% at three and four lanes and 0.5-1.4% at 16 emulated cores; three paired
  N128 groups per core cost up to 9%, five four-simdgroup ones 1.4%;
- the N256/N128 choice at 24 and 32 rows by waves costs up to 6.3% (ties to
  N128) or 3.0% (ties to N256); its reading of the four-lane threshold, the
  32-row N128 k over N256's (3 / 2 = 1.5 N256 tiles per core), lowers the
  regret (RMS log 0.053 to 0.046) but makes the M6's 35B at 16 emulated
  cores 1.2% slower;
- three lanes over four lanes of storage (MPP computes 16-row fragments):
  the 27B's B3 step on each shape's fastest 32-row configuration takes
  1.02-1.09 of its 24-row plans at 10-16 cores on both machines, as the
  32-row N256 tile takes 1.045-1.05 of the 24-row one's time.

The split tiers stay measured: the characterization's model-derived split
law has the higher regret at 24 and 32 rows. A four-simdgroup N128 instance
at 32 rows (32 columns per simdgroup) wins only from 3.25 tiles per core
(the 27B's residual projections on 10-12 cores: B4 Q4 time -4.0% on the M6,
-5.0% and -7.4% at 10 and 12 emulated cores), a rule of its own; it is kept
off this law.

## Tensor short prefill (2026-09-27)

Apple10 and Apple11 prefill chunks of up to 32 rows ran the 32-row MPP
prefill tile (`prefill_linear_q4_*`): one row of 128-column tiles, each
streaming all of K and computing two 16-row MPP fragments however few rows
the chunk held. Such a chunk now runs the decode plan of its rows in whole
lanes where that plan's tiles compute one fragment (up to 16 rows) or split
K (`Linear::baseline`); an up-with-gate projection runs the up pass of the
decode gate/up plan (N256 up-SiLU instances, new at one and two lanes). At
17-32 rows both tiles compute two fragments and only a split grid gains.
Most prompts end in such a chunk: prefill stops at the last 32-token page
boundary before the reply's generation prompt, where the state the next
turn reuses ends, and runs the rest apart.

`policy-bench --suite prefill`, the previous plans and these alternated over
four DRAM-cold runs each after `--check` of every configuration under
`MTL_SHADER_VALIDATION=1` (7,042 on the M5 Pro, 5,854 on the M6): projection
time of one prefill pass (the targets' and draft-context projections by
their counts per pass, the prefill tile's input sums included), this rule
over the previous plans, and over each shape's fastest measured
configuration (the decode tiles at the chunk's padded rows included):

| host | cores | model | 1-8 rows | 12-16 | 20-32 | rule / fastest |
|---|---:|---|---:|---:|---:|---:|
| M5 Pro | 20 | 27B | 0.553 | 0.620 | 0.911 | 1.000-1.004 |
| M5 Pro | 20 | 35B | 0.44 | 0.517 | 0.634 | 1.000-1.037 |
| M5 Pro | 10-80 (emulated) | 27B | 0.27-0.73 | 0.31-0.80 | 0.53-1.00 | 1.000-1.081 |
| M5 Pro | 10-80 (emulated) | 35B | 0.20-0.60 | 0.24-0.63 | 0.37-0.99 | 1.000-1.092 |
| M6 | 12 | 27B | 0.81 | 0.854 | 1.000 | 1.002-1.026 |
| M6 | 12 | 35B | 0.576 | 0.626 | 0.78-0.80 | 1.003-1.047 |
| M6 | 10-20 (emulated) | 27B | 0.67-0.81 | 0.71-0.87 | 0.90-1.00 | 1.001-1.029 |
| M6 | 10-20 (emulated) | 35B | 0.46-0.66 | 0.53-0.66 | 0.65-0.87 | 1.000-1.065 |

No machine, core count, model and row count is slower. On the 20-core M5
Pro a whole prefill pass of the MLX 4-bit models (GPU time of a chunk after
a 512-row prefix, two alternated rounds of five repetitions) takes, for the
27B, 65 instead of 115 ms at 1-8 rows, 72 at 12-16 and 104-105 at 20-32
(0.90-0.92), and for the 35B 11.9 instead of 19.7 ms at one row, 0.77 of the
time at 8, 0.84 at 16 and 0.90-0.95 at 20-32; from 33 rows nothing changes.
Served (SPEED-Bench short row, 8 prompts of 1024 tokens, two alternated
rounds) the time to first token falls from 541 to 506 ms (27B) and from 162.5
to 156.3 ms (35B), and the second turn of two-turn chats from 808 to 783 ms
and from 117.7 to 110.3 ms. The last chunk's numerics change, so outputs
differ and with them the drafts' acceptance: over 30 prompts (each build's
outputs repeat exactly) tokens per cycle move from 6.13 to 6.10 for the 27B
(9 prompts up, 10 down, 11 unchanged) and from 5.152 to 5.147 for the 35B
(14 up, 11 down, 5 unchanged).

Not adopted: the decode plan wherever its padded rows are fewer than 32 (also
at 17-24 rows without a split) takes 0.95-1.07 of this rule's time on the
M6 (the 35B at 16 emulated cores 1.07).

## GGUF register tile on the tensor primitive (2026-09-27)

Apple10 and Apple11 decoded every GGUF projection on the staged tile. At one
lane the exact register tile is faster there for two formats on both
measured machines: over the 27B and 35B dense projections of each format
(policy-bench `--suite gguf`, b6752e1 kernels, each tile at its split tier)
it takes of the staged tile's time

| format | M5 Pro, 20 cores | M6, 12 cores |
|---|---:|---:|
| IQ4_XS | 0.99 | 0.91 |
| Q8_0 | 0.92 | 0.99 |
| Q4_K | 1.01 | 0.99 |
| Q5_K, Q6_K | 1.02-1.05 | 1.04-1.05 |
| IQ3_S, IQ3_XXS, IQ2_S | 1.17-1.42 | 1.19-1.48 |

and 1.1-2.5 times its time at two lanes and more. A one-lane projection (a
gate/up plan's two) whose quantized segments are all IQ4_XS or Q8_0
(`tensorRegistersFormat`) takes the register tile, split by Apple9's tiers,
which come within 2% (M5 Pro) and 3.5% (M6) of each shape's fastest register
split; Q4_K, faster on the M6 only, stays staged. The decode scratch bound
covers both tiles at one lane, as formats are not part of a workload.

The gate (four alternated runs per build, each timing both builds' plans):
one-lane GGUF projection time falls to 0.989 (27B, IQ4_XS) and 0.928 (35B,
Q8_0) on the M5 Pro and to 0.909 and 0.987 on the M6; every other format and
width is unchanged. Decode steps (four ABBA rounds, 512 prompt tokens): the
35B UD-Q4_K_M at one lane takes 26.49 instead of 26.96 ms on the M5 Pro
(0.983) and the 27B UD-Q4_K_M 0.994 there and 0.983 on the M6 (131.2 instead
of 133.5 ms; only one lane fits in its 24 GB), the 27B UD-IQ3_XXS 0.996 on the
M6; no width is slower beyond its noise. The one-lane numerics change, so
outputs differ:
over 96 prompts of the SPEED-Bench short, 8k and 16k rows the 35B's tokens
per cycle move from 4.571 to 4.543 (42 prompts up, 52 down), over the 32
short ones the 27B's from 6.153 to 6.130 (16 up, 12 down).
