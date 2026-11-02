"""The 1.6.0 benchmark matrix: plan its jobs, check its runs, judge them.

    python3 -m dev.benchmarks.serving.matrix plan --spec SPEC --machine m6 \\
        --tree cand=~/dev/vr16/cand --tree pass=~/dev/vr16/pass0bd --commit cand=<sha> \\
        --tools ~/dev/vr16/cand --out ~/bench/1.6.0 --jobs ~/gpuq/pending --first 20
    python3 -m dev.benchmarks.serving.matrix check RUN_DIR LABEL [--minutes 15]
    python3 -m dev.benchmarks.serving.matrix check --running ~/gpuq
    python3 -m dev.benchmarks.serving.matrix analyze --spec SPEC --results DIR... --out DIR

plan writes one job script per run, each holding the machine's GPU lock
around run.py, interleaved so that no tree owns the start or end of a
machine's queue (--references plans the reference runs a cell still lacks).
check judges one run on its own (completion, structural minimum, health,
bounded gaps), or lists the queued jobs running past twice their expected
time. analyze applies the spec's gates (DESIGN-1.6.0 §8.3 as amended by
amend/V.md §5 and the owner's rulings) to every cell with results and writes
verdict.json and verdict.md: PASS, FAIL or INCONCLUSIVE with the numbers, the
trades for the owner, and the runs it did not use and why.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks.serving import clients, stats
from dev.benchmarks.serving.run import SERVER_FAILURES, refuse_protected
from dev.benchmarks.serving.stats import FAIL, INCONCLUSIVE, PASS

# A churn run's work and SSD counters are read at the sample nearest the end
# of its cell's load window, at most this second (work@900, the speed-gap
# forensics' yardstick, bin/cmp.py): run.py polls /status through the drain,
# the references churn_ord.py recorded stopped 5 s after the window.
WORK_SECONDS = 900
THERMAL_VOID = ("serious", "critical")
HIGHER_IS_BETTER = {"work": True, "decode": True, "latency": False, "ttft": False}


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    spec["path"] = str(path)
    spec["order_names"] = {tuple(order): name for name, order in spec["orders"].items()}
    return spec


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as lines:
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a killed writer
    return records


def value(document, path, default=None):
    found = clients.dig(document, path)
    return default if found is None else found


def nearest(samples: list[dict], second: float) -> dict | None:
    return (
        min(samples, key=lambda s: abs(s["t"] - second))["status"] if samples else None
    )


def store_size(status: dict):
    """The experts' size: the experts account (schema 7) or the store's
    chunks (older trees)."""
    experts = next(
        (
            a
            for a in value(status, "memory_governor.accounts", [])
            if a.get("name") == "experts"
        ),
        None,
    )
    return (
        experts["current_bytes"]
        if experts
        else value(status, "streamed_experts.chunks")
    )


def work_second(cell: dict) -> float:
    return min(cell.get("minutes", 15) * 60, WORK_SECONDS)


def run_label(cell: str, tree: str, order: str, repeat: int) -> str:
    return (
        f"{cell}-{tree}-{repeat}" if order == "-" else f"{cell}-{tree}-{order}{repeat}"
    )


def abba_runs(cell: dict) -> list[tuple[str, str, int]]:
    """The runs of an in-session ABBA cell, set by set: reference, cand,
    cand, reference."""
    reference = cell["abba"]["reference"]
    runs = []
    for index in range(cell["abba"].get("sets", 1)):
        first, second = 2 * index + 1, 2 * index + 2
        runs += [
            (reference, "-", first),
            ("cand", "-", first),
            ("cand", "-", second),
            (reference, "-", second),
        ]
    return runs


def locate(results: list[Path], label: str) -> Path | None:
    """The results directory holding a run: its run.py meta.json, written
    also for a run that did not reach its scenario, or its client's
    records."""
    for directory in results:
        if (
            (directory / f"{label}.meta.json").exists()
            or (directory / f"{label}.events.jsonl").exists()
            or (directory / f"{label}.jsonl").exists()
            or any(directory.glob(f"{label}-n*.events.jsonl"))
        ):
            return directory
    return None


# ------------------------------------------------------------------ runs


def run_files(directory: Path, label: str, kind: str) -> list[Path]:
    """A run's LABEL.KIND.jsonl, or a ttft run's, one per concurrency."""
    single = directory / f"{label}.{kind}.jsonl"
    return (
        [single]
        if single.exists()
        else sorted(directory.glob(f"{label}-n*.{kind}.jsonl"))
    )


class Run:
    """One run's records: its done requests, the requests the server failed,
    its end, /status samples and events (a ttft run's over its levels) and
    meta, and the reasons it may not be used: `invalid` all of them,
    `voided` those not the server's own failure. A churn run's work is read
    at `window` seconds."""

    def __init__(
        self,
        directory: Path,
        label: str,
        cell: str = "",
        tree: str = "",
        order: str | None = None,
        window: float = WORK_SECONDS,
    ):
        self.directory, self.label, self.cell, self.tree, self.order = (
            directory,
            label,
            cell,
            tree,
            order,
        )
        self.window = window
        meta = directory / f"{label}.meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.invalid: list[str] = []
        self.voided: list[str] = []
        out = directory / f"{label}.out"
        self.complete = out.exists() and out.read_text().rstrip().endswith("DONE")
        events = [
            e
            for path in run_files(directory, label, "events")
            for e in read_jsonl(path)
        ]
        self.done = {e["tag"]: e for e in events if e.get("kind") == "done"}
        # Churn's HTTP errors but context_length_exceeded (error events of
        # streams the server had begun among them), and each request of a
        # failed TTFT round.
        self.errors = [
            failure
            for e in events
            for failure in (
                [f"{e.get('code')} {e.get('body', '')}"]
                if e.get("kind") == "http_error"
                else e.get("failures", [])
                if e.get("kind") == "round_failed"
                else []
            )
            if "context_length_exceeded" not in failure
        ]
        self.end = next((e for e in events if e.get("kind") == "end"), None)
        self.samples = [
            s
            for path in run_files(directory, label, "status")
            for s in read_jsonl(path)
            if "status" in s
        ]
        self.idle = read_jsonl(directory / f"{label}.idle.jsonl")

    @property
    def last(self) -> dict:
        return self.samples[-1]["status"] if self.samples else {}

    def work(self, prices: dict | None, second: float | None = None) -> float | None:
        status = nearest(self.samples, self.window if second is None else second)
        if status is None or not prices:
            return None
        return (
            status["scheduler"]["prefill_rows"] / prices["R"]
            + status["metrics"]["decode_output_tokens"] / prices["D"]
        )

    def latency(self, tag: str) -> float:
        return self.done[tag]["t"] - self.done[tag]["start"]

    def client_gaps(self) -> tuple[list[float], float]:
        """Every gap after each request's first token, in seconds, and the
        largest, counting the open gaps the end record lists."""
        gaps = []
        for record in self.done.values():
            chunks = record.get("chunks_ms") or []
            gaps += [(b - a) / 1000 for a, b in zip(chunks, chunks[1:])]
        opened = list(((self.end or {}).get("open_gaps") or {}).values())
        return gaps, max(gaps + opened, default=0.0)

    def largest_decode_gap(self) -> float | None:
        """The longest decode gap the run shows, in seconds: its clients'
        (a run.py run), or its /status' (metrics.decode_gap_ms, schema 7;
        kv_spill.history_stream.decode_gap_ms on the older trees, as
        bin/gap.py reads it); None where it records none."""
        found = [
            gap / 1000
            for sample in self.samples
            for gap in (
                value(sample["status"], "metrics.decode_gap_ms.max"),
                value(sample["status"], "kv_spill.history_stream.decode_gap_ms.max"),
            )
            if gap is not None
        ]
        if any(record.get("chunks_ms") for record in self.done.values()):
            found.append(self.client_gaps()[1])
        return max(found, default=None)

    def failed_in_status(self) -> int:
        """The largest requests.failed of the run's /status samples (a ttft
        run's servers each start afresh)."""
        return max(
            (value(s["status"], "requests.failed", 0) for s in self.samples), default=0
        )

    def failed_requests(self) -> int:
        """How many requests the server failed: its /status count or its
        clients', whichever is larger. Each sees failures the other may not:
        /status those of the engine, the clients also those the frontend
        raised."""
        return max(self.failed_in_status(), len(self.errors))

    def restarts(self) -> tuple | None:
        """transport.restarts at the first sample and its largest, when the
        engine restarted during the run."""
        seen = [
            r
            for r in (value(s["status"], "transport.restarts") for s in self.samples)
            if r is not None
        ]
        return (seen[0], max(seen)) if len(set(seen)) > 1 else None


def load_runs(spec: dict, results: list[Path]) -> list[Run]:
    """The churn runs of the spec's cells found in the results: new runs by
    their meta.json, recorded ones by the labels each cell lists."""
    cells = {cell["name"]: cell for cell in spec["cells"]}
    runs = []
    for directory in results:
        for path in sorted(directory.glob("*.meta.json")):
            meta = json.loads(path.read_text())
            cell = cells.get(meta.get("cell"))
            if cell and cell.get("scenario") == "churn":
                order = spec["order_names"].get(tuple(meta.get("order") or ()))
                runs.append(
                    Run(
                        directory,
                        meta["label"],
                        cell["name"],
                        meta.get("tree_name"),
                        order,
                        work_second(cell),
                    )
                )
    for cell in spec["cells"]:
        if cell.get("scenario") != "churn":
            continue
        for tree, orders in (cell.get("recorded") or {}).items():
            for order, labels in orders.items():
                for label in labels:
                    directory = locate(results, label)
                    if directory is not None:
                        runs.append(
                            Run(
                                directory,
                                label,
                                cell["name"],
                                tree,
                                order,
                                work_second(cell),
                            )
                        )
    return runs


def assertion_pid(holder: str) -> int | None:
    """The pid of a `pmset -g assertions` line ("pid 330(powerd): ...")."""
    _, found, rest = holder.partition("pid ")
    digits = rest.split("(", 1)[0].strip()
    return int(digits) if found and digits.isdigit() else None


def validate(run: Run, spec: dict, cell: dict) -> None:
    """Records why a run cannot be used (amend/V.md §5 validity), each reason
    in run.invalid. The server's own failures, an engine restart, a request
    it failed, or its exit, a start not ready in time, a scenario it did not
    let end or a failed request that ended the run (run.py's `failure`),
    fail a candidate's G8; every other reason, run.py refusing to measure (a
    port that answers, a build that is not the tree's, an installation that
    failed), the run's surroundings or the cell's void list, is also in
    run.voided: such a run is to be repeated, and judged on nothing but the
    server's own failures it also shows, which still fail G8."""

    def void(reason: str) -> None:
        run.invalid.append(reason)
        run.voided.append(reason)

    meta = run.meta
    if not run.complete:
        reason = "the run did not end with DONE" + (
            f": {meta['error']}" if meta.get("error") else ""
        )
        if meta.get("failure") in SERVER_FAILURES:
            run.invalid.append(reason)
        else:
            void(reason)
    if run.label in (cell.get("void") or {}):
        void(f"void: {cell['void'][run.label]}")
    thermal = {s["status"].get("thermal_state") for s in run.samples} & set(
        THERMAL_VOID
    )
    if thermal:
        void(f"thermal state {sorted(thermal)}")
    if meta:
        idle = meta.get("gpu_idle_check")
        if idle and not idle["ok"]:
            void(f"the GPU was busy at the start ({idle['max_percent']} %)")
        model = spec["models"][cell["model"]]["model"]
        if meta.get("model") != model:
            void(f"model {meta.get('model')}, the cell's is {model}")
        floor = spec["gates"]["foreign_cpu_percent"]
        for snapshot in meta.get("ps") or []:
            # base.md §2.7: void when a process this run did not start is
            # among the top five by CPU, using at least foreign_cpu_percent
            # (the gates' note says why not any).
            foreign = [
                row for row in snapshot["rows"][:5] if not row[4] and row[1] >= floor
            ]
            if foreign:
                void(
                    f"a process this run did not start in the top five by CPU: "
                    f"{foreign[0][3]} at {foreign[0][1]:g} %"
                )
                break
    if run.idle:
        floor = spec["gates"]["m5_idle_seconds"]
        ours = set(meta.get("pids") or ())
        touched = [
            s
            for s in run.idle
            if s.get("hid_idle_s") is not None and s["hid_idle_s"] < floor
        ]
        held = [
            h
            for s in run.idle
            for h in s.get("holders", [])
            if assertion_pid(h) not in ours
        ]
        if touched:
            void(
                f"the laptop was in use (HIDIdleTime {touched[0]['hid_idle_s']:.1f} s)"
            )
        if held:
            void(f"another process held a sleep assertion: {held[0]}")
    if run.failed_requests():
        run.invalid.append(f"requests the server failed: {run.failed_requests()}")
    restarted = run.restarts()
    if restarted:
        run.invalid.append(f"the engine restarted ({restarted[0]} -> {restarted[1]})")
    if cell.get("ample") and run.samples:
        first = run.samples[0]["status"]
        experts = next(
            (
                a
                for a in value(first, "memory_governor.accounts", [])
                if a.get("name") == "experts"
            ),
            None,
        )
        at_cap = (
            experts["current_bytes"] >= experts["cap_bytes"]
            if experts
            else value(first, "streamed_experts.chunks")
            == value(first, "streamed_experts.cap_chunks")
        )
        if not at_cap or first.get("memory_pressure") != "normal":
            void(
                "ample start: the experts were not at their cap with pressure normal at Ready"
            )


# ---------------------------------------------------------- per-run facts


def legacy_cmp(runs: list[Run], prices: dict) -> list[dict]:
    """bin/cmp.py's numbers for one fixed-order set: per finished run its
    completions, work at 650 and 900 s, and the mean TTFT, mean and max
    latency over the tags every run completed."""
    runs = [run for run in runs if run.complete]
    if not runs:
        return []
    common = set.intersection(*(set(run.done) for run in runs))
    rows = []
    for run in runs:
        tags = [run.done[t] for t in common]
        rows.append(
            {
                "label": run.label,
                "done": len(run.done),
                "common": len(common),
                "work650": run.work(prices, 650),
                "work900": run.work(prices, 900),
                "ttft": statistics.mean(e["ttft"] for e in tags),
                "lat": statistics.mean(e["t"] - e["start"] for e in tags),
                "maxlat": max(e["t"] - e["start"] for e in tags),
            }
        )
    return rows


def legacy_pertag(runs: list[Run]) -> dict:
    """notes/churn/pertag.py's numbers: per common tag, in order of first
    start, each run's TTFT, latency and decode rate; per run the common means
    and its extra tags."""
    common = sorted(
        set.intersection(*(set(run.done) for run in runs)),
        key=lambda tag: min(run.done[tag]["start"] for run in runs),
    )
    table = [
        [
            tag,
            [
                [r.done[tag]["ttft"], r.latency(tag), r.done[tag]["decode_tps"] or 0]
                for r in runs
            ],
        ]
        for tag in common
    ]
    summary = []
    for run in runs:
        tags = [run.done[t] for t in common]
        summary.append(
            {
                "label": run.label,
                "common": len(common),
                "ttft_mean": sum(e["ttft"] for e in tags) / max(1, len(tags)),
                "lat_mean": sum(e["t"] - e["start"] for e in tags) / max(1, len(tags)),
                "extra": [
                    [t, round(run.done[t]["ttft"]), round(run.latency(t))]
                    for t in sorted(set(run.done) - set(common))
                ],
            }
        )
    return {"table": table, "summary": summary}


def prefill_walls(run: Run) -> dict:
    """Each completed request's prefill wall (amend/V.md A08): from the first
    sample where its lane's processed prompt rose above where it started
    (its cached prefix) to the first where it reached the prompt. Its lane is
    the active request with its prompt tokens first seen while it ran."""
    lanes: dict[int, list] = {}
    for sample in run.samples:
        for lane in sample["status"].get("active_requests") or []:
            lanes.setdefault(lane["id"], []).append((sample["t"], lane))
    walls = {}
    for tag, record in run.done.items():
        for observed in lanes.values():
            first_t, first = observed[0]
            if first["prompt_tokens"] != record.get("prompt") or not (
                record["start"] - 2.5 <= first_t <= record["t"] + 2.5
            ):
                continue
            start = first["prompt_processed"]
            rose = next(
                (t for t, lane in observed if lane["prompt_processed"] > start), None
            )
            reached = next(
                (
                    t
                    for t, lane in observed
                    if lane["prompt_processed"] >= lane["prompt_tokens"]
                ),
                None,
            )
            if rose is not None and reached is not None and reached > rose:
                walls[tag] = reached - rose
            break
    return walls


def ssd_counters(run: Run, work: float | None) -> dict:
    """amend/V.md A30: SSD bytes per unit of work, both at the run's work
    second."""
    status = nearest(run.samples, run.window)
    if status is None or not work:
        return {}
    experts = status.get("streamed_experts") or {}
    return {
        "disk_read_per_work": value(status, "disk.read_bytes", 0) / work,
        "disk_written_per_work": value(status, "disk.written_bytes", 0) / work,
        "expert_read_per_work": (
            value(experts, "prefill.read_bytes", 0)
            + value(experts, "decode.read_bytes", 0)
        )
        / work,
    }


def store_churn(run: Run, horizon: float) -> dict:
    """The experts' grows and shrinks over the run, and the units taken back
    within `horizon` seconds of the regrow that brought them (amend/V.md
    A30: must be none)."""
    series = [(s["t"], store_size(s["status"])) for s in run.samples]
    series = [(t, size) for t, size in series if size is not None]
    grows = shrinks = retaken = 0
    last_grow = None
    for (_, before), (t, after) in zip(series, series[1:]):
        if after > before:
            grows += 1
            last_grow = t
        elif after < before:
            shrinks += 1
            if last_grow is not None and t - last_grow <= horizon:
                retaken += 1
    return {"grows": grows, "shrinks": shrinks, "taken_back_within_horizon": retaken}


def queue_readouts(run: Run) -> dict:
    """amend/V.md A05, per 2-s sample: the oldest queued request's age, and
    prefill lanes whose processed prompt did not move while another's did."""
    oldest, left_out, previous = 0.0, 0, None
    for sample in run.samples:
        lanes = sample["status"].get("active_requests") or []
        oldest = max(
            [
                oldest,
                *(lane["age_ms"] for lane in lanes if lane.get("phase") == "queued"),
            ]
        )
        current = {
            lane["id"]: lane["prompt_processed"]
            for lane in lanes
            if lane.get("phase") == "prefill"
        }
        if previous is not None:
            held = [i for i in current if i in previous]
            if any(current[i] > previous[i] for i in held):
                left_out += sum(1 for i in held if current[i] == previous[i])
        previous = current
    return {"oldest_queued_ms": oldest, "left_out_lane_samples": left_out}


def counter_delta(run: Run, path: str) -> float:
    if len(run.samples) < 2:
        return 0
    return value(run.last, path, 0) - value(run.samples[0]["status"], path, 0)


def gap_bound(status: dict, share: float) -> dict | None:
    """amend/V.md A09: the run's own decode gap bound max(G, u/share) with u
    the units term of its last prefill estimate, and that estimate's p95
    relative error (schema 7; None for older trees, and while the scheduler
    reports no target: the contract renders it 0 until a scheduler sets it)."""
    scheduler = status.get("scheduler") or {}
    target, estimate = (
        scheduler.get("decode_gap_target_ms"),
        value(scheduler, "estimate.prefill"),
    )
    if not target or estimate is None:
        return None
    unit = next(
        (
            t["ms_per_unit"]
            for t in estimate.get("terms", [])
            if t.get("quantity") == "units"
        ),
        0.0,
    )
    return {
        "bound_ms": max(target, unit / share),
        "e95": value(estimate, "relative_error.p95", 0.0),
        "contended_p95_ms": value(scheduler, "contended_prefill_ms.p95", 0.0),
        "decode_gap_p50_ms": value(status, "metrics.decode_gap_ms.p50", 0.0),
        "decode_gap_max_ms": value(status, "metrics.decode_gap_ms.max", 0.0),
    }


GAP_CAUSES = (
    ("claim wait", lambda a, b: value(b, "memory_governor.claims.waiting", 0) > 0),
    ("store resize", lambda a, b: store_size(a) != store_size(b)),
    (
        "compaction",
        lambda a, b: (
            value(b, "kv.extent_compactions", 0) > value(a, "kv.extent_compactions", 0)
        ),
    ),
    (
        "replan",
        lambda a, b: (
            value(b, "scheduler.replans", 0) > value(a, "scheduler.replans", 0)
        ),
    ),
    (
        "history wait",
        lambda a, b: (
            value(b, "kv_spill.history_stream.wait_ms.total", 0)
            > value(a, "kv_spill.history_stream.wait_ms.total", 0)
        ),
    ),
    (
        "prefill unit",
        lambda a, b: (
            value(b, "scheduler.prefill_batches", 0)
            > value(a, "scheduler.prefill_batches", 0)
        ),
    ),
)


def gap_causes(run: Run, bound_s: float) -> dict:
    """Each client gap above the bound, attributed to the first cause the
    samples around it show, in GAP_CAUSES order, else unattributed."""
    causes: dict[str, int] = {}
    for record in run.done.values():
        for end, length in (record.get("gap") or {}).get("long", []):
            if length <= bound_s:
                continue
            around = [
                s["status"]
                for s in run.samples
                if end - length - 2 <= s["t"] <= end + 2
            ]
            cause = "unattributed"
            if len(around) >= 2:
                cause = next(
                    (name for name, test in GAP_CAUSES if test(around[0], around[-1])),
                    cause,
                )
            causes[cause] = causes.get(cause, 0) + 1
    return causes


def prefill_shares(status: dict) -> dict:
    """Borrowed item 8 (owner 21:45, review/research/STRATA-PC.md item 2),
    reported only, from a streamed run's /status: the prefill hole share,
    the expert loads' host steps (pickup, work and resume) over the prefill
    wall time, an upper bound of the GPU's idle share; and the SSD-busy
    share, the time the seeded weight-read rate gives the prefill's expert
    reads over that wall time. None where the run streamed nothing or its
    status lacks the field."""
    prefill, wall = (
        value(status, "streamed_experts.prefill"),
        value(status, "metrics.prefill_wall_ms"),
    )
    if not prefill or not wall:
        return {"prefill_hole_share": None, "ssd_busy_share": None}
    steps = prefill.get("host_steps")
    seed = next(
        (
            t["seed_ms_per_unit"]
            for t in value(status, "scheduler.estimate.prefill.terms", [])
            if t.get("quantity") == "weight_read_bytes"
        ),
        None,
    )
    return {
        "prefill_hole_share": sum(
            value(steps, f"{part}_ms.total", 0) for part in ("pickup", "work", "resume")
        )
        / wall
        if steps
        else None,
        "ssd_busy_share": prefill.get("read_bytes", 0) * seed / wall if seed else None,
    }


def g8_failures(run: Run, gates: dict) -> list[str]:
    """G8 (amend/V.md §5): the server's own failures a run shows, from its
    /status samples (a ttft run's over its servers, each starting afresh),
    its clients' records and run.py's: an engine restart, a request it
    failed, a Metal failure, a tick of its loop past the GPU wait limit, or
    an exit, a start not ready in time or a failed request that ended the
    run. A voided run's long tick is not judged: its surroundings (another
    GPU load, heat) can make one, and they cause none of the others."""
    failures = []
    restarted = run.restarts()
    if restarted:
        failures.append(
            f"G8: the engine restarted (transport.restarts {restarted[0]} -> {restarted[1]})"
        )
    failed = run.failed_in_status()
    if failed or run.errors:
        failures.append(
            f"G8: {failed} failed requests in /status, {len(run.errors)} at the client"
        )
    if any(
        value(s["status"], "metrics.metal_failures", 0)
        or not value(s["status"], "metal.healthy", True)
        for s in run.samples
    ):
        failures.append("G8: a Metal failure")
    ticks = [value(s["status"], "loop.max_tick_ms", 0) for s in run.samples]
    if not run.voided and ticks and max(ticks) >= gates["max_tick_ms"]:
        failures.append(f"G8: loop.max_tick_ms {max(ticks):.0f}")
    if run.meta.get("failure") in SERVER_FAILURES:
        failures.append(f"G8: {run.meta.get('error')}")
    return failures


def health(run: Run, machine: dict, gates: dict) -> dict:
    """One candidate run on its own: G4 from its client gaps and its own
    status, G8 and G9, and the readouts verdict.md reports."""
    gaps, client_max = run.client_gaps()
    ordered = sorted(gaps)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] if ordered else 0.0
    facts = {"client_gap_max_s": client_max, "client_gap_p99_s": p99, "failures": []}
    if client_max >= machine["g4c_floor_s"]:
        facts["failures"].append(
            f"G4c: a client gap of {client_max:.2f} s, the floor is {machine['g4c_floor_s']} s"
        )
    bound = gap_bound(run.last, gates["pass_overhead_share"])
    if bound:
        limit = bound["bound_ms"] * (1 + bound["e95"])
        facts["g4"] = {**bound, "limit_ms": limit}
        if bound["contended_p95_ms"] > limit:
            facts["failures"].append(
                f"G4a: contended prefill p95 {bound['contended_p95_ms']:.0f} ms over {limit:.0f} ms"
            )
        if p99 * 1000 > limit + bound["decode_gap_p50_ms"]:
            facts["failures"].append(
                f"G4b: client gap p99 {p99 * 1000:.0f} ms over {limit + bound['decode_gap_p50_ms']:.0f} ms"
            )
        facts["gap_causes"] = gap_causes(run, limit / 1000)
    last = run.last
    facts["failures"] += g8_failures(run, gates)
    for sample in run.samples:
        governor = sample["status"].get("memory_governor") or {}
        # Accounts are checked once the ledger reports any.
        accounts = governor.get("accounts")
        if accounts and sum(a["current_bytes"] for a in accounts) != governor.get(
            "charged_bytes"
        ):
            facts["failures"].append(
                f"G9: the accounts do not sum to charged_bytes at t={sample['t']}"
            )
            break
        if value(governor, "claims.oldest_ms", 0) >= gates["resource_wait_ms"]:
            facts["failures"].append(
                f"G9: a claim waited {value(governor, 'claims.oldest_ms'):.0f} ms"
            )
            break
    if value(last, "memory_governor.owed.bytes", 0):
        facts["failures"].append(
            f"G9: {value(last, 'memory_governor.owed.bytes')} bytes owed at the idle end"
        )
    store = store_churn(run, gates["value_horizon_seconds"])
    if store["taken_back_within_horizon"]:
        facts["failures"].append(
            f"store: {store['taken_back_within_horizon']} units taken back within "
            f"{gates['value_horizon_seconds']} s of their regrow"
        )
    units = counter_delta(run, "scheduler.prefill_units")
    facts["readouts"] = {
        "store": store,
        "relative_error": value(last, "scheduler.estimate.prefill.relative_error"),
        "replans_per_unit": counter_delta(run, "scheduler.replans") / units
        if units
        else None,
        "capture_cuts_per_unit": counter_delta(run, "cache.capture_cuts") / units
        if units
        else None,
        "tick_host_ms_p99": value(last, "loop.tick_host_ms.p99"),
        "value_per_gib": {
            a["name"]: a.get("value_per_gib")
            for a in value(last, "memory_governor.accounts", [])
        },
        "spawn_to_ready_s": [
            part.get("spawn_to_ready_s") for part in run.meta.get("parts", [])
        ],
        "first_start": run.meta.get("first_start"),
        **queue_readouts(run),
        **prefill_shares(last),
    }
    return facts


# ------------------------------------------------------------- the gates


def metric_values(runs: list[Run], prices: dict | None) -> dict:
    """Per run the cell metrics: work@900, and over the tags every run
    given completed, the geometric means of latency, TTFT and decode rate."""
    common = sorted(set.intersection(*(set(run.done) for run in runs))) if runs else []
    return {
        run.label: {
            "work": run.work(prices),
            "latency": stats.geometric_mean([run.latency(t) for t in common]),
            "ttft": stats.geometric_mean([run.done[t]["ttft"] for t in common]),
            "decode": stats.geometric_mean([run.done[t]["decode_tps"] for t in common]),
        }
        for run in runs
    }


def sigma_for(
    runs: list[Run], machine: dict, metric: str, prices: dict | None
) -> tuple[float, int]:
    """The machine's per-run sigma of a metric: pooled over this cell's
    groups of one tree in one order, or the recorded value the spec carries
    when they have no degrees of freedom (amend/V.md §0.4)."""
    groups: dict[tuple, list[Run]] = {}
    for run in runs:
        groups.setdefault((run.tree, run.order), []).append(run)
    pooled = []
    for group in groups.values():
        if len(group) > 1:
            per_run = metric_values(group, prices)
            pooled.append([per_run[r.label][metric] for r in group])
    sigma, df = stats.pooled_sigma(pooled)
    if sigma is not None:
        return sigma, df
    prior = machine["sigma"][metric]
    return prior["sigma"], prior["df"]


def compare(
    runs: list[Run], machine: dict, prices, candidate: str, reference: str, metric: str
) -> dict:
    by_order = {}
    for order in ("A", "B"):
        group = [
            r for r in runs if r.order == order and r.tree in (candidate, reference)
        ]
        per_run = metric_values(group, prices)
        by_order[order] = (
            [per_run[r.label][metric] for r in group if r.tree == candidate],
            [per_run[r.label][metric] for r in group if r.tree == reference],
        )
    sigma, df = sigma_for(runs, machine, metric, prices)
    return stats.interval(by_order, sigma, df)


def freeze(runs: list[Run], reference: str, machine: dict) -> dict:
    """The freezes side of the owner's trade rule, measured on the trailed
    reference's own runs: their longest decode gap, and whether it reaches
    the smallest stall the owner rejected on this machine (G4c's floor)."""
    gaps = [
        gap
        for gap in (run.largest_decode_gap() for run in runs if run.tree == reference)
        if gap is not None
    ]
    longest = max(gaps, default=None)
    return {
        "reference_gap_s": longest,
        "floor_s": machine["g4c_floor_s"],
        "reference_froze": longest is not None and longest >= machine["g4c_floor_s"],
    }


def churn_cell(
    cell: dict, spec: dict, runs: list[Run], gains: dict, prices
) -> list[dict]:
    machine, gates = spec["machines"][cell["machine"]], spec["gates"]
    used = [r for r in runs if not r.invalid]
    release = cell.get("release") or {}
    verdicts = []

    def record(rule, result, numbers, **more):
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": rule,
                "verdict": result,
                "numbers": numbers,
                **more,
            }
        )

    # G4, G8 and G9 are judged on every candidate run (amend/V.md §5) but
    # the voided: one the server's own failures made unusable fails G8; one
    # run.py refused to measure or its surroundings voided (heat, other
    # processes, the laptop's owner) is listed under the runs not used, to be
    # repeated, and judged on G8 alone, which fails on the server's own
    # failures it also shows.
    g4_holds = True
    for run in runs:
        if run.tree not in cell.get("health", ["cand"]):
            continue
        if run.voided:
            verdicts.extend(g8_verdict(cell, spec, run))
            continue
        facts = health(run, machine, gates)
        g4_holds &= not any(f.startswith("G4") for f in facts["failures"])
        record(f"G4 G8 G9 {run.label}", FAIL if facts["failures"] else PASS, facts)
    for candidate in cell.get("candidates", ["cand"]):
        if not any(r.tree == candidate for r in used):
            continue
        throughput, ttft = release.get("throughput"), release.get("ttft")
        if throughput:
            estimates = {
                m: compare(used, machine, prices, candidate, throughput, m)
                for m in HIGHER_IS_BETTER
            }
            pooled = {m: e["pooled"] for m, e in estimates.items()}
            # The owner's trade rule (2026-11-01 20:30): throughput and decode
            # speed may trail the release reference by up to the trade
            # tolerance only with a clear gain on TTFT, completion latency,
            # freezes or concurrent TTFT; without one, not at all. Completion
            # latency is a gain, not a metric to trade: it keeps no loss.
            froze = freeze(used, throughput, machine)
            gain = (
                stats.clearly_better(pooled["ttft"], False)
                or stats.clearly_better(pooled["latency"], False)
                or (g4_holds and froze["reference_froze"])
                or gains.get(cell["machine"], False)
            )
            for metric, rule in (
                ("work", "G1"),
                ("latency", "G2"),
                ("decode", "G2 decode"),
            ):
                tolerance = (
                    gates["trade_tolerance"] if gain and metric != "latency" else 0.0
                )
                result = stats.verdict(
                    pooled[metric], HIGHER_IS_BETTER[metric], tolerance, tolerance
                )
                trailing = (
                    pooled[metric]
                    and (pooled[metric]["log_ratio"] < 0) == HIGHER_IS_BETTER[metric]
                )
                record(
                    f"{rule} {candidate} vs {throughput}",
                    result,
                    {
                        "ratio": stats.ratio_text(pooled[metric]),
                        "per_order": {
                            o: stats.ratio_text(e)
                            for o, e in estimates[metric]["orders"].items()
                        },
                        "tolerance": tolerance,
                        "gain": gain,
                        "freeze": froze,
                        "sigma": round(estimates[metric]["sigma"], 4),
                        "df": estimates[metric]["df"],
                    },
                    trade=bool(result == PASS and trailing),
                )
        if ttft:
            estimate = compare(used, machine, prices, candidate, ttft, "ttft")["pooled"]
            result, why = (
                stats.verdict(estimate, False, gates["tolerance"]),
                "interval rule",
            )
            kept = cell.get("kept_decodes_reference")
            if result != PASS and cell.get("memory_cap") and kept and throughput:
                # OQ2 (owner 20:45): under a memory cap G3 also passes when G2
                # passes, decodes never froze, and the TTFT is no worse than
                # the cell's best tree that kept decodes running.
                g2 = next(
                    (
                        v["verdict"]
                        for v in verdicts
                        if v["rule"] == f"G2 {candidate} vs {throughput}"
                    ),
                    FAIL,
                )
                best = compare(used, machine, prices, candidate, kept, "ttft")["pooled"]
                no_worse = stats.verdict(best, False, 0.0) != FAIL
                result = PASS if g2 == PASS and g4_holds and no_worse else FAIL
                why = f"OQ2: G2 {g2}, G4 {'holds' if g4_holds else 'fails'}, TTFT vs {kept} {stats.ratio_text(best)}"
            record(
                f"G3 {candidate} vs {ttft}",
                result,
                {"ratio": stats.ratio_text(estimate), "why": why},
            )
        for reference in cell.get("best_measured", []):
            for metric in ("work", "latency", "ttft"):
                estimate = compare(used, machine, prices, candidate, reference, metric)[
                    "pooled"
                ]
                if (
                    estimate
                    and stats.verdict(estimate, HIGHER_IS_BETTER[metric], 0.0) == FAIL
                ):
                    record(
                        f"{candidate} vs {reference} {metric}",
                        "trade",
                        {"ratio": stats.ratio_text(estimate)},
                        trade=True,
                    )
        if cell.get("ssd"):
            verdicts.extend(ssd_rule(cell, used, prices, candidate))
        if release:
            verdicts.extend(envelope_rule(cell, used, candidate, set(release.values())))
        verdicts.extend(short_hits(cell, used, candidate))
    if cell.get("floor_rule"):
        verdicts.extend(floor_rule(cell, used, machine, prices))
    if cell.get("ample_gate"):
        reference = cell["ample_gate"]
        for metric in ("work", "latency", "ttft"):
            estimate = compare(used, machine, prices, "cand", reference, metric)[
                "pooled"
            ]
            record(
                f"G6 {metric} vs {reference}",
                stats.verdict(estimate, HIGHER_IS_BETTER[metric], gates["tolerance"]),
                {"ratio": stats.ratio_text(estimate)},
            )
    if cell.get("point_gate"):
        verdicts.extend(point_gate(cell, used, prices, gates, g4_holds))
    return verdicts


def point_gate(
    cell: dict, runs: list[Run], prices, gates: dict, g4_holds: bool
) -> list[dict]:
    """G10: work point estimate at least 1 - tolerance and TTFT at most
    1 + tolerance of the reference's, and no G4 failure; a cell without
    runs on both sides is INCONCLUSIVE, for the owner."""
    reference = cell["point_gate"]
    group = [r for r in runs if r.tree in ("cand", reference)]
    per_run = metric_values(group, prices)
    verdicts = []
    for metric in ("work", "ttft"):
        mine = [per_run[r.label][metric] for r in group if r.tree == "cand"]
        theirs = [per_run[r.label][metric] for r in group if r.tree == reference]
        if not mine or not theirs or None in mine + theirs:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G10 {metric} vs {reference}",
                    "verdict": INCONCLUSIVE,
                    "numbers": None,
                }
            )
            continue
        point = stats.geometric_mean(mine) / stats.geometric_mean(theirs)
        ok = (
            point >= 1 - gates["tolerance"]
            if HIGHER_IS_BETTER[metric]
            else point <= 1 + gates["tolerance"]
        )
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"G10 {metric} vs {reference}",
                "verdict": PASS if ok and g4_holds else FAIL,
                "numbers": {"point": round(point, 4), "g4_holds": g4_holds},
            }
        )
    return verdicts


def ssd_rule(cell: dict, runs: list[Run], prices, candidate: str) -> list[dict]:
    """amend/V.md A30: SSD bytes per unit of work against each release
    reference, tolerance 0, sigma from the references' same-tree runs; FAIL
    when the candidate is surely higher than every reference."""
    verdicts = []
    references = cell["ssd"]
    for counter in cell.get(
        "ssd_counters",
        ("disk_read_per_work", "disk_written_per_work", "expert_read_per_work"),
    ):
        groups: dict[tuple, list[float]] = {}
        for run in runs:
            if run.tree in (candidate, *references):
                found = ssd_counters(run, run.work(prices)).get(counter)
                if found:
                    groups.setdefault((run.tree, run.order), []).append(found)
        sigma, df = stats.pooled_sigma(
            [v for (tree, _), v in groups.items() if tree in references]
        )
        if sigma is None:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"SSD {counter}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "no sigma",
                }
            )
            continue
        estimates, higher = {}, []
        for reference in references:
            by_order = {
                o: (groups.get((candidate, o), []), groups.get((reference, o), []))
                for o in ("A", "B")
            }
            estimate = stats.interval(by_order, sigma, df)["pooled"]
            estimates[reference] = stats.ratio_text(estimate)
            higher.append(
                bool(estimate) and estimate["log_ratio"] - estimate["half_width"] > 0
            )
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"SSD {counter}",
                "verdict": FAIL if higher and all(higher) else PASS,
                "numbers": estimates,
            }
        )
    return verdicts


def envelope_rule(
    cell: dict, runs: list[Run], candidate: str, references: set
) -> list[dict]:
    """amend/V.md A08: per tag every run of the order completed, a metric
    FAILs when every candidate run is worse than worst(E) * worst(E)/best(E),
    E the release references' runs."""
    verdicts = []
    for order in ("A", "B"):
        group = [
            r
            for r in runs
            if r.order == order and (r.tree == candidate or r.tree in references)
        ]
        mine = [r for r in group if r.tree == candidate]
        theirs = [r for r in group if r.tree in references]
        if not mine or not theirs:
            continue
        walls = {r.label: prefill_walls(r) for r in group}
        for tag in sorted(set.intersection(*(set(r.done) for r in group))):
            for metric, higher in (
                ("prefill_wall", False),
                ("decode", True),
                ("latency", False),
                ("ttft", False),
            ):

                def get(run, metric=metric, tag=tag):
                    if metric == "prefill_wall":
                        return walls[run.label].get(tag)
                    if metric == "latency":
                        return run.latency(tag)
                    return run.done[tag]["ttft" if metric == "ttft" else "decode_tps"]

                envelope, candidates = [get(r) for r in theirs], [get(r) for r in mine]
                if None in envelope or None in candidates or not all(envelope):
                    continue
                worst, best = (
                    (min(envelope), max(envelope))
                    if higher
                    else (max(envelope), min(envelope))
                )
                bound = worst * worst / best
                if (
                    all(c < bound for c in candidates)
                    if higher
                    else all(c > bound for c in candidates)
                ):
                    verdicts.append(
                        {
                            "cell": cell["name"],
                            "rule": f"tag {order} {tag} {metric}",
                            "verdict": FAIL,
                            "numbers": {
                                "bound": round(bound, 2),
                                "candidate": [round(c, 2) for c in candidates],
                            },
                        }
                    )
    return verdicts


def short_hits(cell: dict, runs: list[Run], candidate: str) -> list[dict]:
    """amend/V.md A70: candidate requests that reused fewer cached tokens
    than every reference run of their tag (prompts are fixed per order)."""
    flags = []
    for order in ("A", "B"):
        references = [r for r in runs if r.order == order and r.tree != candidate]
        for run in (r for r in runs if r.order == order and r.tree == candidate):
            for tag, record in run.done.items():
                seen = [r.done[tag].get("cached") for r in references if tag in r.done]
                seen = [s for s in seen if s is not None]
                if (
                    seen
                    and record.get("cached") is not None
                    and record["cached"] < min(seen)
                ):
                    flags.append([run.label, tag, record["cached"], min(seen)])
    return (
        [
            {
                "cell": cell["name"],
                "rule": "short cache hits",
                "verdict": "report",
                "numbers": flags,
            }
        ]
        if flags
        else []
    )


def floor_rule(cell: dict, runs: list[Run], machine: dict, prices) -> list[dict]:
    """amend/M.md B17: the store held at its cap becomes Flash-Next's floor
    default only if, pooled, none of work, latency and TTFT is worse than the
    value default's and one is better; mixed results go to the owner."""
    rule = cell["floor_rule"]
    results = {}
    for metric in ("work", "latency", "ttft"):
        estimate = compare(
            runs, machine, prices, rule["candidate"], rule["reference"], metric
        )["pooled"]
        results[metric] = [
            stats.verdict(estimate, HIGHER_IS_BETTER[metric], 0.0),
            stats.ratio_text(estimate),
        ]
    better = [m for m, (v, _) in results.items() if v == PASS]
    worse = [m for m, (v, _) in results.items() if v == FAIL]
    decision = (
        "floor = cap"
        if better and not worse
        else "owner"
        if better and worse
        else "floor stays"
    )
    return [
        {
            "cell": cell["name"],
            "rule": "Flash-Next floor default (B17)",
            "verdict": decision,
            "numbers": results,
        }
    ]


# --------------------------------------------------- ttft, ABBA, prices


def load_ttft_rounds(directory: Path, label: str, rounds: int) -> dict:
    """Per concurrency a TTFT run started, its rounds and how many it was to
    run (its start record's, else `rounds`): a new run's LABEL-nN.events
    files, or a recorded ttft.py LABEL.jsonl."""
    levels: dict[int, dict] = {}

    def level(n):
        return levels.setdefault(int(n), {"expected": rounds, "rounds": []})

    paths = sorted(directory.glob(f"{label}-n*.events.jsonl")) or [
        directory / f"{label}.jsonl"
    ]
    for path in paths:
        for record in read_jsonl(path):
            if record.get("kind") == "start":
                started = record.get("args") or record
                for n in str(started["n"]).split(","):
                    level(n)["expected"] = int(started["rounds"])
            elif record.get("kind") == "round":
                level(record["n"])["rounds"].append(record)
    return levels


def round_class(directory: Path, label: str, record: dict) -> str:
    """memory-staggered: between the round's first submit and its last first
    token the store shrank, or a claim or an admission waited; arrival-split:
    queue times differ by more than one prefill unit; else clean
    (amend/V.md A74)."""
    samples = read_jsonl(directory / f"{label}-n{record['n']}.status.jsonl")
    if "window_s" not in record or not samples:
        return "unknown"
    start, end = record["window_s"]
    inside = [s["status"] for s in samples if start <= s["t"] <= end]
    sizes = [store_size(s) for s in inside]
    if any(
        a is not None and b is not None and b < a for a, b in zip(sizes, sizes[1:])
    ) or any(
        value(s, "memory_governor.claims.waiting", 0)
        or value(s, "admission.waiting", 0)
        for s in inside
    ):
        return "memory-staggered"
    unit = next(
        (
            t["ms_per_unit"]
            for s in reversed(inside)
            for t in value(s, "scheduler.estimate.prefill.terms", [])
            if t.get("quantity") == "units"
        ),
        None,
    )
    queue = record.get("queue_ms") or []
    if unit and queue and max(queue) - min(queue) > unit:
        return "arrival-split"
    return "clean"


def unused_entry(cell: dict, label: str, tree: str, reasons: list[str]) -> dict:
    return {"cell": cell["name"], "label": label, "tree": tree, "reasons": reasons}


def g8_verdict(cell: dict, spec: dict, run: Run) -> list[dict]:
    """G8 of a validated run of a candidate tree: of a ttft, serve1 or warm
    run, and of a voided churn run (a used one's is part of its health in
    churn_cell). A voided run passes nothing: it shows only a FAIL, on the
    server's own failures."""
    if run.tree not in cell.get("health", ["cand"]):
        return []
    failures = g8_failures(run, spec["gates"])
    if run.voided and not failures:
        return []
    return [
        {
            "cell": cell["name"],
            "rule": f"G8 {run.label}",
            "verdict": FAIL if failures else PASS,
            "numbers": failures,
        }
    ]


def ttft_rounds(cell: dict, spec: dict, results: list[Path]) -> tuple[dict, list, list]:
    """{tree: {n: [(directory, label, round)]}} of a TTFT cell's recorded and
    planned runs that are valid, over the concurrencies whose every round
    completed its N requests (A104's structural minimum), the runs and
    concurrencies left out, with why, and the candidate runs' G8."""
    labels = {tree: list(found) for tree, found in (cell.get("recorded") or {}).items()}
    for tree, orders in (cell.get("runs") or {}).items():
        for order, count in orders.items():
            labels.setdefault(tree, []).extend(
                run_label(cell["name"], tree, order, r) for r in range(1, count + 1)
            )
    found: dict = {}
    unused, health_verdicts = [], []
    for tree, names in labels.items():
        for label in names:
            directory = locate(results, label)
            if directory is None:
                continue
            run = Run(directory, label, cell["name"], tree)
            validate(run, spec, cell)
            health_verdicts += g8_verdict(cell, spec, run)
            if run.invalid:
                unused.append(unused_entry(cell, label, tree, run.invalid))
                continue
            levels = load_ttft_rounds(directory, label, cell.get("rounds", 3))
            for n, level in sorted(levels.items()):
                rounds = level["rounds"]
                if len(rounds) < level["expected"] or any(
                    len(r["ttft_ms"]) != n for r in rounds
                ):
                    unused.append(
                        unused_entry(
                            cell,
                            label,
                            tree,
                            [
                                f"N={n}: {len(rounds)} of {level['expected']} rounds "
                                f"completed their {n} requests"
                            ],
                        )
                    )
                    continue
                found.setdefault(tree, {}).setdefault(n, []).extend(
                    (directory, label, r) for r in rounds
                )
    return found, unused, health_verdicts


def ttft_cell(cell: dict, spec: dict, results: list[Path]) -> tuple[list, bool, list]:
    """G5 (amend/V.md A74): per N the median over rounds of each round's
    worst TTFT, cand at most the reference's times 1 + r_N, r_N the
    reference's spread of round worsts over its rounds other than the
    largest; N=1 and the first round reported. With point_gate, G10's point
    rule instead. Each candidate run's G8. Also whether cand's concurrent
    TTFT clearly beats the gain reference's (a gain for the owner's trade
    rule), and the runs and concurrencies not used."""
    found, unused, health_verdicts = ttft_rounds(cell, spec, results)
    verdicts, gain = [], False
    gate = "G10" if cell.get("point_gate") else "G5"
    for n in cell.get("gate_n", [2, 4]):
        reference = cell.get("point_gate") or cell["reference"]
        theirs = [max(r["ttft_ms"]) for _, _, r in found.get(reference, {}).get(n, [])]
        candidate = found.get("cand", {}).get(n, [])
        mine = [max(r["ttft_ms"]) for _, _, r in candidate]
        if not theirs or not mine:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"{gate} N={n}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "no runs",
                }
            )
            continue
        if cell.get("point_gate"):
            point = statistics.median(mine) / statistics.median(theirs)
            result = PASS if point <= 1 + cell.get("tolerance", 0.02) else FAIL
            numbers = {"point": round(point, 4)}
        else:
            trimmed = sorted(theirs)[:-1]
            spread = max(trimmed) / min(trimmed) - 1 if len(trimmed) > 1 else 0.0
            limit = statistics.median(theirs) * (1 + spread)
            result = PASS if statistics.median(mine) <= limit else FAIL
            numbers = {
                "cand_median_worst_ms": statistics.median(mine),
                "limit_ms": round(limit, 1),
                "reference_spread": round(spread, 4),
                "first_round_worst_ms": mine[0],
                "classes": [round_class(d, label, r) for d, label, r in candidate],
                "short_cache_hits": [
                    r["round"]
                    for _, _, r in candidate
                    if min(r["cached"]) < (cell.get("cached_min") or {}).get(str(n), 0)
                ],
            }
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"{gate} N={n}",
                "verdict": result,
                "numbers": numbers,
            }
        )
        gain_reference = cell.get("gain_reference")
        if gain_reference and n == max(cell.get("gate_n", [2, 4])):
            others = sorted(
                max(r["ttft_ms"])
                for _, _, r in found.get(gain_reference, {}).get(n, [])
            )
            gain = bool(others) and max(mine) < min(others)
    if found.get("cand", {}).get(1):
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": "N=1 (report)",
                "verdict": "report",
                "numbers": [max(r["ttft_ms"]) for _, _, r in found["cand"][1]],
            }
        )
    return verdicts + health_verdicts, gain, unused


def abba_metrics(directory: Path, label: str) -> dict:
    """The samples of each metric of a serve1 or warm run."""
    metrics: dict[str, list[float]] = {}
    for record in read_jsonl(directory / f"{label}.events.jsonl"):
        if record.get("kind") == "request":
            metrics.setdefault(f"serve1_{record['size']}_ttft_ms", []).append(
                record["ttft_ms"]
            )
            if record.get("decode_ms_per_token"):
                metrics.setdefault(
                    f"serve1_{record['size']}_decode_ms_per_token", []
                ).append(record["decode_ms_per_token"])
        elif (
            record.get("kind") == "warm"
            and not record["discarded"]
            and record.get("prompt_ms")
        ):
            metrics.setdefault(f"warm_{record['depth']}_prompt_ms", []).append(
                record["prompt_ms"]
            )
    return metrics


def abba_cell(cell: dict, spec: dict, results: list[Path]) -> tuple[list, list]:
    """G7 on in-session ABBA runs (reference, cand, cand, reference, per set):
    every metric must PASS abba.compare; G8 of each cand run; and G8's warm
    start (amend/V.md A16): cand's warm spawn-to-ready at most the
    reference's plus the 256-row warm-up's cost. A set with a run that is
    not valid is not judged; its invalid runs are returned with why."""
    runs = abba_runs(cell)
    verdicts, unused = [], []
    for index in range(0, len(runs), 4):
        labels = [
            run_label(cell["name"], tree, order, repeat)
            for tree, order, repeat in runs[index : index + 4]
        ]
        directories = [locate(results, label) for label in labels]
        invalid = []
        for (tree, _, _), d, label in zip(runs[index : index + 4], directories, labels):
            if d is None:
                continue
            run = Run(d, label, cell["name"], tree)
            validate(run, spec, cell)
            verdicts += g8_verdict(cell, spec, run)
            if run.invalid:
                invalid.append(unused_entry(cell, label, tree, run.invalid))
        unused += invalid
        if None in directories or invalid:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G7 set {index // 4 + 1}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "not run"
                    if None in directories
                    else "runs not used: "
                    + ", ".join(entry["label"] for entry in invalid),
                }
            )
            continue
        rounds = [abba_metrics(d, label) for d, label in zip(directories, labels)]
        for metric in sorted(rounds[0]):
            samples = [r.get(metric, []) for r in rounds]
            if all(samples):
                result = abba.compare_samples(samples)
                verdicts.append(
                    {
                        "cell": cell["name"],
                        "rule": f"G7 {metric} set {index // 4 + 1}",
                        "verdict": result["verdict"],
                        "numbers": abba.describe(result),
                    }
                )
        starts = {}
        for (tree, _, _), d, label in zip(runs[index : index + 4], directories, labels):
            meta = d / f"{label}.meta.json"
            if meta.exists():
                found = json.loads(meta.read_text())
                if not found.get("first_start") and found.get("parts"):
                    starts.setdefault(tree, []).append(
                        found["parts"][0]["spawn_to_ready_s"]
                    )
        warmup = spec["machines"][cell["machine"]].get("warmup_256_s")
        reference = cell["abba"]["reference"]
        if starts.get("cand") and starts.get(reference) and warmup is not None:
            limit = max(starts[reference]) + warmup
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G8 warm start set {index // 4 + 1}",
                    "verdict": PASS if max(starts["cand"]) <= limit else FAIL,
                    "numbers": {
                        "cand_s": starts["cand"],
                        f"{reference}_s": starts[reference],
                        "limit_s": round(limit, 2),
                    },
                }
            )
    return verdicts, unused


def single_cell(cell: dict, results: list[Path]) -> list[dict]:
    """G7 of the backend benchmark comparisons (make test-performance-real
    BASELINE=pass), one per model, each copied beside the run's records as
    LABEL.backend-regression.json."""
    verdicts = []
    for model in cell["models"]:
        label = run_label(cell["name"], f"cand-{model}", "-", 1)
        directory = next(
            (d for d in results if (d / f"{label}.backend-regression.json").exists()),
            None,
        )
        if directory is None:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G7 {model}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "not run",
                }
            )
            continue
        document = json.loads(
            (directory / f"{label}.backend-regression.json").read_text()
        )
        failures = (document.get("comparison") or {}).get(
            "failures", [document.get("error")]
        )
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"G7 {model}",
                "verdict": PASS if document.get("pass") else FAIL,
                "numbers": failures,
            }
        )
    return verdicts


def serve1_prices(label: str, results: list[Path]) -> dict | None:
    """A model's prices from a reference serve1 run on its machine (DESIGN
    §8.3: other models' prices from pass's single-request cell): R the
    prefill tokens per second of the largest prompt, D decode tokens per
    second, each a median."""
    directory = locate(results, label)
    if directory is None:
        return None
    records = [
        r
        for r in read_jsonl(directory / f"{label}.events.jsonl")
        if r.get("kind") == "request"
    ]
    if not records:
        return None
    largest = max(r["size"] for r in records)
    rows = [
        r["prompt"] / (r["ttft_ms"] / 1000) for r in records if r["size"] == largest
    ]
    tokens = [
        1000 / r["decode_ms_per_token"] for r in records if r.get("decode_ms_per_token")
    ]
    return {
        "R": statistics.median(rows),
        "D": statistics.median(tokens),
        "source": label,
    }


# --------------------------------------------------------------- analyze


def analyze(spec: dict, results: list[Path]) -> dict:
    runs = load_runs(spec, results)
    cells = {cell["name"]: cell for cell in spec["cells"]}
    verdicts, invalid, gains = [], [], {}
    for run in runs:
        validate(run, spec, cells[run.cell])
        if run.invalid:
            invalid.append(
                unused_entry(cells[run.cell], run.label, run.tree, run.invalid)
            )
    for cell in spec["cells"]:
        if cell.get("scenario") == "ttft":
            found, gain, unused = ttft_cell(cell, spec, results)
            verdicts.extend(found)
            invalid.extend(unused)
            gains[cell["machine"]] = gains.get(cell["machine"], False) or gain
    for cell in spec["cells"]:
        scenario = cell.get("scenario")
        if scenario == "churn":
            cell_runs = [r for r in runs if r.cell == cell["name"]]
            prices = cell.get("prices") or (
                serve1_prices(cell["prices_from"], results)
                if cell.get("prices_from")
                else None
            )
            if not any(r.tree in cell.get("health", ["cand"]) for r in cell_runs):
                verdicts.append(
                    {
                        "cell": cell["name"],
                        "rule": "cell",
                        "verdict": "not run",
                        "numbers": None,
                    }
                )
            elif prices is None:
                verdicts.append(
                    {
                        "cell": cell["name"],
                        "rule": "prices",
                        "verdict": INCONCLUSIVE,
                        "numbers": f"no {cell.get('prices_from')}",
                    }
                )
            else:
                verdicts.extend(churn_cell(cell, spec, cell_runs, gains, prices))
        elif cell.get("abba"):
            found, unused = abba_cell(cell, spec, results)
            verdicts.extend(found)
            invalid.extend(unused)
        elif scenario == "single":
            verdicts.extend(single_cell(cell, results))
    return {
        "spec": spec["path"],
        "verdicts": verdicts,
        "invalid": invalid,
        "generated": time.time(),
    }


def verdict_markdown(document: dict) -> str:
    lines = [
        "# 1.6.0 matrix verdict",
        "",
        "| cell | rule | verdict | numbers |",
        "| --- | --- | --- | --- |",
    ]
    for v in document["verdicts"]:
        numbers = (
            v["numbers"]
            if isinstance(v["numbers"], str)
            else json.dumps(v["numbers"], default=str)
        )
        lines.append(
            f"| {v['cell']} | {v['rule']} | {str(v['verdict']).upper()} | {numbers} |"
        )
    trades = [v for v in document["verdicts"] if v.get("trade")]
    lines += ["", "## Trades for the owner", ""]
    lines += [
        f"- {v['cell']} {v['rule']}: {json.dumps(v['numbers'], default=str)}"
        for v in trades
    ] or ["- none"]
    lines += ["", "## Runs not used", ""]
    lines += [
        f"- {r['cell']} {r['label']} ({r['tree']}): {'; '.join(r['reasons'])}"
        for r in document["invalid"]
    ] or ["- none"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- check


def check_run(
    directory: Path,
    label: str,
    minutes: float | None,
    machine: dict,
    gates: dict,
    streamed: bool = False,
) -> list[str]:
    """What is wrong with one churn run on its own: no DONE, an end before
    its load window, a worker that completed nothing, an engine restart, HTTP
    errors, a G4/G8/G9 failure of its own status and gaps, or, with
    `streamed`, a model that did not start with its routed experts
    streamed."""
    run = Run(directory, label)
    problems = [] if run.complete else ["the run did not end with DONE"]
    if streamed and not value(
        run.samples[0]["status"] if run.samples else {}, "streamed_experts.cap_bytes"
    ):
        problems.append("the routed experts did not stream from the start")
    if minutes is not None and (run.end is None or run.end["t"] < minutes * 60):
        problems.append(f"the run ended before its {minutes:g} min load window")
    workers = {tag.split("-")[0] for tag in run.done}
    missing = [f"w{w}" for w in range(clients.CHURN_WORKERS) if f"w{w}" not in workers]
    if missing:
        problems.append(f"workers that completed no request: {missing}")
    return problems + health(run, machine, gates)["failures"]


def job_type(name: str) -> str:
    """A queued job's type: its name without the queue number and the
    run's repeat, e.g. 20-m6-fn-churn-cand-A1.sh -> m6-fn-churn-cand."""
    return name.split("-", 1)[-1].removesuffix(".sh").rsplit("-", 1)[0]


def overruns(
    queue: Path, now: float | None = None, churn_minutes: float = 20.0
) -> list[str]:
    """Jobs in QUEUE/running past twice their expected time (owner 13:29):
    the time the first finished job of the same type took on this machine,
    from its log's creation to its last write, else `churn_minutes` per run
    (amend/V.md A104)."""
    now = time.time() if now is None else now
    expected = {}
    for log in sorted((queue / "done").glob("*.sh.log")):
        status = log.stat()
        expected.setdefault(
            job_type(log.name.removesuffix(".log")),
            status.st_mtime - status.st_birthtime,
        )
    late = []
    for log in sorted((queue / "running").glob("*.sh.log")):
        took = now - log.stat().st_birthtime
        allowed = 2 * (
            expected.get(job_type(log.name.removesuffix(".log"))) or churn_minutes * 60
        )
        if took > allowed:
            late.append(
                f"{log.name.removesuffix('.log')}: running {took / 60:.0f} min, twice its expected is {allowed / 60:.0f} min"
            )
    return late


# ------------------------------------------------------------------ plan


def interleave(runs_per_tree: dict[str, dict[str, int]]) -> list[tuple[str, str, int]]:
    """(tree, order, repeat) of a cell's runs so that no tree owns its start
    or end: repeat by repeat, the orders alternating which goes first and the
    trees rotating."""
    trees = list(runs_per_tree)
    orders = sorted({o for counts in runs_per_tree.values() for o in counts})
    most = max(
        (c for counts in runs_per_tree.values() for c in counts.values()), default=0
    )
    sequence = []
    for repeat in range(1, most + 1):
        for index, order in enumerate(orders if repeat % 2 else orders[::-1]):
            shift = (repeat - 1 + index) % len(trees)
            for tree in trees[shift:] + trees[:shift]:
                if runs_per_tree[tree].get(order, 0) >= repeat:
                    sequence.append((tree, order, repeat))
    return sequence


def planned_runs(cell: dict, references: bool) -> list[tuple[str, str, int]]:
    if references:
        return interleave(cell.get("reference_runs") or {})
    if cell.get("abba"):
        return abba_runs(cell)
    return interleave(cell.get("runs") or {})


def draft_path(model: dict, machine: dict) -> str | None:
    """The draft's snapshot in the machine's Hugging Face hub (its HF_HOME's,
    else ~/.cache/huggingface), read by the installer."""
    if not model.get("draft"):
        return None
    owner, name = model["draft"].split("/")
    home = machine.get("hf_home", "$HOME/.cache/huggingface")
    return f"{home}/hub/models--{owner}--{name}/snapshots/{model['draft_revision']}"


def machine_prefix(machine: dict) -> str:
    """What a job runs its commands under on the machine: its Hugging Face
    home and its niceness, where the spec gives them (the mini-M4)."""
    prefix = f'HF_HOME="{machine["hf_home"]}" ' if machine.get("hf_home") else ""
    return prefix + (f"nice -n {machine['nice']} " if machine.get("nice") else "")


def job_script(spec: dict, cell: dict, tree: str, order: str, repeat: int, args) -> str:
    machine = spec["machines"][cell["machine"]]
    base = tree.split("-")[0]
    label = run_label(cell["name"], tree, order, repeat)
    if base not in args.trees:
        raise SystemExit(f"{label}: give --tree {base}=DIR")
    commit = args.commits.get(base) or spec["trees"].get(base, {}).get("commit")
    if cell["scenario"] == "single":
        model_name = tree.split("-", 1)[1]
        model = spec["models"][model_name]
        variables = [
            f"MODEL={model['model']}",
            f"REVISION={model['revision']}",
            f"BASELINE={args.trees[cell['baseline']]}",
        ]
        if model.get("language_only", True):
            variables.append("LANGUAGE_ONLY=1")
        draft = draft_path(model, machine)
        if draft:
            variables.append(f'DRAFT_MODEL="{draft}"')
        results = f"build/release/{model['model'].replace('/', '--').replace(':', '--')}/backend-regression.json"
        return (
            "#!/bin/zsh\n"
            f"# {label}: cell {cell['name']}, {model['model']} against {cell['baseline']} ({spec['path']})\n"
            f"cd {shlex.quote(args.trees[base])} || exit 1\n"
            f"/usr/bin/lockf -k {machine['lock']} env {machine_prefix(machine)}make install "
            f"test-performance-real {' '.join(variables)}\n"
            "status=$?\n"
            f"cp {results} {shlex.quote(args.out)}/{label}.backend-regression.json\n"
            "echo EXIT=$status\n"
        )
    model = spec["models"][cell["model"]]
    server_args = " ".join(
        filter(
            None,
            [
                cell.get("server_args", ""),
                (cell.get("tree_server_args") or {}).get(tree, ""),
            ],
        )
    )
    for key, setting in args.settings.items():
        server_args = server_args.replace("{" + key + "}", setting)
    if "{" in server_args:
        raise SystemExit(
            f"{label}: set the placeholder in {server_args!r} with --set KEY=VALUE"
        )
    command = [
        f"{args.tools}/.venv/bin/python",
        "-m",
        "dev.benchmarks.serving.run",
        "--tree",
        args.trees[base],
        "--tree-name",
        tree,
        "--machine",
        cell["machine"],
        "--model",
        model["model"],
        "--revision",
        model["revision"],
        "--scenario",
        cell["scenario"],
        "--label",
        label,
        "--cell",
        cell["name"],
        "--repeat",
        str(repeat),
        "--port",
        str(machine["port"]),
        "--out",
        args.out,
        # One word: argparse takes a value beginning with "--" only after "=".
        f"--server-args={server_args}",
        "--request-seconds",
        f"{clients.DEADLINE_FACTOR * machine['longest_request_s']:g}",
    ]
    if commit:
        command += ["--commit", commit]
    if cell["scenario"] == "churn":
        command += [
            "--order",
            ",".join(map(str, spec["orders"][order])),
            "--minutes",
            str(cell.get("minutes", 15)),
        ]
    for key, setting in (cell.get("env") or {}).items():
        command += ["--env", f"{key}={setting}"]
    if cell.get("sizes"):
        command += ["--sizes", ",".join(map(str, cell["sizes"]))]
    if cell.get("samples"):
        command += ["--samples", str(cell["samples"])]
    if cell["scenario"] == "warm":
        command += ["--trial", str(repeat)]
    if machine.get("gpu_idle_percent") is not None:
        command += ["--gpu-idle-percent", str(machine["gpu_idle_percent"])]
    if cell["scenario"] == "ttft":
        command += [
            "--n",
            ",".join(map(str, cell.get("n", [1, 2, 4]))),
            "--rounds",
            str(cell.get("rounds", 3)),
        ]
    quoted = " ".join(shlex.quote(word) for word in command)
    draft = draft_path(model, machine)
    if draft:
        quoted += f' --draft-model "{draft}"'
    return (
        "#!/bin/zsh\n"
        f"# {label}: cell {cell['name']}, tree {tree}, order {order}, repeat {repeat} ({spec['path']})\n"
        f"cd {shlex.quote(args.tools)} || exit 1\n"
        f"/usr/bin/lockf -k {machine['lock']} env {machine_prefix(machine)}{quoted}\n"
        "echo EXIT=$?\n"
    )


def plan(spec: dict, args) -> list[Path]:
    jobs = refuse_protected(Path(os.path.expanduser(args.jobs)))
    refuse_protected(Path(os.path.expanduser(args.out)))
    jobs.mkdir(parents=True, exist_ok=True)
    written, number = [], args.first
    for cell in spec["cells"]:
        if cell["machine"] != args.machine or (
            args.cells and cell["name"] not in args.cells
        ):
            continue
        if cell["scenario"] == "single":
            runs = (
                [(f"cand-{model}", "-", 1) for model in cell["models"]]
                if not args.references
                else []
            )
        else:
            runs = planned_runs(cell, args.references)
        for tree, order, repeat in runs:
            path = (
                jobs / f"{number:02d}-{run_label(cell['name'], tree, order, repeat)}.sh"
            )
            path.write_text(job_script(spec, cell, tree, order, repeat, args))
            written.append(path)
            number += 1
    return written


# ------------------------------------------------------------------ main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--machine", required=True)
    p.add_argument(
        "--tree", action="append", default=[], help="NAME=DIR of a tree on that machine"
    )
    p.add_argument(
        "--commit", action="append", default=[], help="NAME=SHA recorded with its runs"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="KEY=VALUE for a cell's {KEY} placeholders",
    )
    p.add_argument(
        "--tools", required=True, help="the checkout whose serving tools run the jobs"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--first", type=int, default=20, help="the first job's queue number")
    p.add_argument("--cells", nargs="*", help="only these cells")
    p.add_argument(
        "--references",
        action="store_true",
        help="plan the cells' reference runs instead",
    )
    c = commands.add_parser("check")
    c.add_argument("directory", type=Path, nargs="?")
    c.add_argument("label", nargs="?")
    c.add_argument("--spec", type=Path, help="takes the machine's floors from it")
    c.add_argument("--machine", default="studio")
    c.add_argument("--minutes", type=float, help="the run's load window")
    c.add_argument(
        "--streamed",
        action="store_true",
        help="the model must have started with its routed experts streamed",
    )
    c.add_argument(
        "--running",
        type=Path,
        help="a gpuq root: list its jobs past twice their expected time",
    )
    a = commands.add_parser("analyze")
    a.add_argument("--spec", required=True, type=Path)
    a.add_argument("--results", required=True, type=Path, nargs="+")
    a.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        # Job scripts name absolute paths of the machine that runs plan.
        args.trees = {
            name: os.path.expanduser(path)
            for name, path in (item.split("=", 1) for item in args.tree)
        }
        args.tools, args.out = (
            os.path.expanduser(args.tools),
            os.path.expanduser(args.out),
        )
        args.commits = dict(item.split("=", 1) for item in args.commit)
        args.settings = dict(item.split("=", 1) for item in args.set)
    if (
        args.command == "check"
        and not args.running
        and not (args.directory and args.label)
    ):
        parser.error("check takes RUN_DIR LABEL, or --running GPUQ")
    return args


DEFAULT_SPEC = Path(__file__).with_name("matrix-1.6.0.json")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        for path in plan(load_spec(args.spec), args):
            print(path)
        return 0
    if args.command == "check":
        if args.running:
            late = overruns(args.running)
            for line in late:
                print(f"LATE {line}")
            return 1 if late else 0
        spec = load_spec(args.spec or DEFAULT_SPEC)
        problems = check_run(
            args.directory,
            args.label,
            args.minutes,
            spec["machines"][args.machine],
            spec["gates"],
            args.streamed,
        )
        for problem in problems:
            print(f"FAIL {problem}")
        print("check: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    out = refuse_protected(args.out)
    out.mkdir(parents=True, exist_ok=True)
    document = analyze(load_spec(args.spec), args.results)
    (out / "verdict.json").write_text(
        json.dumps(document, indent=1, default=str) + "\n"
    )
    (out / "verdict.md").write_text(verdict_markdown(document))
    failed = [v for v in document["verdicts"] if v["verdict"] == FAIL]
    print(
        f"{len(document['verdicts'])} verdicts, {len(failed)} FAIL, {len(document['invalid'])} runs not used"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
