"""The 1.6.0 benchmark matrix: plan its jobs, check its runs, judge them.

    python3 -m dev.benchmarks.serving.matrix plan --spec SPEC --machine m6 \\
        --tree cand=~/dev/vr16/cand --tree pass=~/dev/vr16/pass0bd --commit cand=<sha> \\
        --tools ~/dev/vr16/cand --out ~/bench/1.6.0 --jobs ~/gpuq/pending --first 20
    python3 -m dev.benchmarks.serving.matrix check RUN_DIR LABEL [--minutes 15]
    python3 -m dev.benchmarks.serving.matrix check --running ~/gpuq
    python3 -m dev.benchmarks.serving.matrix disk --spec SPEC --results DIR... --cell CELL
    python3 -m dev.benchmarks.serving.matrix analyze --spec SPEC --results DIR... \\
        --build cand=<build id>[,<build id>...] --out DIR

plan writes one job script per run, each holding the machine's GPU lock
around run.py, interleaved so that no tree owns the start or end of a
machine's queue (--references plans the reference runs a cell still lacks).
check judges one run on its own (completion, structural minimum, health,
bounded gaps, and with its cell every reason analyze would void it for), or
lists the queued jobs running past twice their expected time. disk refuses
(exit 1) a churn run whose free disk would leave its SSD cache incomparable
with its references'. analyze applies the spec's gates
(DESIGN-1.6.0 §8.3 as amended by amend/V.md §5 and the owner's rulings) to
every cell with results, on the runs of each tree's builds under test
(--build; several pooled when the change between them is behaviour-neutral;
a rerun takes the next repeat number, and one copy of a label counts), and
writes verdict.json and
verdict.md: PASS, FAIL or INCONCLUSIVE with the numbers, the trades for the
owner, the runs it did not use and why (a planned run that left no record as
not run), and per fixed-order set of each churn cell the numbers of
bin/cmp.py and pertag.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
# Content chunks closer than this are one token event's: the server sends
# each token of an event as its own chunk, back to back (server/backend.py
# CallbackStreamer.put_tokens), DFlash2 committing about eight a lane per
# command, while a lane's next event comes a decode command later, at least
# 17.3 ms (Flint at width 1, the shortest measured: dev/tests/engine/
# TimingProfile.cpp). Half of that is the split farthest from both.
TOKEN_EVENT_MS = 17.3 / 2
THERMAL_VOID = ("serious", "critical")
# The suffix of a single cell's backend regression result beside its label.
SINGLE_RESULTS = ".backend-regression.json"
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


def abba_runs(cell: dict, sets: int) -> list[tuple[str, str, int]]:
    """The runs of `sets` sets of an in-session ABBA cell, set by set:
    reference, cand, cand, reference."""
    reference = cell["abba"]["reference"]
    runs = []
    for index in range(sets):
        first, second = 2 * index + 1, 2 * index + 2
        runs += [
            (reference, "-", first),
            ("cand", "-", first),
            ("cand", "-", second),
            (reference, "-", second),
        ]
    return runs


def holds(directory: Path, label: str) -> bool:
    """Whether a results directory holds a run: its run.py meta.json,
    written also for a run that did not reach its scenario, or its client's
    records."""
    return (
        (directory / f"{label}.meta.json").exists()
        or (directory / f"{label}.events.jsonl").exists()
        or (directory / f"{label}.jsonl").exists()
        or any(directory.glob(f"{label}-n*.events.jsonl"))
    )


def locate(results: list[Path], label: str) -> Path | None:
    """The first results directory holding a run."""
    return next((directory for directory in results if holds(directory, label)), None)


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
    """One run's records: its done requests, the requests the server failed
    and those the client ended at their deadline, its end, /status samples
    and events (a ttft run's over its levels) and meta, and the reasons it
    may not be used: `invalid` all of them, `voided` those not the server's
    own failure, nor an SSD cache not comparable with its references'. A
    churn run's work is read at `window` seconds."""

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
        text = out.read_text() if out.exists() else ""
        self.complete = text.rstrip().endswith("DONE")
        # The build run.py measured: its meta.json's, or for a run killed
        # before it wrote one, the build its .out named before the server
        # started ("root ... build ID").
        self.build = self.meta.get("build_id") or next(
            (
                line.rsplit(" build ", 1)[1]
                for line in text.splitlines()
                if line.startswith("root ") and " build " in line
            ),
            None,
        )
        events = [
            e
            for path in run_files(directory, label, "events")
            for e in read_jsonl(path)
        ]
        self.done = {e["tag"]: e for e in events if e.get("kind") == "done"}
        # Churn's requests the client ended at their deadline.
        self.expired = [e for e in events if e.get("kind") == "deadline"]
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
        """Every gap between token events after each request's first, in
        seconds, as metrics.decode_gap_ms counts them (chunks less than
        TOKEN_EVENT_MS apart are one event's), and the largest, counting
        the open gaps the end record lists."""
        gaps = []
        for record in self.done.values():
            chunks = record.get("chunks_ms") or []
            gaps += [
                (b - a) / 1000
                for a, b in zip(chunks, chunks[1:])
                if b - a >= TOKEN_EVENT_MS
            ]
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
    """The churn runs of the spec's cells found in the results, every copy
    of a label (analyze keeps one: usable): new runs by their meta.json, the
    runs each cell plans also by their client's records (a run killed before
    it wrote its meta.json), and recorded ones by the labels each cell
    lists."""
    cells = {cell["name"]: cell for cell in spec["cells"]}
    runs, found = [], set()

    def add(directory, label, cell, tree, order):
        found.add((cell["name"], label))
        runs.append(Run(directory, label, cell["name"], tree, order, work_second(cell)))

    for directory in results:
        for path in sorted(directory.glob("*.meta.json")):
            meta = json.loads(path.read_text())
            cell = cells.get(meta.get("cell"))
            if cell and cell.get("scenario") == "churn":
                order = spec["order_names"].get(tuple(meta.get("order") or ()))
                add(directory, meta["label"], cell, meta.get("tree_name"), order)
    for cell in spec["cells"]:
        if cell.get("scenario") != "churn":
            continue
        listed = [
            (tree, order, run_label(cell["name"], tree, order, repeat))
            for tree, order, repeat in planned_runs(cell, False)
        ] + [
            (tree, order, label)
            for tree, orders in (cell.get("recorded") or {}).items()
            for order, labels in orders.items()
            for label in labels
        ]
        for tree, order, label in listed:
            directory = locate(results, label)
            if (cell["name"], label) not in found and directory is not None:
                add(directory, label, cell, tree, order)
    return runs


def not_run(cell: dict, runs: list[Run]) -> list[tuple[str, str]]:
    """(tree, label) of the runs a churn cell plans that left no record."""
    labels = {run.label for run in runs}
    return [
        (tree, label)
        for tree, order, repeat in planned_runs(cell, False)
        if (label := run_label(cell["name"], tree, order, repeat)) not in labels
    ]


def cell_metas(results: list[Path], cell: dict) -> list[dict]:
    """The meta.json of every run.py run of a cell in the results, planned
    or not (a rerun takes the next repeat number)."""
    metas = [
        json.loads(path.read_text())
        for directory in results
        for path in sorted(directory.glob("*.meta.json"))
    ]
    return [meta for meta in metas if meta.get("cell") == cell["name"]]


def runs_of(
    cell: dict, labels: list[tuple[str, str]], results: list[Path]
) -> list[Run]:
    """Every copy in the results of each (tree, label), label by label, a
    label taking the tree it is first listed with."""
    trees: dict[str, str] = {}
    for tree, label in labels:
        trees.setdefault(label, tree)
    return [
        Run(directory, label, cell["name"], tree)
        for label, tree in trees.items()
        for directory in results
        if holds(directory, label)
    ]


def other_build(spec: dict, tree: str | None, build: str | None) -> str | None:
    """Why a run of a tree that measured `build` (Run.build, or the build a
    comparison's rounds name) did not measure one of the tree's builds under
    test, analyze --build's, or None: cand-floorcap and cand-27b-q4 are
    cand's, and a tree without one takes any. A tree may have several: the
    release binary's confirmation runs pooled with the earlier binary's
    runs when the change between them is behaviour-neutral (AMENDMENTS,
    Final test start)."""
    expected = spec["trees"].get((tree or "").split("-")[0], {}).get("build_id")
    if expected is None:
        return None
    builds = [expected] if isinstance(expected, str) else expected
    if build in builds:
        return None
    return (
        f"build {build or 'not recorded'}, not the build under test {', '.join(builds)}"
    )


def usable(spec: dict, cell: dict, runs: list[Run]) -> tuple[list[Run], list[dict]]:
    """Of a cell's runs found in the results, one per label, the first of
    its tree's build under test, and every other listed with why: one of
    another build, or one more copy of a label (metric_values keys runs by
    label)."""
    kept: dict[str, Run] = {}
    unused = []
    for run in runs:
        reason = other_build(spec, run.tree, run.build)
        if reason is None and run.label in kept:
            reason = f"also in {kept[run.label].directory}, whose run is used"
        if reason:
            unused.append(unused_entry(cell, run.label, run.tree, [reason]))
        else:
            kept[run.label] = run
    return list(kept.values()), unused


def app_assertion(holder: str, ours: set) -> bool:
    """Whether a `pmset -g assertions` line ("pid 351(powerd): ...") is an
    app's: held by a process the run did not start other than powerd. macOS's
    power manager holds "Powerd - Prevent sleep while display is on" itself
    whenever the display is on, on the M5 Pro (displaysleep 0) always: every
    sample of its six references of 11-01 shows it, at HIDIdleTime 11,263 s
    or more. It is a system process following the display, not an app in
    use; the owner's use shows in HIDIdleTime."""
    _, found, rest = holder.partition("pid ")
    digits, _, rest = rest.partition("(")
    pid = int(digits) if found and digits.strip().isdigit() else None
    return pid not in ours and rest.partition(")")[0] != "powerd"


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
            h for s in run.idle for h in s.get("holders", []) if app_assertion(h, ours)
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
        if not at_cap:
            void("ample start: the experts were not at their cap at Ready")
        # amend/V.md A10's pressure is the host's own before the run, macOS's
        # level as run.py takes it before it starts the server. /status at
        # Ready does not show it: memory_pressure is the engine's band state
        # (AMENDMENTS X9), warning in 33-95 % of the recorded ample runs'
        # samples, and right after the load macOS's own level read warning
        # for the first 42-44 s of sNFb and sNFa, then normal to their end.
        level = value(meta, "host_at_start.pressure_level")
        if level not in (None, 1):
            void(
                f"ample start: macOS's memory pressure level was {level} before the run"
            )


def ssd_cache(run: Run, tolerance: float) -> dict | None:
    """A churn run's SSD cache in its /status samples (disk, on every
    tree): the most it held within its limit (active KV, which may use the
    volume's whole room, can take it past), the smallest limit it came
    within the tolerance of (it filled it; None where it never did), its
    smallest limit, the volume's free space at Ready, and the range of
    limits under which it would have held within the tolerance of what it
    held: from 1 - tolerance of the most it held up to, where it filled a
    limit, that limit over 1 - tolerance. None for a run with no samples."""
    disks = [sample["status"]["disk"] for sample in run.samples]
    if not disks:
        return None
    held = max(min(disk["used_bytes"], disk["limit_bytes"]) for disk in disks)
    filled = min(
        (
            disk["limit_bytes"]
            for disk in disks
            if disk["used_bytes"] >= (1 - tolerance) * disk["limit_bytes"]
        ),
        default=None,
    )
    return {
        "held": held,
        "filled": filled,
        "limit": min(disk["limit_bytes"] for disk in disks),
        "free": disks[0]["volume_available_bytes"],
        "range": (
            (1 - tolerance) * held,
            math.inf if filled is None else filled / (1 - tolerance),
        ),
    }


def compared_trees(cell: dict, tree: str) -> set[str]:
    """The trees a churn cell's gates compare a tree under test with: the
    floor = cap control's its reference; a candidate's the release, best
    measured and SSD references and G3's, G6's and G10's."""
    rule = cell.get("floor_rule") or {}
    if tree == rule.get("candidate"):
        return {rule["reference"]}
    if tree not in cell.get("candidates", ["cand"]):
        return set()
    return {
        *(cell.get("release") or {}).values(),
        *cell.get("best_measured", []),
        *cell.get("ssd", []),
        cell.get("kept_decodes_reference"),
        cell.get("ample_gate"),
        cell.get("point_gate"),
    } - {None}


def compare_ssd_caches(cell: dict, runs: list[Run], tolerance: float) -> None:
    """Leaves out of the comparisons (run.invalid) each usable run of a tree
    under test whose SSD cache is not comparable with that of a usable run
    of a tree its cell's gates compare it with: no limit lies in both runs'
    ranges (ssd_cache). Under --max-cache-disk auto the limit is a quarter
    of the volume's free space beyond a floor (docs/server.md), so it moves
    with whatever else the disk holds that day; it shapes a run only where
    the run fills it, and the gates would credit the code with a cache held
    back in one run and not in the other, or held back at different limits.
    The tolerance is the gates' own: a cache that holds within it is within
    their resolution. Such a run is not voided: its G4, G8 and G9 do not
    depend on the disk, and it is judged on them. A tree compared with a
    tree under test (the floor = cap control with cand) is checked after
    it, against its runs still usable."""

    def gib(size):
        return f"{size / 2**30:.2f} GiB"

    def held(cache):
        return f"held at most {gib(cache['held'])}" + (
            f", filling its {gib(cache['filled'])} limit"
            if cache["filled"] is not None
            else f" under a {gib(cache['limit'])} limit"
        )

    used = [run for run in runs if not run.invalid]
    under_test = {run.tree for run in used if compared_trees(cell, run.tree)}
    for run in sorted(
        used, key=lambda r: bool(compared_trees(cell, r.tree) & under_test)
    ):
        trees = compared_trees(cell, run.tree)
        for other in (r for r in used if r.tree in trees and not r.invalid):
            mine, theirs = ssd_cache(run, tolerance), ssd_cache(other, tolerance)
            if mine is None or theirs is None:
                continue
            if max(mine["range"][0], theirs["range"][0]) > min(
                mine["range"][1], theirs["range"][1]
            ):
                reason = (
                    f"SSD cache not comparable with {other.tree}'s {other.label}: "
                    f"this run {held(mine)} ({gib(mine['free'])} free at Ready); "
                    f"{other.label} {held(theirs)}"
                )
                run.invalid.append(reason)
                break


def ssd_window(cell: dict, runs: list[Run], tree: str, tolerance: float) -> dict:
    """The free disk at a run's start under which compare_ssd_caches keeps
    a run of `tree` comparable with every valid run of the trees its cell
    compares it with, whatever the run holds: its limit L must reach
    (1 - tolerance) of the highest lower end of their ranges (where it fills
    L) and (1 - tolerance) L may not pass the lowest upper end (where it
    holds L). L is a quarter of the free space beyond the engine's floor
    (docs/server.md), the floor as the references' own first samples show
    it (free + held - 4 x limit). Bounds are None where nothing sets them."""
    trees = compared_trees(cell, tree)
    caches = [
        (run, cache)
        for run in runs
        if run.tree in trees and not run.invalid
        for cache in [ssd_cache(run, tolerance)]
        if cache is not None
    ]
    if not caches:
        return {"low": None, "high": None, "references": []}
    floors = [
        first["volume_available_bytes"] + first["used_bytes"] - 4 * first["limit_bytes"]
        for first in (run.samples[0]["status"]["disk"] for run, _ in caches)
    ]
    floor = statistics.median(floors)
    low = max(cache["range"][0] for _, cache in caches) * (1 - tolerance)
    high = min(cache["range"][1] for _, cache in caches) / (1 - tolerance)
    return {
        "low": 4 * low + floor,
        "high": None if math.isinf(high) else 4 * high + floor,
        "references": sorted(run.label for run, _ in caches),
    }


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


def legacy_sets(cell: dict, runs: list[Run], prices: dict) -> list[dict]:
    """bin/cmp.py's and pertag.py's numbers for each fixed-order set of a
    churn cell: its valid runs of the order that ended with DONE, where they
    completed a tag in common."""
    sets = []
    for order in ("A", "B"):
        group = [r for r in runs if r.order == order and r.complete and not r.invalid]
        if group and set.intersection(*(set(r.done) for r in group)):
            sets.append(
                {
                    "cell": cell["name"],
                    "order": order,
                    "cmp": legacy_cmp(group, prices),
                    "pertag": legacy_pertag(group),
                }
            )
    return sets


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


def per_15_minutes(run: Run, path: str) -> float | None:
    """A counter's change over the run's samples per 15 minutes, None where
    its status lacks the counter or holds one sample."""
    span = run.samples[-1]["t"] - run.samples[0]["t"] if run.samples else 0
    if not span or value(run.last, path) is None:
        return None
    return counter_delta(run, path) * 900 / span


def unit_sizes(run: Run) -> dict:
    """amend/V.md A08: the rows of the prefill units the run ran. A command
    adds its rows to scheduler.prefill_rows and one to prefill_batches as it
    commits, and runs them in one unit or, a pass command, in each of its
    passes (prefill_units counts units as they complete), so each sample
    interval that committed commands gives their rows per unit: p10, p50,
    p90 and max over those intervals, and the run's pass commands and units."""
    sizes = []
    for earlier, later in zip(run.samples, run.samples[1:]):
        commands = value(later["status"], "scheduler.prefill_batches", 0) - value(
            earlier["status"], "scheduler.prefill_batches", 0
        )
        if commands > 0:
            rows = value(later["status"], "scheduler.prefill_rows", 0) - value(
                earlier["status"], "scheduler.prefill_rows", 0
            )
            sizes.append(rows / commands)
    ordered = sorted(sizes)
    return {
        "rows_per_unit": {
            "intervals": len(ordered),
            **{
                f"p{round(q * 100)}": ordered[
                    min(len(ordered) - 1, int(q * len(ordered)))
                ]
                for q in (0.1, 0.5, 0.9)
            },
            "max": ordered[-1],
        }
        if ordered
        else None,
        "pass_commands": counter_delta(run, "scheduler.pass_commands"),
        "prefill_units": counter_delta(run, "scheduler.prefill_units"),
    }


def gap_bound(status: dict, share: float) -> dict | None:
    """amend/V.md A09: the run's own decode gap bound max(G, u/share) with u
    the units term of its last prefill estimate, and that estimate's p95
    relative error (schema 7; None for older trees, and while the scheduler
    reports no target: the contract renders it 0 until a scheduler sets it).
    The contended units' p95 is within a millisecond above theirs and no
    higher than the largest (FineLatencyHistogram,
    runtime/model/LatencyHistogram.hpp); trees before it report a log2
    bucket's upper edge, so it is read no higher than the largest unit."""
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
    p95 = value(scheduler, "contended_prefill_ms.p95", 0.0)
    return {
        "bound_ms": max(target, unit / share),
        "e95": value(estimate, "relative_error.p95", 0.0),
        "contended_p95_ms": min(p95, value(scheduler, "contended_prefill_ms.max", p95)),
        "decode_gap_max_ms": value(status, "metrics.decode_gap_ms.max", 0.0),
    }


def floor_unit(run: Run) -> dict | None:
    """The ANE row floor (AMENDMENTS, owner 11-03 03:10): where the Neural
    Engine's FFN split is on, a contended unit is never cut below the split's
    threshold (ane_ffn.minimum_rows, model::CommandLimits::minimumUnitRows;
    the engine keeps it after the split stops), so a unit of those rows may
    take longer than max(G, u/share): as long as the engine prices it. That
    price, from the run's samples: of each interval whose prefill commands,
    with those of the interval before, averaged fewer than twice the floor's
    rows (a prompt's last rows go to one unit or two even ones, Scheduler
    planPrefill), the price per gap of the prefill units the decode gaps
    past the bound waited on (metrics.decode_gap_causes: prefill_predicted_ms
    over prefill_unit.gaps); the longest, over `intervals` of them, as
    unit_ms (None where none shows). None where the split was never on
    (state off, or no ane_ffn: 1.5.0), so those runs keep max(G, u/share)."""
    split = [
        value(s["status"], "ane_ffn")
        for s in run.samples
        if value(s["status"], "ane_ffn.state") in ("split", "stopped")
    ]
    rows = max((value(a, "minimum_rows", 0) for a in split), default=0)
    if rows <= 1:
        return None
    statuses = [s["status"] for s in run.samples]
    prices = []
    for i in range(1, len(statuses)):
        first, earlier, later = statuses[max(0, i - 2)], statuses[i - 1], statuses[i]
        commands, committed = (
            value(later, f"scheduler.{path}", 0) - value(first, f"scheduler.{path}", 0)
            for path in ("prefill_batches", "prefill_rows")
        )
        gaps, priced = (
            value(later, f"metrics.decode_gap_causes.{path}", 0)
            - value(earlier, f"metrics.decode_gap_causes.{path}", 0)
            for path in ("prefill_unit.gaps", "prefill_predicted_ms")
        )
        # A restart sets the counters back: its intervals tell nothing.
        if commands > 0 and 0 <= committed < 2 * rows * commands and gaps > 0 and priced > 0:
            prices.append(priced / gaps)
    return {
        "minimum_rows": rows,
        "unit_ms": max(prices, default=None),
        "intervals": len(prices),
    }


def rose(path: str):
    """A gap cause shown by a counter that rose across the samples."""
    return lambda around: value(around[-1], path, 0) > value(around[0], path, 0)


# The causes of a decode gap (Borrowed item 4, owner 21:05), in the order
# gap_causes tries them on the /status samples around the gap: a claim
# waiting in any of them, the experts' size changing between any two, or a
# counter rising across them. Nearly every churn window commits a prefill
# command, so a prefill unit comes last.
GAP_CAUSES = (
    (
        "claim wait",
        lambda around: any(
            value(s, "memory_governor.claims.waiting", 0) > 0 for s in around
        ),
    ),
    ("store resize", lambda around: len({store_size(s) for s in around}) > 1),
    ("compaction", rose("kv.extent_compactions")),
    ("replan", rose("scheduler.replans")),
    ("history wait", rose("kv_spill.history_stream.wait_ms.total")),
    ("prefill unit", rose("scheduler.prefill_batches")),
)


def gap_causes(run: Run, bound_s: float) -> dict:
    """Every client gap above the bound, attributed to the first cause in
    GAP_CAUSES the samples from 2 s before it to 2 s after it show, else
    unattributed: those between the chunks of each request that ended or
    that the client ended at its deadline (a gap summary lists only those
    of a second or more), and each gap the end record lists as open, ending
    at that record."""
    gaps = [
        ((later - earlier) / 1000, record["start"] + later / 1000)
        for record in [*run.done.values(), *run.expired]
        for chunks in [record.get("chunks_ms") or []]
        for earlier, later in zip(chunks, chunks[1:])
    ] + [
        (length, run.end["t"])
        for length in ((run.end or {}).get("open_gaps") or {}).values()
    ]
    causes: dict[str, int] = {}
    for length, end in gaps:
        if length <= bound_s:
            continue
        around = [
            s["status"] for s in run.samples if end - length - 2 <= s["t"] <= end + 2
        ]
        cause = "unattributed"
        if len(around) >= 2:
            cause = next((name for name, shows in GAP_CAUSES if shows(around)), cause)
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


def clamped_shares(status: dict) -> dict:
    """Borrowed item 6 (owner 21:05), reported for stage 2 to decide on: per
    command estimate (scheduler.estimate.prefill, decode) the commands or
    passes that trained it since the server started, those whose time it
    clamped, and their share; None for an estimate the status lacks or
    nothing trained."""
    shares = {}
    for kind in ("prefill", "decode"):
        estimate = value(status, f"scheduler.estimate.{kind}") or {}
        samples, clamped = estimate.get("samples"), estimate.get("clamped", 0)
        shares[kind] = (
            {"samples": samples, "clamped": clamped, "share": clamped / samples}
            if samples
            else None
        )
    return shares


def prefetch_coverage(run: Run) -> dict:
    """DESIGN §0 Q6, reported only: per use of the streamed experts
    (prefill, decode), of the experts its loads read (not resident) those
    read ahead, Δprefetched over Δread, over the sample intervals in which a
    history stream read (kv_spill.history_stream.bytes rose) and over the
    rest; coverage None where they read none, and None for a use the status
    lacks."""
    coverage = {}
    for use in ("prefill", "decode"):
        read, ahead = (
            f"streamed_experts.{use}.read",
            f"streamed_experts.{use}.prefetched",
        )
        if value(run.last, ahead) is None:
            coverage[use] = None
            continue
        parts = {"history_stream": [0, 0], "rest": [0, 0]}
        for earlier, later in zip(run.samples, run.samples[1:]):
            a, b = earlier["status"], later["status"]
            streamed = value(b, "kv_spill.history_stream.bytes", 0) > value(
                a, "kv_spill.history_stream.bytes", 0
            )
            part = parts["history_stream" if streamed else "rest"]
            part[0] += value(b, read, 0) - value(a, read, 0)
            part[1] += value(b, ahead, 0) - value(a, ahead, 0)
        coverage[use] = {
            name: {
                "read": read_count,
                "prefetched": prefetched,
                "coverage": prefetched / read_count if read_count else None,
            }
            for name, (read_count, prefetched) in parts.items()
        }
    return coverage


def host_memory(run: Run, growth_bound: float) -> tuple[dict, dict]:
    """Borrowed items 2 and 9 (owner 21:05), reported only, from the samples
    whose memory_governor.host_telemetry macOS filled ("not reported" where
    every sample lacks it or has null: the references and every tree before
    M3a), and the series they come from.

    compressed: the most bytes of the engine's phys_footprint macOS held
    compressed and the largest share of it they were; the fix round changes
    a rule only if the engine's own memory is compressed.

    unbooked: phys_footprint less the ledger's accounts, both of one
    snapshot, as its change since the first sample with `ready` true,
    against the live context (each active request's encoded prompt and
    generated tokens): its largest change while no request was live and
    while one was, with that sample's context, and whether either passed
    `growth_bound`, past which the fix round books it."""
    telemetry = [
        (s["t"], s["status"], value(s["status"], "memory_governor.host_telemetry"))
        for s in run.samples
    ]
    telemetry = [(t, status, host) for t, status, host in telemetry if host]
    if not telemetry:
        return {"compressed": "not reported", "unbooked": "not reported"}, {}
    compressed = [
        (
            t,
            host["compressed_bytes"],
            host["compressed_bytes"] / host["phys_footprint_bytes"],
        )
        for t, _, host in telemetry
    ]
    readouts = {
        "compressed": {
            "max_bytes": max(c for _, c, _ in compressed),
            "max_share": round(max(share for _, _, share in compressed), 4),
        },
        "unbooked": "no sample with ready true",
    }
    series = {"compressed": [[t, c, round(share, 4)] for t, c, share in compressed]}
    ready = next(
        (i for i, (_, status, _) in enumerate(telemetry) if status.get("ready")), None
    )
    if ready is None:
        return readouts, series
    unbooked = []
    for t, status, host in telemetry[ready:]:
        booked = sum(
            a["current_bytes"] for a in value(status, "memory_governor.accounts", [])
        )
        context = sum(
            lane["prompt_processed"] + lane["generated_tokens"]
            for lane in status.get("active_requests") or []
        )
        unbooked.append((t, host["phys_footprint_bytes"] - booked, context))
    at_ready = unbooked[0][1]
    growth = [[t, held - at_ready, tokens] for t, held, tokens in unbooked]
    idle = max((g for _, g, tokens in growth if not tokens), default=None)
    live, live_tokens = max(
        ((g, tokens) for _, g, tokens in growth if tokens), default=(None, None)
    )
    readouts["unbooked"] = {
        "at_ready_bytes": at_ready,
        "growth_idle_max_bytes": idle,
        "growth_live_max_bytes": live,
        "context_tokens_at_live_max": live_tokens,
        "bound_bytes": growth_bound,
        "past_bound": any(g > growth_bound for _, g, _ in growth),
    }
    series["unbooked"] = growth
    return readouts, series


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
    status, G8 and G9, and the readouts verdict.md reports. G4b allows the
    typical gap with no prefill unit between two tokens (amend/V.md A09) as
    the median of the same client gaps, exact where metrics.decode_gap_ms'
    p50 is a log2 bucket's upper edge."""
    gaps, client_max = run.client_gaps()
    ordered = sorted(gaps)

    def quantile(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0

    p50, p99 = quantile(0.5), quantile(0.99)
    facts = {
        "client_gap_max_s": client_max,
        "client_gap_p99_s": p99,
        "client_gap_p50_s": p50,
        "failures": [],
    }
    if client_max >= machine["g4c_floor_s"]:
        facts["failures"].append(
            f"G4c: a client gap of {client_max:.2f} s, the floor is {machine['g4c_floor_s']} s"
        )
    bound = gap_bound(run.last, gates["pass_overhead_share"])
    if bound:
        # Where the ANE row floor holds units above max(G, u/share), the bound
        # is the price of the floor's units: max(G, u/share, that price).
        floor = floor_unit(run)
        if floor:
            bound["floor"] = floor
            bound["bound_ms"] = max(bound["bound_ms"], floor["unit_ms"] or 0.0)
        limit = bound["bound_ms"] * (1 + bound["e95"])
        facts["g4"] = {**bound, "limit_ms": limit}
        if bound["contended_p95_ms"] > limit:
            facts["failures"].append(
                f"G4a: contended prefill p95 {bound['contended_p95_ms']:.0f} ms over {limit:.0f} ms"
            )
        if p99 * 1000 > limit + p50 * 1000:
            facts["failures"].append(
                f"G4b: client gap p99 {p99 * 1000:.0f} ms over {limit + p50 * 1000:.0f} ms"
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
    memory, facts["series"] = host_memory(run, gates["unbooked_growth_bytes"])
    facts["readouts"] = {
        "store": store,
        "relative_error": value(last, "scheduler.estimate.prefill.relative_error"),
        "replans_per_unit": counter_delta(run, "scheduler.replans") / units
        if units
        else None,
        "capture_cuts_per_unit": counter_delta(run, "cache.capture_cuts") / units
        if units
        else None,
        "bound_cuts": counter_delta(run, "scheduler.bound_cuts"),
        "index_page_conflicts_per_15min": per_15_minutes(
            run, "scheduler.index_page_conflicts"
        ),
        **unit_sizes(run),
        "tick_host_ms_p99": value(last, "loop.tick_host_ms.p99"),
        "value_per_gib": {
            a["name"]: a.get("value_per_gib")
            for a in value(last, "memory_governor.accounts", [])
        },
        # The experts' value input beside it: the decode loads' line of work
        # against reads (miss_ms, the marginal read; the M6 measures
        # 0.130-0.155 ms, DESIGN section 0, owner 11-03 00:35), so that a
        # value_per_gib off D2 tells a wrong miss price from a wrong ghost rate.
        "miss_line": value(last, "streamed_experts.decode.read_line"),
        "spawn_to_ready_s": [
            part.get("spawn_to_ready_s") for part in run.meta.get("parts", [])
        ],
        "first_start": run.meta.get("first_start"),
        **queue_readouts(run),
        **prefill_shares(last),
        "clamped": clamped_shares(last),
        "prefetch_coverage": prefetch_coverage(run),
        **memory,
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
    # the voided, also on one the server's own failures made unusable (it
    # fails G8) or whose SSD cache is not comparable with its references';
    # one run.py refused to measure or its surroundings voided (heat, other
    # processes, the laptop's owner) is listed under the runs not used, to be
    # repeated, and judged on G8 alone, which fails on the server's own
    # failures it also shows. G4 holds for a tree when none of its runs
    # fails it: each candidate's own, never another tree's (the floor = cap
    # control's), takes part in its gates. Index page conflicts past S's
    # bound are reported to the owner (amend/S.md D1, AMENDMENTS X23).
    g4_holds: dict[str, bool] = {}
    for run in runs:
        if run.tree not in cell.get("health", ["cand"]):
            continue
        if run.voided:
            verdicts.extend(g8_verdict(cell, spec, run))
            continue
        facts = health(run, machine, gates)
        g4_holds[run.tree] = g4_holds.get(run.tree, True) and not any(
            f.startswith("G4") for f in facts["failures"]
        )
        series = facts.pop("series")
        record(
            f"G4 G8 G9 {run.label}",
            FAIL if facts["failures"] else PASS,
            facts,
            series=series,
        )
        unbooked = facts["readouts"]["unbooked"]
        if isinstance(unbooked, dict) and unbooked["past_bound"]:
            record(f"unbooked memory {run.label}", "report", unbooked)
        conflicts = facts["readouts"]["index_page_conflicts_per_15min"]
        if (
            conflicts is not None
            and conflicts > gates["index_page_conflicts_per_15min"]
        ):
            record(
                f"index page conflicts {run.label}",
                "report",
                {
                    "per_15min": round(conflicts, 1),
                    "bound": gates["index_page_conflicts_per_15min"],
                },
            )
    throughput, ttft = release.get("throughput"), release.get("ttft")
    for candidate in cell.get("candidates", ["cand"]):
        holds = g4_holds.get(candidate, True)
        if not any(r.tree == candidate for r in used):
            # Its gates are not judged, and say so (owner OQ3: cells not run
            # by the cut are reported as not run).
            labels = sorted(
                {r.label for r in runs if r.tree == candidate}
                | {label for tree, label in not_run(cell, runs) if tree == candidate}
            )
            why = f"no usable {candidate} run: {', '.join(labels) or 'none planned'}"
            rules = (
                [
                    f"{rule} {candidate} vs {throughput}"
                    for rule in ("G1", "G2", "G2 decode")
                ]
                if throughput
                else []
            )
            rules += [f"G3 {candidate} vs {ttft}"] if ttft else []
            rules += (
                [f"SSD {counter}" for counter in ssd_counters_of(cell)]
                if cell.get("ssd")
                else []
            )
            for rule in rules:
                record(rule, INCONCLUSIVE, why)
            continue
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
            concurrent = gains.get(cell["machine"]) or {}
            gain = bool(
                stats.clearly_better(pooled["ttft"], False)
                or stats.clearly_better(pooled["latency"], False)
                or (holds and froze["reference_froze"])
                or concurrent.get("gain")
            )
            # Each trade goes to the owner with the numbers of every gain
            # it may take (owner 2026-11-01 20:30; freezes: "freeze").
            gain_numbers = {
                f"ttft vs {throughput}": stats.ratio_text(pooled["ttft"]),
                f"latency vs {throughput}": stats.ratio_text(pooled["latency"]),
                "concurrent_ttft": concurrent or None,
            }
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
                        "gains": gain_numbers,
                        "freeze": froze,
                        "sigma": round(estimates[metric]["sigma"], 4),
                        "df": estimates[metric]["df"],
                        **stats.compared_text(estimates[metric]),
                    },
                    trade=bool(result == PASS and trailing),
                )
        if ttft:
            found = compare(used, machine, prices, candidate, ttft, "ttft")
            estimate = found["pooled"]
            result, why = (
                stats.verdict(estimate, False, gates["tolerance"]),
                "interval rule",
            )
            kept = cell.get("kept_decodes_reference")
            if result != PASS and cell.get("memory_cap") and kept and throughput:
                # OQ2 (owner 20:45): under a memory cap G3 also passes when G2
                # passes, decodes never froze, and the TTFT is no worse than
                # the cell's best tree that kept decodes running, which
                # without a usable run of that tree is not known.
                g2 = next(
                    (
                        v["verdict"]
                        for v in verdicts
                        if v["rule"] == f"G2 {candidate} vs {throughput}"
                    ),
                    FAIL,
                )
                best = compare(used, machine, prices, candidate, kept, "ttft")["pooled"]
                if g2 != PASS or not holds or stats.verdict(best, False, 0.0) == FAIL:
                    result = FAIL
                else:
                    result = PASS if best else INCONCLUSIVE
                why = f"OQ2: G2 {g2}, G4 {'holds' if holds else 'fails'}, TTFT vs {kept} {stats.ratio_text(best)}"
            record(
                f"G3 {candidate} vs {ttft}",
                result,
                {
                    "ratio": stats.ratio_text(estimate),
                    "why": why,
                    **stats.compared_text(found),
                },
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
            verdicts.extend(ssd_rule(cell, used, machine, prices, candidate))
        if release:
            verdicts.extend(envelope_rule(cell, used, candidate, set(release.values())))
        verdicts.extend(short_hits(cell, used, candidate))
    if cell.get("floor_rule"):
        verdicts.extend(floor_rule(cell, used, machine, prices))
    if cell.get("ample_gate"):
        reference = cell["ample_gate"]
        for metric in ("work", "latency", "ttft"):
            found = compare(used, machine, prices, "cand", reference, metric)
            record(
                f"G6 {metric} vs {reference}",
                stats.verdict(
                    found["pooled"], HIGHER_IS_BETTER[metric], gates["tolerance"]
                ),
                {
                    "ratio": stats.ratio_text(found["pooled"]),
                    **stats.compared_text(found),
                },
            )
    if cell.get("point_gate"):
        verdicts.extend(
            point_gate(cell, used, machine, prices, gates, g4_holds.get("cand", True))
        )
    return verdicts


def point_gate(
    cell: dict, runs: list[Run], machine: dict, prices, gates: dict, g4_holds: bool
) -> list[dict]:
    """G10 (amend/V.md §5's interval rule; amend/base.md: an undecided G10
    goes to the owner): work and TTFT of cand against the reference, PASS
    when the bound on the good side clears the tolerance, FAIL when the
    bound on the bad side is past equality, else INCONCLUSIVE; a G4 failure
    fails it. A cell without runs on both sides in each order is
    INCONCLUSIVE."""
    reference = cell["point_gate"]
    verdicts = []
    for metric in ("work", "ttft"):
        found = compare(runs, machine, prices, "cand", reference, metric)
        estimate = found["pooled"]
        result = stats.verdict(estimate, HIGHER_IS_BETTER[metric], gates["tolerance"])
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"G10 {metric} vs {reference}",
                "verdict": FAIL if estimate and not g4_holds else result,
                "numbers": {
                    "ratio": stats.ratio_text(estimate),
                    "point": estimate and round(math.exp(estimate["log_ratio"]), 4),
                    "g4_holds": g4_holds,
                    **stats.compared_text(found),
                },
            }
        )
    return verdicts


def ssd_counters_of(cell: dict) -> list[str]:
    """The SSD counters a cell's SSD rule compares (ssd_counters)."""
    return cell.get(
        "ssd_counters",
        ["disk_read_per_work", "disk_written_per_work", "expert_read_per_work"],
    )


def ssd_rule(
    cell: dict, runs: list[Run], machine: dict, prices, candidate: str
) -> list[dict]:
    """amend/V.md A30: SSD bytes per unit of work against each release
    reference, tolerance 0; FAIL when the candidate is surely higher than
    every reference, PASS when it is not surely higher than one, else (a
    reference it cannot be compared with in an order, surely higher than the
    others) INCONCLUSIVE. Sigma as sigma_for takes it: pooled over the
    compared trees' groups of one tree in one order, the candidate's
    included, else the machine's recorded per-run sigma of SSD bytes per
    work (the spec's "ssd"), so the ratios are always reported."""
    verdicts = []
    references = cell["ssd"]
    for counter in ssd_counters_of(cell):
        groups: dict[tuple, list[float]] = {}
        for run in runs:
            if run.tree in (candidate, *references):
                found = ssd_counters(run, run.work(prices)).get(counter)
                if found:
                    groups.setdefault((run.tree, run.order), []).append(found)
        sigma, df = stats.pooled_sigma(list(groups.values()))
        if sigma is None:
            sigma, df = machine["sigma"]["ssd"]["sigma"], machine["sigma"]["ssd"]["df"]
        numbers, higher, open_ = {"sigma": round(sigma, 4), "df": df}, [], []
        for reference in references:
            by_order = {
                o: (groups.get((candidate, o), []), groups.get((reference, o), []))
                for o in ("A", "B")
            }
            found = stats.interval(by_order, sigma, df)
            estimate = found["pooled"]
            numbers[reference] = {
                "ratio": stats.ratio_text(estimate),
                **stats.compared_text(found),
            }
            if estimate is None:
                open_.append(reference)
            else:
                higher.append(estimate["log_ratio"] - estimate["half_width"] > 0)
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": f"SSD {counter}",
                "verdict": (
                    PASS if not all(higher) else INCONCLUSIVE if open_ else FAIL
                ),
                "numbers": numbers,
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
        found = compare(
            runs, machine, prices, rule["candidate"], rule["reference"], metric
        )
        results[metric] = [
            stats.verdict(found["pooled"], HIGHER_IS_BETTER[metric], 0.0),
            stats.ratio_text(found["pooled"]),
            stats.compared_text(found),
        ]
    better = [m for m, (v, *_) in results.items() if v == PASS]
    worse = [m for m, (v, *_) in results.items() if v == FAIL]
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
    """{tree: {n: [(directory, label, round)]}} of a TTFT cell's recorded,
    planned and rerun runs that are valid and of their tree's build under
    test, over the concurrencies whose every round completed its N requests
    (A104's structural minimum), the runs and concurrencies left out, with
    why, and the candidate runs' G8."""
    labels = [
        (tree, label)
        for tree, found in (cell.get("recorded") or {}).items()
        for label in found
    ]
    for tree, orders in (cell.get("runs") or {}).items():
        for order, count in orders.items():
            labels += [
                (tree, run_label(cell["name"], tree, order, r))
                for r in range(1, count + 1)
            ]
    labels += [
        (meta.get("tree_name"), meta["label"]) for meta in cell_metas(results, cell)
    ]
    runs, unused = usable(spec, cell, runs_of(cell, labels, results))
    found: dict = {}
    health_verdicts = []
    for run in runs:
        directory, label, tree = run.directory, run.label, run.tree
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


def ttft_cell(
    cell: dict, spec: dict, results: list[Path]
) -> tuple[list, dict | None, list]:
    """G5 (amend/V.md A74): per N the median over rounds of each round's
    worst TTFT, cand at most the reference's times 1 + r_N, r_N the
    reference's spread of round worsts over its rounds other than the
    largest; N=1 and the first round reported. With point_gate, G10's interval
    rule instead. Each candidate run's G8. Also whether cand's concurrent
    TTFT clearly beats the gain reference's, with both numbers (a gain for
    the owner's trade rule, None without one), and the runs and
    concurrencies not used."""
    found, unused, health_verdicts = ttft_rounds(cell, spec, results)
    verdicts, gain = [], None
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
            # G10 by the interval rule (amend/V.md §5) on the rounds' worst
            # TTFTs, their per-round sigma pooled within each tree's rounds,
            # else the machine's recorded TTFT sigma.
            sigma, df = stats.pooled_sigma([mine, theirs])
            if sigma is None:
                prior = spec["machines"][cell["machine"]]["sigma"]["ttft"]
                sigma, df = prior["sigma"], prior["df"]
            estimate = stats.interval({"-": (mine, theirs)}, sigma, df)["pooled"]
            result = stats.verdict(estimate, False, spec["gates"]["tolerance"])
            numbers = {
                "ratio": stats.ratio_text(estimate),
                "point": round(statistics.median(mine) / statistics.median(theirs), 4),
                "sigma": round(sigma, 4),
                "df": df,
            }
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
            if others:
                gain = {
                    "gain": max(mine) < min(others),
                    "cell": cell["name"],
                    "n": n,
                    "cand_worst_max_ms": max(mine),
                    f"{gain_reference}_worst_min_ms": min(others),
                }
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
    reference's plus the 256-row warm-up's cost. The planned sets and those
    a rerun began (a rerun takes the next repeat numbers) are judged; one
    whose cand runs are all of another build than the build under test is
    passed over, its runs listed, and one with a run that is not valid or
    not of its tree's build is not judged (a second copy of a run it uses
    is only listed); the runs not used are returned with why."""
    planned = cell["abba"].get("sets", 1)
    begun = [((meta.get("repeat") or 0) + 1) // 2 for meta in cell_metas(results, cell)]
    runs = abba_runs(cell, max([planned, *begun]))
    verdicts, unused = [], []
    for index in range(0, len(runs), 4):
        members = runs[index : index + 4]
        labels = [
            run_label(cell["name"], tree, order, repeat)
            for tree, order, repeat in members
        ]
        trees = [tree for tree, _, _ in members]
        found, copies = usable(
            spec, cell, runs_of(cell, list(zip(trees, labels)), results)
        )
        kept = {run.label for run in found}
        cand = [label for tree, label in zip(trees, labels) if tree == "cand"]
        if not kept & set(cand) and {entry["label"] for entry in copies} & set(cand):
            unused += copies + [
                unused_entry(
                    cell,
                    run.label,
                    run.tree,
                    ["its set's cand runs are of another build"],
                )
                for run in found
            ]
            continue
        if not found and not copies and index // 4 >= planned:
            continue
        # A copy of a run that is used leaves its set whole; one of a label
        # none of whose copies is used leaves it without that run.
        unused += copies
        left_out = [entry["label"] for entry in copies if entry["label"] not in kept]
        for run in found:
            validate(run, spec, cell)
            verdicts += g8_verdict(cell, spec, run)
            if run.invalid:
                unused.append(unused_entry(cell, run.label, run.tree, run.invalid))
                left_out.append(run.label)
        if len(found) < 4 or left_out:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G7 set {index // 4 + 1}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "runs not used: " + ", ".join(dict.fromkeys(left_out))
                    if left_out
                    else "not run",
                }
            )
            continue
        directories = [run.directory for run in found]
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
        for run in found:
            if not run.meta.get("first_start") and run.meta.get("parts"):
                starts.setdefault(run.tree, []).append(
                    run.meta["parts"][0]["spawn_to_ready_s"]
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
    if not any(v["rule"].startswith("G7") for v in verdicts):
        verdicts.append(
            {
                "cell": cell["name"],
                "rule": "G7",
                "verdict": INCONCLUSIVE,
                "numbers": "no set of the build under test judged",
            }
        )
    return verdicts, unused


def round_build(document: dict, version: str) -> str | None:
    """The build a backend regression comparison's rounds of one version
    (candidate, baseline) ran, None unless they name one."""
    builds = {
        (record.get("identity") or {}).get("build_id")
        for record in document.get("rounds") or []
        if record.get("version") == version
    }
    return builds.pop() if len(builds) == 1 else None


def single_verdict(document: dict) -> str:
    """G7 of one backend regression comparison: PASS when it passed;
    INCONCLUSIVE when all it failed on is metrics whose ABBA spread was past
    abba's noise limit (they measured nothing: "rerun idle"); else FAIL."""
    if document.get("pass"):
        return PASS
    comparison = document.get("comparison") or {}
    noisy = [
        name
        for name, result in (comparison.get("speed") or {}).items()
        if result.get("verdict") == abba.INCONCLUSIVE
    ]
    others = [
        failure
        for failure in comparison.get("failures", [document.get("error")])
        if not any(str(failure).startswith(f"{name}: ") for name in noisy)
    ]
    images = (document.get("weights") or {}).get("pass") or document.get(
        "expect_image_change"
    )
    return (
        INCONCLUSIVE
        if noisy and not others and images and not document.get("error")
        else FAIL
    )


def single_cell(cell: dict, spec: dict, results: list[Path]) -> tuple[list, list]:
    """G7 of the backend benchmark comparisons (make test-performance-real
    BASELINE=pass), each copied beside the run's records as
    LABEL.backend-regression.json: every repeat of each model's (a rerun
    takes the next repeat number) whose candidate rounds ran the cand build
    under test, and whose baseline rounds the baseline's where analyze is
    given it, and every one that ended in an error before it wrote its
    rounds (backend_regression.py writes them once all four ran), judged on
    that error. The others are returned with why. A comparison too noisy to
    measure (single_verdict's INCONCLUSIVE) is superseded by a later repeat
    that measured, and listed; a FAIL stays."""
    verdicts, unused = [], []
    for model in cell["models"]:
        tree = f"cand-{model}"
        stem = f"{cell['name']}-{tree}-"
        repeats = sorted(
            int(repeat)
            for repeat in {
                path.name[len(stem) : -len(SINGLE_RESULTS)]
                for directory in results
                for path in directory.glob(f"{stem}*{SINGLE_RESULTS}")
            }
            if repeat.isdigit()
        )
        judged: dict[str, Path] = {}
        rows: list[tuple[int, dict]] = []
        for repeat in repeats:
            label = run_label(cell["name"], tree, "-", repeat)
            for directory in results:
                path = directory / f"{label}{SINGLE_RESULTS}"
                if not path.exists():
                    continue
                document = json.loads(path.read_text())
                reason = (
                    other_build(spec, tree, round_build(document, "candidate"))
                    or other_build(
                        spec, cell["baseline"], round_build(document, "baseline")
                    )
                    if document.get("rounds")
                    else None
                )
                # The job's own validity verdict on its surroundings (no
                # run.py here: final160 lib.sh single() writes LABEL.void
                # when the M5 Pro was touched during the make, amend/base.md
                # §2.7).
                void = directory / f"{label}.void"
                if reason is None and void.exists():
                    reason = f"void: {void.read_text().strip()}"
                if reason is None and label in judged:
                    reason = f"also in {judged[label]}, whose run is used"
                if reason:
                    error = document.get("error")
                    unused.append(
                        unused_entry(
                            cell,
                            label,
                            tree,
                            [reason + (f": {error}" if error else "")],
                        )
                    )
                    continue
                judged[label] = directory
                failures = (document.get("comparison") or {}).get(
                    "failures", [document.get("error")]
                )
                rows.append(
                    (
                        repeat,
                        {
                            "cell": cell["name"],
                            "rule": f"G7 {label}",
                            "verdict": single_verdict(document),
                            "numbers": failures,
                        },
                    )
                )
        measured = [r for r, row in rows if row["verdict"] != INCONCLUSIVE]
        for repeat, row in rows:
            later = [r for r in measured if r > repeat]
            if row["verdict"] == INCONCLUSIVE and later:
                unused.append(
                    unused_entry(
                        cell,
                        row["rule"].removeprefix("G7 "),
                        tree,
                        [f"too noisy to measure, superseded by repeat {later[0]}"],
                    )
                )
            else:
                verdicts.append(row)
        if not judged:
            verdicts.append(
                {
                    "cell": cell["name"],
                    "rule": f"G7 {model}",
                    "verdict": INCONCLUSIVE,
                    "numbers": "no run of the build under test"
                    if repeats
                    else "not run",
                }
            )
    return verdicts, unused


def serve1_prices(label: str, results: list[Path]) -> dict | None:
    """A model's prices from a reference serve1 run on its machine (DESIGN
    §8.3: other models' prices from pass's single-request cell): R the
    prefill tokens per second of the largest prompt, D decode tokens per
    second, each a median. Taken from the newest complete run of the
    label's cell and tree (a rerun takes the next repeat number): a run that
    stopped early holds only the sizes it reached."""
    stem = label.rsplit("-", 1)[0] + "-"
    found = sorted(
        (
            (int(path.name[len(stem) : -len(".out")]), directory)
            for directory in results
            for path in directory.glob(f"{stem}*.out")
            if path.name[len(stem) : -len(".out")].isdigit()
        ),
        key=lambda item: -item[0],
    )
    complete = next(
        (
            (directory, f"{stem}{repeat}")
            for repeat, directory in found
            if Run(directory, f"{stem}{repeat}").complete
        ),
        None,
    )
    if complete is None:
        return None
    directory, label = complete
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
    """The verdicts on the runs in the results of each tree's build under
    test (spec trees' build_id, analyze --build), the runs not used and the
    legacy numbers."""
    loaded = load_runs(spec, results)
    cells = {cell["name"]: cell for cell in spec["cells"]}
    verdicts, invalid, gains, legacy, runs = [], [], {}, [], []
    for cell in spec["cells"]:
        kept, copies = usable(spec, cell, [r for r in loaded if r.cell == cell["name"]])
        runs += kept
        invalid += copies
    for run in runs:
        validate(run, spec, cells[run.cell])
    for name, cell in cells.items():
        compare_ssd_caches(
            cell, [r for r in runs if r.cell == name], spec["gates"]["tolerance"]
        )
    for run in runs:
        if run.invalid:
            invalid.append(
                unused_entry(cells[run.cell], run.label, run.tree, run.invalid)
            )
    for cell in spec["cells"]:
        if cell.get("scenario") == "ttft":
            found, gain, unused = ttft_cell(cell, spec, results)
            verdicts.extend(found)
            invalid.extend(unused)
            if gain and not (gains.get(cell["machine"]) or {}).get("gain"):
                gains[cell["machine"]] = gain
    for cell in spec["cells"]:
        scenario = cell.get("scenario")
        if scenario == "churn":
            cell_runs = [r for r in runs if r.cell == cell["name"]]
            invalid += [
                unused_entry(cell, label, tree, ["not run"])
                for tree, label in not_run(
                    cell, [r for r in loaded if r.cell == cell["name"]]
                )
            ]
            prices = cell.get("prices") or (
                serve1_prices(cell["prices_from"], results)
                if cell.get("prices_from")
                else None
            )
            if prices is not None:
                legacy += legacy_sets(cell, cell_runs, prices)
            if not any(r.tree in cell.get("health", ["cand"]) for r in cell_runs):
                verdicts.append(
                    {
                        "cell": cell["name"],
                        "rule": "cell",
                        "verdict": "not run",
                        "numbers": None,
                    }
                )
                continue
            if prices is None:
                # Its work gates are INCONCLUSIVE; the rest are judged.
                verdicts.append(
                    {
                        "cell": cell["name"],
                        "rule": "prices",
                        "verdict": INCONCLUSIVE,
                        "numbers": f"no complete {cell.get('prices_from')} or rerun of it",
                    }
                )
            verdicts.extend(churn_cell(cell, spec, cell_runs, gains, prices))
        elif cell.get("abba"):
            found, unused = abba_cell(cell, spec, results)
            verdicts.extend(found)
            invalid.extend(unused)
        elif scenario == "single":
            found, unused = single_cell(cell, spec, results)
            verdicts.extend(found)
            invalid.extend(unused)
    return {
        "spec": spec["path"],
        "builds": {
            name: tree["build_id"]
            for name, tree in spec["trees"].items()
            if tree.get("build_id")
        },
        "verdicts": verdicts,
        "used": [
            {"cell": r.cell, "label": r.label, "tree": r.tree, "build": r.build}
            for r in runs
            if not r.invalid
        ],
        "invalid": invalid,
        "legacy": legacy,
        "generated": time.time(),
    }


def verdict_markdown(document: dict) -> str:
    lines = [
        "# 1.6.0 matrix verdict",
        "",
        "Builds under test: "
        + ", ".join(
            f"{name} {build if isinstance(build, str) else ' + '.join(build)}"
            for name, build in document["builds"].items()
        ),
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
    lines += ["", "## Per-run numbers (bin/cmp.py, pertag.py)"]

    def number(value):
        return "-" if value is None else f"{value:.0f}"

    for found in document["legacy"]:
        rows = found["cmp"]
        lines += [
            "",
            f"### {found['cell']} order {found['order']}: {rows[0]['common']} common tags",
            "",
        ]
        lines += [
            f"    {row['label']:7s} done={row['done']:2d} work650={number(row['work650']):>6} "
            f"work900={number(row['work900']):>6}  common ttft={row['ttft']:6.1f} "
            f"lat={row['lat']:6.1f} maxlat={row['maxlat']:5.0f}"
            for row in rows
        ]
        lines += [
            f"    {summary['label']}: common n={summary['common']} ttft_mean={summary['ttft_mean']:.0f} "
            f"lat_mean={summary['lat_mean']:.0f}; extra=["
            + ", ".join(
                f"('{tag}', {ttft}, {latency})"
                for tag, ttft, latency in summary["extra"]
            )
            + "]"
            for summary in found["pertag"]["summary"]
        ]
    if not document["legacy"]:
        lines += ["", "- none"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- check


def check_run(
    directory: Path,
    label: str,
    minutes: float | None,
    machine: dict,
    gates: dict,
    streamed: bool = False,
    spec: dict | None = None,
) -> list[str]:
    """What is wrong with one run on its own, so that its job fails at once
    rather than at the batch's analyze: no DONE, the server's own failures
    (G8), and, for a run of a spec cell, every reason analyze would void it
    for (validate: a process it did not start in the top five by CPU, the
    laptop in use or another process's sleep assertion, heat, a busy GPU at
    its start, another model, an ample start below the cap). A churn run
    also: an end before its load window, a worker that completed nothing, a
    G4/G9 failure of its own status and gaps, or, with `streamed`, a model
    that did not start with its routed experts streamed."""
    run = Run(directory, label)
    cell = next(
        (c for c in (spec or {}).get("cells", []) if c["name"] == run.meta.get("cell")),
        None,
    )
    problems = [] if run.complete else ["the run did not end with DONE"]
    if cell is not None:
        run = Run(
            directory,
            label,
            cell["name"],
            run.meta.get("tree_name") or "",
            window=work_second(cell),
        )
        validate(run, spec, cell)
        # An end without DONE is the line above.
        problems += [
            f"voided: {reason}"
            for reason in run.voided
            if not reason.startswith("the run did not end with DONE")
        ]
    if (cell or run.meta).get("scenario", "churn") != "churn":
        return problems + g8_failures(run, gates)
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


def free_bytes(directory: Path) -> int:
    """The free space of the volume holding `directory` as the engine
    measures it (DiskSpace: statfs f_bavail)."""
    found = os.statvfs(directory)
    return found.f_bavail * found.f_frsize


def disk(spec: dict, results: list[Path], name: str, tree: str) -> int:
    """matrix.py disk: whether the volume's free space now lies in the
    window ssd_window gives a run of `tree` in cell `name` against the valid
    runs in the results, printed with it; 1 outside it (analyze would leave
    the run out of the comparisons, compare_ssd_caches), else 0: also for a
    cell outside the spec (the 122B check's), which analyze never reads."""
    cell = next((c for c in spec["cells"] if c["name"] == name), None)
    if cell is None:
        print(f"disk {name}: not a cell of {spec['path']}: no run of it is compared")
        return 0
    runs = [r for r in load_runs(spec, results) if r.cell == name]
    for run in runs:
        validate(run, spec, cell)
    window = ssd_window(cell, runs, tree, spec["gates"]["tolerance"])
    free = free_bytes(results[0])
    inside = (window["low"] is None or free >= window["low"]) and (
        window["high"] is None or free <= window["high"]
    )

    def gib(size):
        return "-" if size is None else f"{size / 2**30:.2f} GiB"

    print(
        f"disk {name} {tree}: {gib(free)} free; the SSD cache stays comparable with "
        f"{', '.join(window['references']) or 'no reference run'} from "
        f"{gib(window['low'])} to {gib(window['high'])} free: "
        + ("ok" if inside else "REFUSED")
    )
    return 0 if inside else 1


def job_type(name: str) -> str:
    """A queued job's type: its name without the queue number and the
    run's repeat, e.g. 20-m6-fn-churn-cand-A1.sh -> m6-fn-churn-cand."""
    return name.split("-", 1)[-1].removesuffix(".sh").rsplit("-", 1)[0]


# The line a final-test job writes inside its locks right before its run
# (final160 lib.sh locked, the Studio's and the mini's cell wrappers).
RUN_STARTED = re.compile(r"\brun started at epoch (\d+)")


def measured_seconds(log: Path) -> float | None:
    """How long a finished job's run took: from its RUN_STARTED line, else
    its log's creation, to the log's last write; None unless the log ends
    with EXIT=0 (a refusal, a failure or a stop measured nothing)."""
    text = log.read_text(errors="replace")
    lines = text.rstrip().splitlines()
    if not lines or lines[-1] != "EXIT=0":
        return None
    status = log.stat()
    started = RUN_STARTED.search(text)
    return status.st_mtime - (int(started.group(1)) if started else status.st_birthtime)


def overruns(
    queue: Path, now: float | None = None, churn_minutes: float = 20.0
) -> list[str]:
    """Jobs in QUEUE/running past twice their expected time (owner 13:29),
    the wait for their locks included: the first measured run of the same
    type on this machine (amend/V.md A104: measured_seconds, its lock wait
    left out), else the minutes the job states (its "EXPECT=N" or "Expected
    ~N min" line, amend/base.md §2.10), else `churn_minutes` per run."""
    now = time.time() if now is None else now
    expected = {}
    for log in sorted((queue / "done").glob("*.sh.log")):
        seconds = measured_seconds(log)
        if seconds is not None:
            expected.setdefault(job_type(log.name.removesuffix(".log")), seconds)
    late = []
    for log in sorted((queue / "running").glob("*.sh.log")):
        took = now - log.stat().st_birthtime
        job = log.with_suffix("")
        stated = re.search(
            r"\bEXPECT=(\d+)|[Ee]xpected:? ~?(\d+) min",
            job.read_text(errors="replace") if job.exists() else "",
        )
        allowed = 2 * (
            expected.get(job_type(log.name.removesuffix(".log")))
            or (stated and 60 * int(stated.group(1) or stated.group(2)))
            or churn_minutes * 60
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
        return abba_runs(cell, cell["abba"].get("sets", 1))
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
        # The comparison is copied only as this make wrote it: an earlier
        # run's, left when make fails before backend_regression.py writes
        # one, is removed first. (status is read-only in zsh.)
        return (
            "#!/bin/zsh\n"
            f"# {label}: cell {cell['name']}, {model['model']} against {cell['baseline']} ({spec['path']})\n"
            f"cd {shlex.quote(args.trees[base])} && rm -f {results} || exit 1\n"
            f"/usr/bin/lockf -k {machine['lock']} env {machine_prefix(machine)}make install "
            f"test-performance-real {' '.join(variables)}\n"
            "code=$?\n"
            f"test -f {results} && cp {results} {shlex.quote(args.out)}/{label}{SINGLE_RESULTS}\n"
            "echo EXIT=$code\n"
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
    c.add_argument(
        "--spec",
        type=Path,
        help="takes the machine's floors and the run's cell (its void rules) from it",
    )
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
    d = commands.add_parser(
        "disk",
        help="refuse a churn run whose free disk would make its SSD cache "
        "incomparable with its references' (exit 1)",
    )
    d.add_argument("--spec", required=True, type=Path)
    d.add_argument("--results", required=True, type=Path, nargs="+")
    d.add_argument("--cell", required=True)
    d.add_argument("--tree", default="cand")
    a = commands.add_parser("analyze")
    a.add_argument("--spec", required=True, type=Path)
    a.add_argument("--results", required=True, type=Path, nargs="+")
    a.add_argument("--out", required=True, type=Path)
    a.add_argument(
        "--build",
        action="append",
        default=[],
        help="NAME=BUILD_ID[,BUILD_ID...]: the builds under test of a tree (cand's "
        "at least: its build/engine/build-identity.json), several pooled when the "
        "change between them is behaviour-neutral; runs of another build are not used",
    )
    args = parser.parse_args(argv)
    if args.command == "analyze":
        args.builds = {}
        for item in args.build:
            name, builds = item.split("=", 1)
            args.builds.setdefault(name, []).extend(b for b in builds.split(",") if b)
        if not args.builds.get("cand"):
            parser.error("analyze judges cand's builds: give --build cand=BUILD_ID")
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
            spec,
        )
        for problem in problems:
            print(f"FAIL {problem}")
        print("check: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    if args.command == "disk":
        return disk(load_spec(args.spec), args.results, args.cell, args.tree)
    out = refuse_protected(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(args.spec)
    for name, builds in args.builds.items():
        if name not in spec["trees"]:
            raise SystemExit(f"--build {name}: {args.spec} has no tree {name}")
        spec["trees"][name]["build_id"] = builds[0] if len(builds) == 1 else builds
    document = analyze(spec, args.results)
    (out / "verdict.json").write_text(
        json.dumps(document, indent=1, default=str) + "\n"
    )
    (out / "verdict.md").write_text(verdict_markdown(document))
    failed = [v for v in document["verdicts"] if v["verdict"] == FAIL]
    open_ = [v for v in document["verdicts"] if v["verdict"] == INCONCLUSIVE]
    print(
        f"{len(document['verdicts'])} verdicts, {len(failed)} FAIL, {len(open_)} INCONCLUSIVE, "
        f"{len(document['invalid'])} runs not used"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
