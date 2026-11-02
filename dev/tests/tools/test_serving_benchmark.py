import hashlib
import json
import math
import os
import re
import shlex
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from dev.benchmarks.serving import clients, matrix, run, stats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "serving"
SPEC = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "serving" / "matrix-1.6.0.json"
)
OLD = json.loads((FIXTURES / "old_clients.json").read_text())
PRICES = {"R": 585, "D": 46}


def synthetic_corpus() -> str:
    # The corpus old_clients.json's ttft digests were taken on.
    return "\n".join(
        f"line {i}: " + "abcdefghij"[i % 10] * (i % 53) for i in range(40000)
    )


def write_jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def status(t, rows=0, tokens=0, **fields):
    document = {
        "scheduler": {"prefill_rows": rows, "prefill_batches": 0},
        "metrics": {"decode_output_tokens": tokens, "metal_failures": 0},
        "metal": {"healthy": True},
        "requests": {"failed": 0},
        "thermal_state": "nominal",
        "memory_pressure": "normal",
        "transport": {"restarts": 0},
    }
    for path, value in fields.items():
        node = document
        keys = path.split("__")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return {"t": t, "status": document}


def churn_run(
    directory: Path, label: str, done, samples, end=None, complete=True
) -> matrix.Run:
    """A churn run's files: done records (tag, start, t, ttft, decode_tps
    and optional chunks_ms, gap, cached, prompt) and /status samples."""
    events = [{"kind": "done", "out": 100, **record} for record in done]
    events.append(
        {"kind": "end", "t": end if end is not None else 960, "open_gaps": {}}
    )
    write_jsonl(directory / f"{label}.events.jsonl", events)
    write_jsonl(directory / f"{label}.status.jsonl", samples)
    (directory / f"{label}.out").write_text("DONE\n" if complete else "")
    return matrix.Run(directory, label)


class ClientRequestTests(unittest.TestCase):
    def test_churn_bodies_are_churn_ords(self):
        # old_clients.json holds the SHA-256 of churn_ord.py's own request
        # bodies, taken from its source (model "fixture-model"): each worker's
        # first rounds and round 5, at the default and the Studio's sizes.
        for setting, expected in OLD["churn_bodies"].items():
            if setting == "resend":
                continue
            base, step = map(int, setting.split(","))
            for worker, round_number, digest in expected:
                body = clients.churn_body(
                    "fixture-model",
                    clients.churn_messages(worker, round_number, base, step),
                )
                self.assertEqual(
                    hashlib.sha256(body.encode()).hexdigest(),
                    digest,
                    (setting, worker, round_number),
                )
        messages = clients.churn_messages(
            0, 3, clients.CHURN_DEFAULT_BASE, clients.CHURN_DEFAULT_STEP
        )
        resend = clients.churn_body(
            "fixture-model", clients.churn_resend(messages, "1\n2\n3")
        )
        self.assertEqual(
            hashlib.sha256(resend.encode()).hexdigest(), OLD["churn_bodies"]["resend"]
        )

    def test_ttft_prompts_are_ttft_pys(self):
        # ttft.py's digest of every warm-up and round request per N, on the
        # same corpus.
        text = synthetic_corpus()
        for n, expected in OLD["ttft_prompts"].items():
            n = int(n)
            digest = hashlib.sha256()
            conversations = [clients.ttft_conversation(text, n, c) for c in range(n)]
            for warm, _ in conversations:
                digest.update(json.dumps(warm).encode())
            for r in range(3):
                for warm, slot in conversations:
                    digest.update(
                        json.dumps(clients.ttft_follow_up(text, warm, slot, r)).encode()
                    )
            self.assertEqual(digest.hexdigest()[:16], expected, n)

    def test_gaps_follow_the_chunks_after_the_first(self):
        start = 1000.0
        chunks = [start + t for t in (1.0, 1.01, 1.03, 2.53, 2.54)]
        summary = clients.gap_summary(chunks, start)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["max_s"], 1.5)
        self.assertEqual(summary["long"], [[2.53, 1.5]])
        # 10 ms and 20 ms gaps fall in [8, 16) and [16, 32); 1.5 s in [1024, 2048).
        self.assertEqual(summary["hist_ms_log2"][3], 2)
        self.assertEqual(summary["hist_ms_log2"][4], 1)
        self.assertEqual(summary["hist_ms_log2"][10], 1)
        self.assertEqual(sum(summary["hist_ms_log2"]), 4)
        self.assertEqual(clients.gap_summary(chunks[:1], start)["n"], 0)

    def test_idle_ignores_restore_points_waiting_for_a_copy(self):
        idle = {
            "requests": {"submitted": 2, "completed": 2},
            "disk": {"write_behind": {"waiting": 3}},
        }
        self.assertFalse(clients.busy(idle))
        self.assertTrue(
            clients.busy({**idle, "memory_governor": {"owed": {"transfers": 1}}})
        )
        self.assertTrue(clients.busy({**idle, "admission": {"waiting": 1}}))
        self.assertTrue(
            clients.busy({**idle, "requests": {"submitted": 3, "completed": 2}})
        )


class LegacyNumberTests(unittest.TestCase):
    """analyze reproduces bin/cmp.py and pertag.py on recorded M6 runs
    (trimmed to what they read; cmp.txt and pertag.txt are their output on the
    full runs, pertag's without its trailing blanks)."""

    def runs(self, labels):
        return [matrix.Run(FIXTURES / "m6", label) for label in labels]

    def test_cmp_numbers(self):
        text = (FIXTURES / "cmp.txt").read_text()
        sets = {"A": ["m150a", "m150a2", "mMNa", "mMNa2"], "B": ["m150b", "mMNb"]}
        pattern = re.compile(
            r"\s+(\S+)\s+done=\s*(\d+) work650=\s*(\d+) work900=\s*(\d+)\s+common ttft=\s*([\d.]+) "
            r"lat=\s*([\d.]+) maxlat=\s*(\d+)"
        )
        expected = {m.group(1): m.groups()[1:] for m in pattern.finditer(text)}
        self.assertEqual(len(expected), 6)
        for name, labels in sets.items():
            rows = matrix.legacy_cmp(self.runs(labels), PRICES)
            self.assertIn(f"order {name}: {rows[0]['common']} common tags", text)
            for row in rows:
                got = (
                    str(row["done"]),
                    f"{row['work650']:.0f}",
                    f"{row['work900']:.0f}",
                    f"{row['ttft']:.1f}",
                    f"{row['lat']:.1f}",
                    f"{row['maxlat']:.0f}",
                )
                self.assertEqual(got, expected[row["label"]], row["label"])

    def test_pertag_numbers(self):
        labels = ["m150a", "m150a2", "mMNa", "mMNa2"]
        lines = (FIXTURES / "pertag.txt").read_text().splitlines()
        result = matrix.legacy_pertag(self.runs(labels))
        rows = [
            line
            for line in lines[1:]
            if not line.startswith(tuple(f"{label}:" for label in labels))
        ]
        self.assertEqual(
            [row.split()[0] for row in rows], [tag for tag, _ in result["table"]]
        )
        for row, (tag, values) in zip(rows, result["table"]):
            cells = re.findall(r"([\d.]+)/\s*([\d.]+)/\s*([\d.]+)", row[len(tag) :])
            self.assertEqual(
                cells,
                [
                    (f"{t:.0f}", f"{lat:.0f}", f"{decode:.1f}")
                    for t, lat, decode in values
                ],
                tag,
            )
        for line, summary in zip(lines[-4:], result["summary"]):
            extra = ", ".join(f"('{t}', {a}, {b})" for t, a, b in summary["extra"])
            self.assertEqual(
                line,
                f"{summary['label']}: common n={summary['common']} ttft_mean={summary['ttft_mean']:.0f} "
                f"lat_mean={summary['lat_mean']:.0f}; extra=[{extra}]",
            )


class IntervalRuleTests(unittest.TestCase):
    def test_t_quantiles(self):
        self.assertAlmostEqual(stats.t_quantile(0.9, 1), 3.0777, places=3)
        self.assertAlmostEqual(stats.t_quantile(0.9, 3), 1.6377, places=3)
        self.assertAlmostEqual(stats.t_quantile(0.9, 10), 1.3722, places=3)
        self.assertAlmostEqual(stats.t_quantile(0.9, 10000), 1.2816, places=3)

    def test_half_widths_are_amend_v_section_4_2s(self):
        # M6 G1: 2 cand vs 3 r150 runs per order, sigma 2.6 % with 10 df.
        by_order = {o: ([1.0, 1.0], [1.0, 1.0, 1.0]) for o in ("A", "B")}
        result = stats.interval(by_order, 0.026, 10)
        self.assertAlmostEqual(result["orders"]["A"]["half_width"], 0.0326, places=4)
        self.assertAlmostEqual(result["pooled"]["half_width"], 0.0230, places=4)

    def test_verdicts(self):
        def estimate(ratio, half):
            return {"log_ratio": math.log(ratio), "half_width": half}

        # Throughput: the lower bound must clear -2 %; FAIL once the upper is below it.
        self.assertEqual(stats.verdict(estimate(1.00, 0.01), True, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(0.95, 0.01), True, 0.02), stats.FAIL)
        self.assertEqual(
            stats.verdict(estimate(0.99, 0.03), True, 0.02), stats.INCONCLUSIVE
        )
        # Latency: the upper bound at most +2 %.
        self.assertEqual(stats.verdict(estimate(1.00, 0.015), False, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(1.05, 0.01), False, 0.02), stats.FAIL)
        self.assertEqual(stats.verdict(None, False, 0.02), stats.INCONCLUSIVE)
        # A tolerance of zero: equal runs cannot pass.
        self.assertEqual(
            stats.verdict(estimate(1.0, 0.02), True, 0.0), stats.INCONCLUSIVE
        )
        self.assertTrue(stats.clearly_better(estimate(0.9, 0.05), False))
        self.assertFalse(stats.clearly_better(estimate(0.98, 0.05), False))

    def test_pooled_sigma(self):
        sigma, df = stats.pooled_sigma(
            [[100, 100 * math.exp(0.02)], [50, 50 * math.exp(-0.02)], [70]]
        )
        self.assertEqual(df, 2)
        self.assertAlmostEqual(sigma, 0.02 / math.sqrt(2), places=6)


class GateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.spec = matrix.load_spec(SPEC)
        self.machine = self.spec["machines"]["m6"]
        self.gates = self.spec["gates"]

    def schema7(self, t, **fields):
        estimate = {
            "terms": [{"quantity": "units", "ms_per_unit": 80.0}],
            "relative_error": {"p50": 0.05, "p95": 0.2},
        }
        defaults = {
            "scheduler__decode_gap_target_ms": 500,
            "scheduler__estimate__prefill": estimate,
            "scheduler__contended_prefill_ms": {"p95": 700.0},
            "metrics__decode_gap_ms": {"p50": 40.0, "max": 900.0},
            "memory_governor__charged_bytes": 300,
            "memory_governor__accounts": [
                {"name": "kv", "current_bytes": 100, "cap_bytes": 100},
                {"name": "experts", "current_bytes": 200, "cap_bytes": 200},
            ],
            "memory_governor__owed": {"bytes": 0},
            "memory_governor__claims": {"oldest_ms": 0},
        }
        return status(t, **{**defaults, **fields})

    def test_g4_takes_the_runs_own_bound(self):
        # bound = max(500, 80 / 0.1) = 800 ms; with e95 0.2 the limit is 960 ms.
        samples = [self.schema7(0), self.schema7(900)]
        smooth = [
            {
                "tag": f"w{w}-r1",
                "start": 0,
                "t": 100,
                "ttft": 5,
                "decode_tps": 30,
                "chunks_ms": [0, 100, 200],
            }
            for w in range(3)
        ]
        facts = matrix.health(
            churn_run(self.directory, "smooth", smooth, samples),
            self.machine,
            self.gates,
        )
        self.assertEqual(facts["failures"], [])
        self.assertAlmostEqual(facts["g4"]["limit_ms"], 960)
        stalled = [dict(smooth[0], chunks_ms=[0, 7400])] + smooth[1:]
        failures = matrix.health(
            churn_run(self.directory, "stall", stalled, samples),
            self.machine,
            self.gates,
        )["failures"]
        self.assertTrue(any(f.startswith("G4c") for f in failures), failures)
        self.assertTrue(any(f.startswith("G4b") for f in failures), failures)
        contended = [
            self.schema7(0),
            self.schema7(900, scheduler__contended_prefill_ms={"p95": 1000.0}),
        ]
        failures = matrix.health(
            churn_run(self.directory, "slow", smooth, contended),
            self.machine,
            self.gates,
        )["failures"]
        self.assertEqual([f[:4] for f in failures], ["G4a:"])

    def test_g9_checks_the_ledger(self):
        smooth = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        leak = [self.schema7(0), self.schema7(900, memory_governor__charged_bytes=301)]
        failures = matrix.health(
            churn_run(self.directory, "leak", smooth, leak), self.machine, self.gates
        )["failures"]
        self.assertTrue(any("charged_bytes" in f for f in failures), failures)
        owed = [
            self.schema7(0),
            self.schema7(900, memory_governor__owed={"bytes": 4096}),
        ]
        failures = matrix.health(
            churn_run(self.directory, "owed", smooth, owed), self.machine, self.gates
        )["failures"]
        self.assertTrue(any("owed" in f for f in failures), failures)

    def test_an_unreported_bound_or_ledger_is_not_judged(self):
        # Schema 7 renders the scheduler's target 0 and the ledger empty until
        # the tracks that own them fill them in, as on the fork base.
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        done[0]["chunks_ms"] = [0, 330, 660]
        unfilled = [
            self.schema7(
                t, scheduler__decode_gap_target_ms=0, memory_governor__accounts=[]
            )
            for t in (0, 900)
        ]
        facts = matrix.health(
            churn_run(self.directory, "unfilled", done, unfilled),
            self.machine,
            self.gates,
        )
        self.assertEqual(facts["failures"], [])
        self.assertNotIn("g4", facts)

    def test_store_taken_back_soon_after_its_regrow(self):
        sizes = [(0, 4), (10, 5), (40, 4), (200, 5), (400, 4)]
        samples = [status(t, streamed_experts__chunks=size) for t, size in sizes]
        run_ = churn_run(self.directory, "store", [], samples)
        self.assertEqual(
            matrix.store_churn(run_, 60),
            {"grows": 2, "shrinks": 2, "taken_back_within_horizon": 1},
        )

    def test_prefill_wall_follows_the_lanes_progress(self):
        lane = {"id": 3, "prompt_tokens": 9000, "phase": "prefill", "age_ms": 0}
        samples = [
            status(0, active_requests=[{**lane, "prompt_processed": 1000}]),
            status(2, active_requests=[{**lane, "prompt_processed": 1000}]),
            status(4, active_requests=[{**lane, "prompt_processed": 3000}]),
            status(
                10,
                active_requests=[{**lane, "prompt_processed": 9000, "phase": "decode"}],
            ),
        ]
        done = [
            {
                "tag": "w0-r1",
                "start": 0.5,
                "t": 40,
                "ttft": 10,
                "decode_tps": 30,
                "prompt": 9000,
            }
        ]
        self.assertEqual(
            matrix.prefill_walls(churn_run(self.directory, "wall", done, samples)),
            {"w0-r1": 6},
        )

    def test_envelope_fails_only_when_every_candidate_run_is_outside(self):
        cell = {"name": "c"}

        def make(label, tree, ttft):
            r = churn_run(
                self.directory,
                label,
                [
                    {
                        "tag": "w0-r1",
                        "start": 0,
                        "t": 100,
                        "ttft": ttft,
                        "decode_tps": 30,
                    }
                ],
                [status(0)],
            )
            r.tree, r.order = tree, "A"
            return r

        references = [
            make("r1", "r150", 70),
            make("r2", "r150", 77),
            make("m1", "main", 72),
        ]
        # worst 77, best 70: bound 77 * 77 / 70 = 84.7.
        outside = matrix.envelope_rule(
            cell,
            references + [make("c1", "cand", 90), make("c2", "cand", 86)],
            "cand",
            {"r150", "main"},
        )
        self.assertEqual([v["rule"] for v in outside], ["tag A w0-r1 ttft"])
        straddling = references + [make("c3", "cand", 90), make("c4", "cand", 80)]
        self.assertEqual(
            matrix.envelope_rule(cell, straddling, "cand", {"r150", "main"}), []
        )

    def test_g5_takes_the_median_round_worst_and_the_references_spread(self):
        def ttft_run(label, worsts):
            write_jsonl(
                self.directory / f"{label}.jsonl",
                [
                    {
                        "kind": "round",
                        "n": 2,
                        "round": i,
                        "ttft_ms": [100, w],
                        "cached": [7968, 7968],
                        "queue_ms": [0, 0],
                    }
                    for i, w in enumerate(worsts)
                ],
            )

        # pass's N=2 rounds (amend/V.md §0.4): median 4622 ms, spread 4622/4399 - 1.
        ttft_run("ref", [6976, 4622, 4399])
        cell = {
            "name": "t",
            "reference": "pass",
            "gate_n": [2],
            "recorded": {"pass": ["ref"], "cand": ["fast"]},
        }
        ttft_run("fast", [4700, 4800, 9000])
        verdicts, _ = matrix.ttft_cell(cell, [self.directory])
        self.assertEqual(verdicts[0]["verdict"], stats.PASS, verdicts)
        self.assertAlmostEqual(
            verdicts[0]["numbers"]["limit_ms"], 4622 * 4622 / 4399, places=0
        )
        ttft_run("slow", [5000, 5100, 5200])
        cell["recorded"]["cand"] = ["slow"]
        self.assertEqual(
            matrix.ttft_cell(cell, [self.directory])[0][0]["verdict"], stats.FAIL
        )

    def test_trade_rule_needs_a_gain_on_the_other_side(self):
        # cand trails r150 by 3 % in work: without a gain it cannot pass; with
        # a clearly better TTFT the 5 % tolerance applies.
        def make(label, tree, order, work, ttft):
            done = [
                {
                    "tag": f"w{w}-r1",
                    "start": 0,
                    "t": 200,
                    "ttft": ttft,
                    "decode_tps": 30,
                }
                for w in range(3)
            ]
            r = churn_run(
                self.directory,
                label,
                done,
                [status(900, rows=int(work * 585), tokens=0)],
            )
            r.tree, r.order = tree, order
            return r

        cell = {
            "name": "c",
            "machine": "m6",
            "release": {"throughput": "r150"},
            "health": [],
        }
        refs = [
            make(f"r{o}{i}", "r150", o, 1000 + i, 100) for o in "AB" for i in range(3)
        ]
        flat = refs + [
            make(f"c{o}{i}", "cand", o, 970 + i, 100) for o in "AB" for i in range(2)
        ]
        g1 = next(
            v
            for v in matrix.churn_cell(cell, self.spec, flat, {}, PRICES)
            if v["rule"].startswith("G1")
        )
        self.assertNotEqual(g1["verdict"], stats.PASS, g1)
        self.assertEqual(g1["numbers"]["tolerance"], 0.0)
        faster = refs + [
            make(f"f{o}{i}", "cand", o, 970 + i, 60) for o in "AB" for i in range(2)
        ]
        g1 = next(
            v
            for v in matrix.churn_cell(cell, self.spec, faster, {}, PRICES)
            if v["rule"].startswith("G1")
        )
        self.assertEqual(g1["verdict"], stats.PASS, g1)
        self.assertTrue(g1["trade"])

    def test_check_finds_a_silent_worker_and_a_short_run(self):
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in (0, 1)
        ]
        churn_run(self.directory, "short", done, [status(0), status(300)], end=310)
        problems = matrix.check_run(
            self.directory, "short", 15, self.machine, self.gates
        )
        self.assertTrue(any("load window" in p for p in problems), problems)
        self.assertTrue(any("['w2']" in p for p in problems), problems)

    def test_running_jobs_past_twice_their_expected_time(self):
        queue = Path(tempfile.mkdtemp())
        (queue / "done").mkdir()
        (queue / "running").mkdir()
        finished = queue / "done" / "20-m6-fn-churn-cand-A1.sh.log"
        finished.write_text("EXIT=0\n")
        born = finished.stat().st_birthtime
        os.utime(finished, (born + 1200, born + 1200))
        (queue / "running" / "21-m6-fn-churn-cand-B1.sh.log").write_text("")
        self.assertEqual(matrix.overruns(queue, now=time.time() + 60), [])
        late = matrix.overruns(queue, now=time.time() + 2500)
        self.assertEqual(len(late), 1)
        self.assertTrue(late[0].startswith("21-m6-fn-churn-cand-B1.sh"), late)

    def test_abba_cell_compares_rounds_in_session(self):
        cell = {"name": "s1", "machine": "m6", "abba": {"reference": "pass", "sets": 1}}
        for (tree, _, repeat), ttft in zip(
            matrix.abba_runs(cell), (1000, 1010, 1005, 1002)
        ):
            write_jsonl(
                self.directory / f"s1-{tree}-{repeat}.events.jsonl",
                [
                    {
                        "kind": "request",
                        "size": 300,
                        "sample": s,
                        "ttft_ms": ttft + s,
                        "decode_ms_per_token": 20.0,
                    }
                    for s in range(3)
                ],
            )
        verdicts = matrix.abba_cell(cell, self.spec, [self.directory])
        self.assertEqual(
            {v["rule"] for v in verdicts},
            {"G7 serve1_300_ttft_ms set 1", "G7 serve1_300_decode_ms_per_token set 1"},
        )
        self.assertTrue(all(v["verdict"] == stats.PASS for v in verdicts), verdicts)


class PlanTests(unittest.TestCase):
    def test_interleaving_owns_no_end(self):
        counts = {"cand": {"A": 2, "B": 2}, "cand-floorcap": {"A": 2, "B": 2}}
        sequence = matrix.interleave(counts)
        self.assertEqual(
            sorted(sequence),
            sorted({(t, o, r) for t in counts for o in "AB" for r in (1, 2)}),
        )
        self.assertNotEqual(sequence[0][0], sequence[-1][0])
        self.assertEqual(
            sequence[:4],
            [
                ("cand", "A", 1),
                ("cand-floorcap", "A", 1),
                ("cand-floorcap", "B", 1),
                ("cand", "B", 1),
            ],
        )

    def test_job_scripts_run_run_py_under_the_lock(self):
        spec = matrix.load_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            arguments = matrix.parse_args(
                [
                    "plan",
                    "--spec",
                    str(SPEC),
                    "--machine",
                    "m6",
                    "--tree",
                    "cand=/trees/cand",
                    "--commit",
                    "cand=abc123",
                    "--set",
                    "experts_cap=7G",
                    "--tools",
                    "/trees/cand",
                    "--out",
                    f"{directory}/out",
                    "--jobs",
                    f"{directory}/jobs",
                    "--cells",
                    "m6-fn-churn",
                ]
            )
            jobs = matrix.plan(spec, arguments)
            self.assertEqual(
                [job.name for job in jobs][:3],
                [
                    "20-m6-fn-churn-cand-A1.sh",
                    "21-m6-fn-churn-cand-floorcap-A1.sh",
                    "22-m6-fn-churn-cand-floorcap-B1.sh",
                ],
            )
            script = jobs[1].read_text().splitlines()
            self.assertEqual(script[2], "cd /trees/cand || exit 1")
            command = shlex.split(script[3].replace('"$HOME', '"/home'))
            self.assertEqual(command[:3], ["/usr/bin/lockf", "-k", "/tmp/m6-gpu.lock"])
            parsed = run.parse_args(
                command[command.index("dev.benchmarks.serving.run") + 1 :]
            )
            self.assertEqual(
                (parsed.scenario, parsed.order_list, parsed.cell),
                ("churn", [2, 0, 1], "m6-fn-churn"),
            )
            self.assertEqual(
                (parsed.tree, parsed.commit), (Path("/trees/cand"), "abc123")
            )
            self.assertEqual(parsed.server_args, "--expert-memory 7G")
            self.assertTrue(
                parsed.draft_model.endswith(
                    "models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"
                )
            )

    def test_every_cell_names_known_trees_models_and_machines(self):
        spec = matrix.load_spec(SPEC)
        for cell in spec["cells"]:
            self.assertIn(cell["machine"], spec["machines"], cell["name"])
            for name in cell.get("models", [cell.get("model")]):
                self.assertIn(name, spec["models"], cell["name"])
            for tree in list(cell.get("recorded") or {}) + list(cell.get("runs") or {}):
                self.assertIn(tree.split("-")[0], spec["trees"], cell["name"])
            if cell.get("prices_from"):
                self.assertTrue(
                    any(
                        cell["prices_from"].startswith(c["name"]) for c in spec["cells"]
                    ),
                    cell["name"],
                )

    def test_every_machines_jobs_parse(self):
        # Every job plan writes, the runs and the reference runs, is a
        # command its tool takes.
        spec = matrix.load_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            for machine in spec["machines"]:
                for references in ([], ["--references"]):
                    arguments = matrix.parse_args(
                        [
                            "plan",
                            "--spec",
                            str(SPEC),
                            "--machine",
                            machine,
                            "--tree",
                            "cand=/trees/cand",
                            "--tree",
                            "pass=/trees/pass",
                            "--set",
                            "experts_cap=7G",
                            "--tools",
                            "/trees/cand",
                            "--out",
                            f"{directory}/out",
                            "--jobs",
                            f"{directory}/{machine}{len(references)}",
                            *references,
                        ]
                    )
                    for job in matrix.plan(spec, arguments):
                        line = (
                            job.read_text().splitlines()[3].replace('"$HOME', '"/home')
                        )
                        words = shlex.split(line)
                        if "dev.benchmarks.serving.run" in words:
                            parsed = run.parse_args(
                                words[words.index("dev.benchmarks.serving.run") + 1 :]
                            )
                            self.assertTrue(job.name.endswith(f"-{parsed.label}.sh"))
                            if machine == "mini":
                                # Its own HF home, nice 15 and the GPU idle check.
                                self.assertEqual(
                                    words[3:7],
                                    [
                                        "env",
                                        "HF_HOME=/home/dev/splash-lead/hf",
                                        "nice",
                                        "-n",
                                    ],
                                )
                                self.assertEqual(parsed.gpu_idle_percent, 0)
                                self.assertTrue(
                                    parsed.draft_model.startswith(
                                        "/home/dev/splash-lead/hf/hub/"
                                    )
                                )
                        else:
                            self.assertIn("test-performance-real", words, job.name)
                            self.assertIn("BASELINE=/trees/pass", words, job.name)

    def test_analyze_judges_every_cell(self):
        # Recorded references alone: every candidate cell is "not run", and
        # every recorded run is valid.
        document = matrix.analyze(matrix.load_spec(SPEC), [FIXTURES / "m6"])
        verdicts = {(v["cell"], v["rule"]): v["verdict"] for v in document["verdicts"]}
        self.assertEqual(verdicts[("m6-fn-churn", "cell")], "not run")
        self.assertEqual(verdicts[("m6-fn-ttft", "G5 N=2")], stats.INCONCLUSIVE)
        self.assertEqual(document["invalid"], [])
        self.assertIn("## Runs not used", matrix.verdict_markdown(document))


class RunTests(unittest.TestCase):
    def test_outputs_resolving_into_models_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "tree" / "install" / "models" / "blob"
            models.mkdir(parents=True)
            (root / "results").symlink_to(models)
            with self.assertRaisesRegex(run.RunError, "install/models"):
                run.refuse_protected(root / "results" / "run1")
            home = root / "home"
            (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
            (root / "cache").symlink_to(home / ".cache" / "huggingface" / "hub")
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                with self.assertRaisesRegex(run.RunError, "huggingface"):
                    run.refuse_protected(root / "cache")
                with self.assertRaisesRegex(run.RunError, "install/models"):
                    run.main(
                        [
                            "--tree",
                            directory,
                            "--machine",
                            "m6",
                            "--model",
                            "m",
                            "--scenario",
                            "churn",
                            "--label",
                            "x",
                            "--out",
                            str(root / "results"),
                        ]
                    )
            self.assertEqual(
                run.refuse_protected(root / "elsewhere"),
                Path(os.path.realpath(root / "elsewhere")),
            )

    def test_gpu_utilization_reads_ioreg(self):
        line = (
            '  | "PerformanceStatistics" = {"In use system memory"=1,'
            '"Device Utilization %"=37,"Renderer Utilization %"=12}'
        )
        with mock.patch.object(run, "command_output", return_value=line):
            self.assertEqual(run.gpu_utilization(), 37)

    def test_a_port_that_answers_is_refused(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertTrue(run.port_answers(port))
            with tempfile.TemporaryDirectory() as directory:
                code = run.main(
                    [
                        "--tree",
                        directory,
                        "--machine",
                        "m6",
                        "--model",
                        "m",
                        "--scenario",
                        "churn",
                        "--label",
                        "busy",
                        "--out",
                        directory,
                        "--port",
                        str(port),
                    ]
                )
                self.assertEqual(code, 1)
                self.assertIn(
                    "already answers", (Path(directory) / "busy.out").read_text()
                )
        self.assertFalse(run.port_answers(port))


if __name__ == "__main__":
    unittest.main()
