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

    def test_serve1_samples_its_idle_status_and_fails_on_an_error(self):
        # serve1 records the /status each request waited idle for and one
        # after the last, so that G8 sees a restart or a Metal failure; a
        # request the server fails ends it as the server's failure.
        statuses = iter({"n": i} for i in range(10))

        class FakeStream:
            def __init__(self, port, body):
                self.sent, self.chunks, self.error_body = 0.0, [0.5, 0.6, 0.7], ""
                self.status = 500 if "Log s2048" in body else 200
                self.usage = {"completion_tokens": 3, "prompt_tokens": 300}

        def sampled(directory, label):
            return [
                json.loads(line)["status"]["n"]
                for line in (Path(directory) / f"{label}.status.jsonl").open()
            ]

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.multiple(
                clients,
                served_model=lambda port: "m",
                wait_idle=lambda port, settle: next(statuses),
                Stream=FakeStream,
            ),
        ):
            clients.run_serve1(1, directory, "ok", samples=2, sizes=(300,))
            self.assertEqual(sampled(directory, "ok"), [0, 1, 2])
            with self.assertRaisesRegex(clients.RequestFailed, "serve1 2048: 500"):
                clients.run_serve1(1, directory, "bad", samples=1, sizes=(300, 2048))
            self.assertEqual(sampled(directory, "bad"), [3, 4])


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

        # DESIGN §8.3: throughput PASSes when the lower bound clears -2 %
        # and FAILs once the upper bound is below 1.00; between them it is
        # INCONCLUSIVE.
        self.assertEqual(stats.verdict(estimate(1.00, 0.01), True, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(0.99, 0.005), True, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(0.985, 0.01), True, 0.02), stats.FAIL)
        self.assertEqual(
            stats.verdict(estimate(0.99, 0.03), True, 0.02), stats.INCONCLUSIVE
        )
        # Latency and TTFT: PASS with the upper bound at most +2 %, FAIL once
        # the lower is past 1.00, as G3 at 1.04 [1.01, 1.07].
        self.assertEqual(stats.verdict(estimate(1.00, 0.015), False, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(1.01, 0.005), False, 0.02), stats.PASS)
        self.assertEqual(stats.verdict(estimate(1.04, 0.03), False, 0.02), stats.FAIL)
        self.assertEqual(
            stats.verdict(estimate(1.01, 0.03), False, 0.02), stats.INCONCLUSIVE
        )
        self.assertEqual(stats.verdict(None, False, 0.02), stats.INCONCLUSIVE)
        # A tolerance of zero: equal runs cannot pass.
        self.assertEqual(
            stats.verdict(estimate(1.0, 0.02), True, 0.0), stats.INCONCLUSIVE
        )
        # The owner's trade (20:30): with a gain, more than 5 % fails.
        self.assertEqual(
            stats.verdict(estimate(0.97, 0.01), True, 0.05, 0.05), stats.PASS
        )
        self.assertEqual(
            stats.verdict(estimate(0.95, 0.02), True, 0.05, 0.05), stats.INCONCLUSIVE
        )
        self.assertEqual(
            stats.verdict(estimate(0.93, 0.01), True, 0.05, 0.05), stats.FAIL
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

    def test_a_restart_fails_g8_even_in_a_run_not_used(self):
        # The engine crashed once under churn and its supervision relaunched
        # it at once, resuming the requests: none failed, but
        # transport.restarts moved.
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        restarted = [
            status(0),
            status(450, transport__restarts=1),
            status(960, transport__restarts=1),
        ]
        churn_run(self.directory, "restarted", done, restarted)
        problems = matrix.check_run(
            self.directory, "restarted", 15, self.machine, self.gates
        )
        self.assertIn("G8: the engine restarted (transport.restarts 0 -> 1)", problems)
        # analyze: the run is not used, and still fails G8; a run left out
        # only for its surroundings is just listed.
        cell = {"name": "c", "machine": "m6", "model": "fn-q2", "scenario": "churn"}
        hot = [status(0), status(960, thermal_state="serious")]
        churn_run(self.directory, "hot", done, hot)
        runs = [
            matrix.Run(self.directory, label, "c", "cand", "A")
            for label in ("restarted", "hot")
        ]
        for run_ in runs:
            matrix.validate(run_, self.spec, cell)
            self.assertTrue(run_.invalid, run_.label)
        verdicts = matrix.churn_cell(cell, self.spec, runs, {}, PRICES)
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts],
            [("G4 G8 G9 restarted", stats.FAIL)],
        )

    def run_meta(self, label, **fields):
        model = self.spec["models"]["fn-q2"]["model"]
        (self.directory / f"{label}.meta.json").write_text(
            json.dumps({"label": label, "model": model, "complete": True, **fields})
        )

    def test_only_the_servers_own_failures_fail_g8(self):
        # s-fn-churn-18g on the final day: a stale server still answers on
        # the port, and run.py refuses the run. An M5 run is voided because
        # the laptop's owner was on a call, and its loop shows a 5.2 s tick.
        # Both are listed, to be repeated, and fail nothing. A run whose
        # server exited during the scenario fails G8.
        cell = {"name": "c", "machine": "m6", "model": "fn-q2", "scenario": "churn"}
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        refusal = "port 8611 already answers: a stale server would serve this run"
        (self.directory / "refused.out").write_text(f"FAIL {refusal}\n")
        self.run_meta("refused", complete=False, failure="port", error=refusal)
        churn_run(
            self.directory,
            "called",
            done,
            [status(0), status(960, loop__max_tick_ms=5200)],
        )
        self.run_meta("called")
        write_jsonl(
            self.directory / "called.idle.jsonl", [{"hid_idle_s": 3.0, "holders": []}]
        )
        churn_run(
            self.directory, "exited", done, [status(0), status(300)], complete=False
        )
        exit_error = "the server exited -9 during the scenario"
        self.run_meta("exited", complete=False, failure="server", error=exit_error)
        runs = [
            matrix.Run(self.directory, label, "c", "cand", "A")
            for label in ("refused", "called", "exited")
        ]
        for run_ in runs:
            matrix.validate(run_, self.spec, cell)
        verdicts = matrix.churn_cell(cell, self.spec, runs, {}, PRICES)
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts],
            [("G4 G8 G9 exited", stats.FAIL)],
        )
        self.assertIn(f"G8: {exit_error}", verdicts[0]["numbers"]["failures"])
        self.assertEqual(
            [(r.label, r.invalid, bool(r.voided)) for r in runs],
            [
                ("refused", [f"the run did not end with DONE: {refusal}"], True),
                ("called", ["the laptop was in use (HIDIdleTime 3.0 s)"], True),
                ("exited", [f"the run did not end with DONE: {exit_error}"], False),
            ],
        )

    def test_g8_judges_every_scenarios_candidate_runs(self):
        # m6-fn-ttft: cand's engine crashed during its N=2 level and its
        # supervision relaunched it; the rounds after still completed. The
        # run is not used for G5 and fails G8, as does one with a failed
        # round.
        cell = {
            "name": "t",
            "machine": "m6",
            "model": "fn-q2",
            "scenario": "ttft",
            "reference": "pass",
            "gate_n": [2],
            "recorded": {"pass": ["ref"]},
            "runs": {"cand": {"-": 2}},
        }

        def level(label, n, samples, failed_round=None):
            write_jsonl(
                self.directory / f"{label}-n{n}.events.jsonl",
                [{"kind": "start", "n": n, "rounds": 3}]
                + [
                    {
                        "kind": "round_failed",
                        "n": n,
                        "round": r,
                        "failures": ["500 engine_execution_failed"],
                    }
                    if r == failed_round
                    else {
                        "kind": "round",
                        "n": n,
                        "round": r,
                        "ttft_ms": [4000] * n,
                        "cached": [7968] * n,
                    }
                    for r in range(3)
                ],
            )
            write_jsonl(self.directory / f"{label}-n{n}.status.jsonl", samples)

        for label in ("ref", "t-cand-1", "t-cand-2"):
            (self.directory / f"{label}.out").write_text("DONE\n")
        level("ref", 2, [status(0), status(30)])
        # Each level's server starts afresh at 0 restarts.
        level(
            "t-cand-1",
            2,
            [
                status(0),
                status(10, transport__restarts=1),
                status(30, transport__restarts=1),
            ],
        )
        level("t-cand-1", 4, [status(0), status(30)])
        level("t-cand-2", 2, [status(0), status(30)], failed_round=1)
        verdicts, _, unused = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts],
            [
                ("G5 N=2", stats.INCONCLUSIVE),
                ("G8 t-cand-1", stats.FAIL),
                ("G8 t-cand-2", stats.FAIL),
            ],
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in unused],
            [
                ("t-cand-1", ["the engine restarted (0 -> 1)"]),
                ("t-cand-2", ["requests the server failed: 1"]),
            ],
        )
        # p-serve1: cand's 14K request came back 500 and ended its run; its
        # other server exited before it was ready, leaving only run.py's
        # records. Both fail G8, and the set is not judged.
        cell = {
            "name": "s1",
            "machine": "m5",
            "model": "fn-q2",
            "scenario": "serve1",
            "abba": {"reference": "pass", "sets": 1},
        }
        for label in ("s1-pass-1", "s1-pass-2", "s1-cand-1"):
            write_jsonl(
                self.directory / f"{label}.events.jsonl",
                [
                    {
                        "kind": "request",
                        "size": 300,
                        "sample": 0,
                        "ttft_ms": 1000,
                        "decode_ms_per_token": 20.0,
                    }
                ],
            )
            write_jsonl(
                self.directory / f"{label}.status.jsonl", [status(0), status(10)]
            )
        for label in ("s1-pass-1", "s1-pass-2"):
            (self.directory / f"{label}.out").write_text("DONE\n")
            self.run_meta(label)
        failed = "serve1 14336: 500 engine_execution_failed"
        (self.directory / "s1-cand-1.out").write_text(f"FAIL {failed}\n")
        self.run_meta("s1-cand-1", complete=False, failure="request", error=failed)
        never = "the server exited 1 before it was ready"
        (self.directory / "s1-cand-2.out").write_text(f"FAIL {never}\n")
        self.run_meta("s1-cand-2", complete=False, failure="server", error=never)
        verdicts, unused = matrix.abba_cell(cell, self.spec, [self.directory])
        self.assertEqual(
            [(v["rule"], v["verdict"], v["numbers"]) for v in verdicts],
            [
                ("G8 s1-cand-1", stats.FAIL, [f"G8: {failed}"]),
                ("G8 s1-cand-2", stats.FAIL, [f"G8: {never}"]),
                ("G7 set 1", stats.INCONCLUSIVE, "runs not used: s1-cand-1, s1-cand-2"),
            ],
        )
        self.assertEqual([u["label"] for u in unused], ["s1-cand-1", "s1-cand-2"])

    def test_work_is_read_at_the_end_of_the_cells_load_window(self):
        # mini-35b-churn runs 10 minutes. pass's recorded series stopped at
        # 604.94 s; cand's run.py series runs on through the drain, where it
        # keeps counting output. cand did 10 % less work in the window.
        spec = self.spec
        cell = next(c for c in spec["cells"] if c["name"] == "mini-35b-churn")
        self.assertEqual(matrix.work_second(cell), 600)
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        churn_run(
            self.directory,
            "mini35PSa",
            done,
            [
                status(0),
                status(600.5, rows=585 * 1000, disk__read_bytes=1000),
                status(604.94, rows=585 * 1003, disk__read_bytes=1003),
            ],
        )
        label = "mini-35b-churn-cand-A1"
        churn_run(
            self.directory,
            label,
            done,
            [
                status(0),
                status(599, rows=585 * 900, disk__read_bytes=900),
                status(900, rows=585 * 1350, disk__read_bytes=1350),
                status(1020, rows=585 * 1500, disk__read_bytes=1500),
            ],
        )
        (self.directory / f"{label}.meta.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "cell": "mini-35b-churn",
                    "tree_name": "cand",
                    "order": [2, 0, 1],
                }
            )
        )
        runs = [
            r
            for r in matrix.load_runs(spec, [self.directory])
            if r.cell == cell["name"]
        ]
        self.assertEqual(sorted(r.tree for r in runs), ["cand", "pass"])
        work = next(
            v
            for v in matrix.point_gate(cell, runs, PRICES, self.gates, True)
            if v["rule"] == "G10 work vs pass"
        )
        self.assertEqual((work["verdict"], work["numbers"]["point"]), (stats.FAIL, 0.9))
        candidate = next(r for r in runs if r.tree == "cand")
        self.assertEqual(matrix.ssd_counters(candidate, 900)["disk_read_per_work"], 1.0)

    def test_prefill_hole_and_ssd_busy_shares(self):
        # Borrowed item 8: host steps 100 + 700 + 200 ms of 10 s of prefill
        # wall; 40 GB read at the seeded 8 GB/s is 5 s of it.
        steps = {
            f"{part}_ms": {"total": total}
            for part, total in (("pickup", 100), ("work", 700), ("resume", 200))
        }
        streamed = status(
            900,
            streamed_experts__prefill={"read_bytes": 40e9, "host_steps": steps},
            metrics__prefill_wall_ms=10000,
            scheduler__estimate__prefill={
                "terms": [
                    {"quantity": "units", "seed_ms_per_unit": 25},
                    {"quantity": "weight_read_bytes", "seed_ms_per_unit": 1 / 8e6},
                ]
            },
        )
        readouts = matrix.health(
            churn_run(self.directory, "shares", [], [status(0), streamed]),
            self.machine,
            self.gates,
        )["readouts"]
        self.assertAlmostEqual(readouts["prefill_hole_share"], 0.1)
        self.assertAlmostEqual(readouts["ssd_busy_share"], 0.5)
        resident = matrix.health(
            churn_run(self.directory, "resident", [], [status(0), status(900)]),
            self.machine,
            self.gates,
        )["readouts"]
        self.assertEqual(
            (resident["prefill_hole_share"], resident["ssd_busy_share"]), (None, None)
        )

    def test_a_process_this_run_did_not_start_voids_it(self):
        # base.md §2.7 with the gates' floor: the run.py smoke's start, with
        # mediaanalysisd at 117.8 % and nothing of the run's in the top five,
        # is void; WindowServer's and sysmond's steady 10-12 % are not.
        cell = {"name": "c", "machine": "m5", "model": "fn-q2", "scenario": "serve1"}
        model = self.spec["models"]["fn-q2"]["model"]

        def validated(label, rows, holders=()):
            write_jsonl(
                self.directory / f"{label}.idle.jsonl",
                [{"hid_idle_s": 300.0, "holders": list(holders)}],
            )
            (self.directory / f"{label}.out").write_text("DONE\n")
            (self.directory / f"{label}.meta.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "model": model,
                        "pids": [700, 701],
                        "ps": [{"rows": rows}],
                    }
                )
            )
            run_ = matrix.Run(self.directory, label)
            matrix.validate(run_, self.spec, cell)
            return run_.invalid

        busy = [
            [55828, 117.8, 0.5, "mediaanalysisd", False],
            [68643, 11.2, 1.0, ".venv/bin/python", False],
            [145, 9.8, 0.1, "WindowServer", False],
            [70493, 3.0, 0.1, ".venv/bin/python", False],
            [1162, 2.9, 0.0, "WallpaperAerialsExtension", False],
        ]
        self.assertEqual(
            validated("busy", busy),
            [
                "a process this run did not start in the top five by CPU: "
                "mediaanalysisd at 117.8 %"
            ],
        )
        quiet = [
            [145, 11.9, 0.1, "WindowServer", False],
            [322, 10.6, 0.0, "sysmond", False],
            [700, 8.0, 4.4, "build/splash", True],
            [701, 3.0, 0.1, ".venv/bin/python", True],
            [57229, 7.1, 0.1, "Activity Monitor", False],
        ]
        self.assertEqual(validated("quiet", quiet), [])
        # A sleep assertion counts by pid: this run's engine may hold one,
        # another python may not.
        ours = "pid 700(splash): [0x0001] 00:05:00 PreventUserIdleSystemSleep named: x"
        theirs = "pid 4242(python3.12): [0x0002] 00:05:00 PreventUserIdleSystemSleep named: y"
        self.assertEqual(validated("held", quiet, [ours]), [])
        self.assertEqual(
            validated("other", quiet, [ours, theirs]),
            [f"another process held a sleep assertion: {theirs}"],
        )

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
            (self.directory / f"{label}.out").write_text("DONE\n")

        # pass's N=2 rounds (amend/V.md §0.4): median 4622 ms, spread 4622/4399 - 1.
        ttft_run("ref", [6976, 4622, 4399])
        cell = {
            "name": "t",
            "reference": "pass",
            "gate_n": [2],
            "recorded": {"pass": ["ref"], "cand": ["fast"]},
        }
        ttft_run("fast", [4700, 4800, 9000])
        verdicts, _, unused = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual(verdicts[0]["verdict"], stats.PASS, verdicts)
        self.assertEqual(unused, [])
        self.assertAlmostEqual(
            verdicts[0]["numbers"]["limit_ms"], 4622 * 4622 / 4399, places=0
        )
        ttft_run("slow", [5000, 5100, 5200])
        cell["recorded"]["cand"] = ["slow"]
        self.assertEqual(
            matrix.ttft_cell(cell, self.spec, [self.directory])[0][0]["verdict"],
            stats.FAIL,
        )

    def test_a_ttft_level_counts_only_with_every_round_complete(self):
        # m6-fn-ttft: cand's N=2 records hold one of its three rounds; the
        # one left is no G5 median. A run without DONE is not used at all.
        cell = {
            "name": "t",
            "machine": "m6",
            "model": "fn-q2",
            "scenario": "ttft",
            "reference": "pass",
            "gate_n": [2],
            "recorded": {"pass": ["ref"]},
            "runs": {"cand": {"-": 1}},
        }

        def level(label, n, rounds, done=True):
            write_jsonl(
                self.directory / f"{label}-n{n}.events.jsonl",
                [{"kind": "start", "n": n, "rounds": 3}]
                + [
                    {
                        "kind": "round",
                        "n": n,
                        "round": r,
                        "ttft_ms": [4000, 4500],
                        "cached": [7968, 7968],
                    }
                    for r in sorted(rounds)
                ],
            )
            (self.directory / f"{label}.out").write_text("DONE\n" if done else "")

        level("ref", 2, {0, 1, 2})
        level("t-cand-1", 2, {1})
        verdicts, _, unused = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual(verdicts[0]["verdict"], stats.INCONCLUSIVE, verdicts)
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in unused],
            [("t-cand-1", ["N=2: 1 of 3 rounds completed their 2 requests"])],
        )
        level("t-cand-1", 2, {0, 1, 2}, done=False)
        verdicts, _, unused = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual(verdicts[0]["verdict"], stats.INCONCLUSIVE, verdicts)
        self.assertEqual(unused[0]["reasons"], ["the run did not end with DONE"])
        level("t-cand-1", 2, {0, 1, 2})
        verdicts, _, unused = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual((verdicts[0]["verdict"], unused), (stats.PASS, []))

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

    def test_the_freeze_gain_is_measured_on_the_trailed_reference(self):
        # p-fn-churn-18g on the M5: cand trails r150 by 4 % in work with the
        # same TTFT and latency and holds G4. That is a gain only where r150's
        # own runs froze (a decode gap at or past the M5's 8.6 s floor), and
        # completion latency never takes the trade tolerance.
        def make(label, tree, order, work, gap_ms=None):
            done = [
                {
                    "tag": f"w{w}-r1",
                    "start": 0,
                    "t": 200,
                    "ttft": 100,
                    "decode_tps": 30,
                    "chunks_ms": [0, 100, 200],
                }
                for w in range(3)
            ]
            fields = (
                {}
                if gap_ms is None
                else {"kv_spill__history_stream__decode_gap_ms": {"max": gap_ms}}
            )
            r = churn_run(
                self.directory,
                label,
                done,
                [status(900, rows=int(work * 1000), **fields)],
            )
            r.tree, r.order = tree, order
            return r

        cell = {
            "name": "p",
            "machine": "m5",
            "release": {"throughput": "r150"},
            "health": ["cand"],
        }
        prices = {"R": 1000, "D": 75}
        cand = [make(f"c{o}{i}", "cand", o, 960 + i) for o in "AB" for i in range(2)]

        def verdicts(gap_ms):
            refs = [
                make(f"r{o}{i}", "r150", o, 1000 + i, gap_ms)
                for o in "AB"
                for i in range(2)
            ]
            found = matrix.churn_cell(cell, self.spec, refs + cand, {}, prices)
            return {
                v["rule"].split(" cand")[0]: v for v in found if " vs " in v["rule"]
            }

        smooth = verdicts(4600)
        self.assertNotEqual(smooth["G1"]["verdict"], stats.PASS, smooth["G1"])
        self.assertFalse(smooth["G1"]["numbers"]["freeze"]["reference_froze"])
        froze = verdicts(424800)
        self.assertEqual(froze["G1"]["verdict"], stats.PASS, froze["G1"])
        self.assertTrue(froze["G1"]["trade"])
        self.assertEqual(froze["G1"]["numbers"]["freeze"]["reference_gap_s"], 424.8)
        self.assertEqual(froze["G2"]["numbers"]["tolerance"], 0.0)

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
        cell = {
            "name": "s1",
            "machine": "m5",
            "model": "fn-iq3s",
            "scenario": "serve1",
            "abba": {"reference": "pass", "sets": 1},
        }
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
            (self.directory / f"s1-{tree}-{repeat}.out").write_text("DONE\n")
        verdicts, unused = matrix.abba_cell(cell, self.spec, [self.directory])
        self.assertEqual(
            {v["rule"] for v in verdicts},
            {
                "G8 s1-cand-1",
                "G8 s1-cand-2",
                "G7 serve1_300_ttft_ms set 1",
                "G7 serve1_300_decode_ms_per_token set 1",
            },
        )
        self.assertTrue(all(v["verdict"] == stats.PASS for v in verdicts), verdicts)
        self.assertEqual(unused, [])
        # p-serve1 on the owner's laptop: typing during a cand run voids it,
        # its set is not judged, and the voided run is not judged on G8.
        write_jsonl(
            self.directory / "s1-cand-2.idle.jsonl",
            [{"hid_idle_s": 300.0, "holders": []}, {"hid_idle_s": 2.5, "holders": []}],
        )
        verdicts, unused = matrix.abba_cell(cell, self.spec, [self.directory])
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts],
            [("G8 s1-cand-1", stats.PASS), ("G7 set 1", stats.INCONCLUSIVE)],
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in unused],
            [("s1-cand-2", ["the laptop was in use (HIDIdleTime 2.5 s)"])],
        )


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

    def test_snapshots_are_taken_while_the_server_serves(self):
        # The process snapshots that void a run see its server and engine
        # among the run's own: none before it is ready or after it stopped.
        events = []

        class FakeServer:
            def __init__(self, tree, root, model, port, server_args, log):
                self.process = mock.Mock(pid=4242, **{"poll.return_value": None})

            def wait_ready(self):
                events.append("ready")
                return 1.0

            def engine_pids(self):
                return {4242, 4243}

            def stop(self):
                events.append("stop")
                return "stopped by SIGINT, exit 0"

        def snapshot(ours):
            events.append(frozenset(ours))
            return {"at": 0.0, "rows": [], "therm": ""}

        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            (tree / "build").mkdir()
            (tree / "build" / "splash").write_bytes(b"splash")
            with (
                mock.patch.multiple(
                    run,
                    Server=FakeServer,
                    ps_snapshot=snapshot,
                    port_answers=lambda port: False,
                    tree_build_id=lambda tree: "src-1",
                    host_memory=lambda: {},
                    check_identity=lambda port, server, build_id: {
                        "pid": 4242,
                        "build_id": build_id,
                        "thermal_state": "nominal",
                    },
                ),
                mock.patch.object(run.clients, "run_serve1"),
            ):
                code = run.main(
                    [
                        "--tree",
                        directory,
                        "--machine",
                        "studio",
                        "--model",
                        "m",
                        "--model-root",
                        directory,
                        "--scenario",
                        "serve1",
                        "--label",
                        "s",
                        "--out",
                        f"{directory}/out",
                    ]
                )
            meta = json.loads((tree / "out" / "s.meta.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(events[0], "ready")
        self.assertEqual(events[-1], "stop")
        snapshots = events[1:-1]
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all({4242, 4243} <= ours for ours in snapshots), snapshots)
        self.assertTrue({4242, 4243} <= set(meta["pids"]))

    def test_the_servers_own_failures_are_recorded_as_such(self):
        # A request the server failed, or a server gone by the end of the
        # scenario, ends the run as the server's failure, which fails G8;
        # run.py's refusals are recorded by their own kinds.
        exited = {}

        class FakeServer:
            def __init__(self, tree, root, model, port, server_args, log):
                self.process = mock.Mock(pid=4242, returncode=-9)
                self.process.poll = lambda: exited.get("code")

            def wait_ready(self):
                return 1.0

            def engine_pids(self):
                return {4242}

            def stop(self):
                return "exited -9"

        def recorded(label, serve1):
            with tempfile.TemporaryDirectory() as directory:
                tree = Path(directory)
                (tree / "build").mkdir()
                (tree / "build" / "splash").write_bytes(b"splash")
                with (
                    mock.patch.multiple(
                        run,
                        Server=FakeServer,
                        ps_snapshot=lambda ours: {"at": 0.0, "rows": [], "therm": ""},
                        port_answers=lambda port: False,
                        tree_build_id=lambda tree: "src-1",
                        host_memory=lambda: {},
                        check_identity=lambda port, server, build_id: {
                            "pid": 4242,
                            "build_id": build_id,
                            "thermal_state": "nominal",
                        },
                    ),
                    mock.patch.object(run.clients, "run_serve1", side_effect=serve1),
                ):
                    code = run.main(
                        [
                            "--tree",
                            directory,
                            "--machine",
                            "studio",
                            "--model",
                            "m",
                            "--model-root",
                            directory,
                            "--scenario",
                            "serve1",
                            "--label",
                            label,
                            "--out",
                            f"{directory}/out",
                        ]
                    )
                meta = json.loads((tree / "out" / f"{label}.meta.json").read_text())
            return code, meta.get("failure"), meta.get("error")

        def failed_request(*args, **kwargs):
            raise clients.RequestFailed("serve1 14336: 500 engine_execution_failed")

        self.assertEqual(
            recorded("request", failed_request),
            (1, "request", "serve1 14336: 500 engine_execution_failed"),
        )

        def server_gone(*args, **kwargs):
            exited["code"] = -9

        self.assertEqual(
            recorded("gone", server_gone),
            (1, "server", "the server exited -9 during the scenario"),
        )

        def connection_refused(*args, **kwargs):
            raise ConnectionRefusedError(61, "Connection refused")

        code, failure, error = recorded("refused", connection_refused)
        self.assertEqual((code, failure), (1, "server"))
        self.assertTrue(error.startswith("the server exited -9 during the scenario: "))
        exited.clear()
        with self.assertRaises(ConnectionRefusedError):
            recorded("alive", connection_refused)

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
                meta = json.loads((Path(directory) / "busy.meta.json").read_text())
                self.assertEqual((meta["complete"], meta["failure"]), (False, "port"))
        self.assertFalse(run.port_answers(port))


if __name__ == "__main__":
    unittest.main()
