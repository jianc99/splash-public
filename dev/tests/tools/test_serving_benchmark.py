import hashlib
import http.client
import http.server
import io
import json
import math
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
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
            def __init__(self, port, body, timeout):
                self.sent, self.chunks, self.error_body = 0.0, [0.5, 0.6, 0.7], ""
                self.status = 500 if "Log s2048" in body else 200
                self.ok = self.status == 200
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

    def test_prompts_take_the_tokens_their_sizes_name(self):
        # The prompt_tokens the servers recorded for churn_doc prompts (Qwen
        # tokenizers, which split digits one by one): serve1's size-300
        # prompt of sample 0, 75 words at the old 4 tokens a word, came back
        # as 662 tokens (smoke-s-serve1-pass-1-132500 and
        # smoke-a-mini-serve1-cand-1-201707), its size-2048 one, 512 words,
        # as 4,190; churn's first rounds of 2,000 and 7,300 words as
        # 16,205-16,259 (smoke-a-fn-churn-A1) and 59,043-59,144
        # (smoke-a-s-fn-churn-18g-cand-A1-193918). A prompt of w words takes
        # about 662 + (w - 75) x 8.09 tokens. Each serve1 size and s-warm's
        # depths then come within 2 %, and s-warm's longest prompt, its
        # 134K prefix and a suffix, fits its --max-context.
        def tokens(words):
            return 662 + (59141 - 662) / (7300 - 75) * (words - 75)

        def words(messages):
            return len(messages[-1]["content"].split("\n")[1].split())

        for size in clients.SERVE1_SIZES + (32768,):
            prompt = tokens(words(clients.serve1_messages(size, 0)))
            self.assertAlmostEqual(prompt, size, delta=0.02 * size)
        cell = next(c for c in matrix.load_spec(SPEC)["cells"] if c["name"] == "s-warm")
        context = int(cell["server_args"].split()[-1])
        for depth in cell["sizes"]:
            prefix = clients.warm_prefix(depth, 1)
            self.assertAlmostEqual(tokens(words(prefix)), depth, delta=0.02 * depth)
            suffix = clients.warm_suffix(prefix, depth, 1, 9)[-1:]
            self.assertLess(tokens(words(prefix) + words(suffix)), context)


# A request that fails after its stream began: server.py's _sse_error.
FAILED_EVENT = {
    "error": {
        "message": "internal server error",
        "type": "server_error",
        "code": "internal_server_error",
    }
}


def content_event(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


KEEPALIVE = object()


def keepalives_forever():
    while True:
        yield KEEPALIVE


def finished(target, seconds=30):
    """target()'s outcome, ("returned", value) or ("raised", error), or
    ("running", None) if it has not ended within `seconds`."""
    outcome = ["running", None]

    def call():
        try:
            outcome[:] = ["returned", target()]
        except Exception as error:
            outcome[:] = ["raised", error]

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(seconds)
    return tuple(outcome)


class StreamingServer:
    """A server on 127.0.0.1 streaming each chat completion, status 200, as
    server.py does: the role chunk, then the events answer(body) returns
    (KEEPALIVE: server.py's keepalive, 0.05 s after the last write), then
    [DONE]; /status idle."""

    def __init__(self, answer):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                models = {"data": [{"id": "m"}]}
                document = models if self.path == "/v1/models" else {}
                data = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                role = {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
                try:
                    self.wfile.write(f"data: {json.dumps(role)}\n\n".encode())
                    for event in answer(body):
                        if event is KEEPALIVE:
                            time.sleep(0.05)
                            self.wfile.write(b": splash-keepalive\n\n")
                        else:
                            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client ended the request

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class ErrorEventTests(unittest.TestCase):
    """A request the server fails after the role chunk went out answers 200
    and ends with an error event: every scenario records it as failed."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def events(self, label):
        path = Path(self.directory) / f"{label}.events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_the_stream_records_the_error_event(self):
        with StreamingServer(lambda body: [content_event("1"), FAILED_EVENT]) as fake:
            stream = clients.Stream(fake.port, "{}")
        self.assertEqual((stream.status, stream.ok), (200, False))
        self.assertEqual(json.loads(stream.error_body), FAILED_EVENT["error"])

    def test_churn_records_an_http_error(self):
        # m6-fn-churn cand: the engine fails each request during its
        # prefill, after the role chunk.
        def failing(body):
            time.sleep(0.1)
            return [FAILED_EVENT]

        with (
            StreamingServer(failing) as fake,
            mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
        ):
            clients.run_churn(
                fake.port, self.directory, "c", 0.01, [0, 1, 2], poll_seconds=0.05
            )
        kinds = [e["kind"] for e in self.events("c")]
        self.assertNotIn("done", kinds)
        errors = [e for e in self.events("c") if e["kind"] == "http_error"]
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(
            all(
                e["code"] == 200 and "internal_server_error" in e["body"]
                for e in errors
            )
        )
        self.assertEqual(len(matrix.Run(Path(self.directory), "c").errors), len(errors))

    def test_ttft_fails_the_round(self):
        # m6-fn-ttft cand at N=2: the engine refuses one lane's memory claim
        # in round 0 after the role chunk went out.
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 1900},
            },
        }

        def answer(body):
            if "Round 0." in body and "Conversation 2.1." in body:
                return [FAILED_EVENT]
            return [content_event("OK"), usage]

        with StreamingServer(answer) as fake:
            summary = clients.run_ttft(
                fake.port,
                self.directory,
                "t-n2",
                2,
                rounds=1,
                settle=0,
                poll_seconds=0.05,
                text=synthetic_corpus(),
            )
        self.assertEqual(summary["rounds"], 0)
        (failed,) = [e for e in self.events("t-n2") if e["kind"] == "round_failed"]
        self.assertEqual(len(failed["failures"]), 1)
        self.assertTrue(failed["failures"][0].startswith("200 "))
        self.assertIn("internal_server_error", failed["failures"][0])

    def test_serve1_and_warm_end_on_it(self):
        # p-serve1 cand: the 300-token request fails after its first token;
        # s-warm cand: a suffix request fails after its role chunk.
        with StreamingServer(lambda body: [content_event("1"), FAILED_EVENT]) as fake:
            with self.assertRaisesRegex(
                clients.RequestFailed, "serve1 300: 200 .*internal_server_error"
            ):
                clients.run_serve1(
                    fake.port, self.directory, "s", samples=1, settle=0, sizes=(300,)
                )

        def answer(body):
            return [FAILED_EVENT] if "Note 0:" in body else [content_event("OK")]

        with StreamingServer(answer) as fake:
            with self.assertRaisesRegex(
                clients.RequestFailed, "warm 300 suffix 0: 200 .*internal_server_error"
            ):
                clients.run_warm(fake.port, self.directory, "w", [300], 0, settle=0)


class DeadlineTests(unittest.TestCase):
    """server.py keeps every stream alive with a keepalive every 2 s and sets
    no deadline of its own, so a request a scheduler or ledger bug starves
    ends at its time as the server's failure instead of holding the run, and
    its machine's GPU lock, forever."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def events(self, label):
        path = Path(self.directory) / f"{label}.events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_keepalives_do_not_hold_a_request_open(self):
        with StreamingServer(lambda body: keepalives_forever()) as fake:
            outcome = finished(lambda: clients.Stream(fake.port, "{}", timeout=0.5), 10)
        self.assertEqual(
            outcome[0], "raised", "keepalives held a request open past its time"
        )
        self.assertIsInstance(outcome[1], clients.Deadline)

    def test_churn_fails_a_request_the_server_keeps_open(self):
        # m6-fn-churn cand: w1's first request stays queued while w0 and w2
        # keep working, and the server keeps its stream alive.
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 2},
        }

        def answer(body):
            if "Log 1-" in body:
                yield from keepalives_forever()
            time.sleep(0.2)
            yield from (content_event("1"), content_event("2"), usage)

        with (
            StreamingServer(answer) as fake,
            mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
        ):
            outcome = finished(
                lambda: clients.run_churn(
                    fake.port,
                    self.directory,
                    "c",
                    0.01,
                    [0, 1, 2],
                    poll_seconds=0.05,
                    timeout=1,
                )
            )
        self.assertEqual(outcome[0], "raised", outcome)
        self.assertIsInstance(outcome[1], clients.Deadline)
        self.assertIn("w1-r1", str(outcome[1]))
        events = self.events("c")
        (expired,) = [e for e in events if e["kind"] == "deadline"]
        self.assertEqual((expired["tag"], expired["chunks"]), ("w1-r1", 0))
        (end,) = [e for e in events if e["kind"] == "end"]
        self.assertEqual(end["unanswered"], ["w1-r1"])
        done = {e["tag"].split("-")[0] for e in events if e["kind"] == "done"}
        self.assertEqual(done, {"w0", "w2"})

    def test_a_sender_that_fails_before_the_barrier_releases_the_others(self):
        # m6-fn-ttft cand at N=2: one sender of a round cannot connect; the
        # one that did does not wait at the barrier for it forever.
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 1},
        }
        connect, lock, refused = (
            http.client.HTTPConnection.connect,
            threading.Lock(),
            [],
        )

        def refuse_one_sender(connection):
            with lock:
                refuse = (
                    threading.current_thread().name.endswith("(send)") and not refused
                )
                if refuse:
                    refused.append(threading.current_thread().name)
            if refuse:
                raise ConnectionRefusedError(61, "Connection refused")
            return connect(connection)

        with (
            StreamingServer(lambda body: [content_event("OK"), usage]) as fake,
            mock.patch.object(http.client.HTTPConnection, "connect", refuse_one_sender),
        ):
            outcome = finished(
                lambda: clients.run_ttft(
                    fake.port,
                    self.directory,
                    "t-n2",
                    2,
                    rounds=1,
                    settle=0,
                    poll_seconds=0.05,
                    text=synthetic_corpus(),
                )
            )
        self.assertEqual(outcome[0], "returned", outcome)
        self.assertEqual((len(refused), outcome[1]["rounds"]), (1, 0))
        (failed,) = [e for e in self.events("t-n2") if e["kind"] == "round_failed"]
        self.assertEqual(len(failed["failures"]), 2)


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

    def test_g4b_takes_one_gap_per_token_event(self):
        # DFlash2 commits about eight tokens a lane per decode command, and
        # the server sends each as its own chunk, within a millisecond. G4b's
        # p99 is over the gaps between token events, as decode_gap_ms counts
        # them: 9 of the 297 here (3 %) are 3 s stalls, past the limit plus
        # p50 (1000 ms) and under G4c's 7.3 s. Over every chunk gap, 7 in 8
        # of them 0-1 ms, the stalls fall past the p99.
        samples = [self.schema7(0), self.schema7(900)]
        done = []
        for w in range(3):
            chunks, start = [], 0
            for event in range(100):
                chunks += [start + piece // 4 for piece in range(8)]
                start += 3000 if event % 33 == 32 else 100
            done.append(
                {
                    "tag": f"w{w}-r1",
                    "start": 0,
                    "t": 400,
                    "ttft": 5,
                    "decode_tps": 30,
                    "chunks_ms": chunks,
                }
            )
        facts = matrix.health(
            churn_run(self.directory, "bursts", done, samples),
            self.machine,
            self.gates,
        )
        self.assertEqual([f[:4] for f in facts["failures"]], ["G4b:"])
        self.assertAlmostEqual(facts["client_gap_p99_s"], 2.999)

    def test_g4b_allows_the_runs_own_median_client_gap(self):
        # G4b's allowance, the typical gap with no prefill unit between two
        # tokens, is the median of the run's own client gaps between token
        # events. metrics.decode_gap_ms.p50 is a log2 bucket's upper edge
        # (runtime/model/LatencyHistogram.hpp), up to twice the gap it
        # stands for: smoke-a-fn-churn-A1's read 524.288. Here a token event
        # comes every 100 ms and 5 of the 297 gaps (1.7 %) are 1.2 s, over
        # the 960 ms limit + the 100 ms median, under the limit + the 524 ms
        # edge.
        samples = [
            self.schema7(0),
            self.schema7(900, metrics__decode_gap_ms={"p50": 524.288, "max": 1200.0}),
        ]
        done = [
            {
                "tag": f"w{w}-r1",
                "start": 0,
                "t": 400,
                "ttft": 5,
                "decode_tps": 30,
                "chunks_ms": [
                    100 * event + 1100 * ((event + w) // 50) for event in range(100)
                ],
            }
            for w in range(3)
        ]
        facts = matrix.health(
            churn_run(self.directory, "edge", done, samples),
            self.machine,
            self.gates,
        )
        self.assertEqual(
            facts["failures"], ["G4b: client gap p99 1200 ms over 1060 ms"]
        )
        self.assertAlmostEqual(facts["client_gap_p50_s"], 0.1)

    def test_g4a_reads_no_quantile_above_the_largest_unit(self):
        # LatencyHistogram reports a quantile as its log2 bucket's upper edge,
        # never clamped to the largest value it holds: smoke-a-27b-churn-A1's
        # decode_gap_ms p95 read 1048.576 with max 609.03. A contended p95
        # read as 1048.576 whose largest unit took 900 ms is within the
        # 960 ms limit.
        samples = [
            self.schema7(0),
            self.schema7(
                900, scheduler__contended_prefill_ms={"p95": 1048.576, "max": 900.0}
            ),
        ]
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
            churn_run(self.directory, "edge", smooth, samples),
            self.machine,
            self.gates,
        )
        self.assertEqual(facts["failures"], [])
        self.assertAlmostEqual(facts["g4"]["contended_p95_ms"], 900.0)

    def test_an_ample_run_starts_at_the_cap_on_a_host_without_pressure(self):
        # s-fn-churn-ample: ntie's recorded sNFa reads warning at Ready, right
        # after the load, in the engine's band state and in macOS's own level,
        # its experts at their cap: it counts. A cand run whose host was at
        # macOS warning before run.py started the server, or whose experts
        # were below their cap at Ready, is voided.
        cell = next(c for c in self.spec["cells"] if c["name"] == "s-fn-churn-ample")
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        voided = {}
        for label, chunks, level in (
            ("sNFa", 4, None),
            ("s-fn-churn-ample-cand-A1", 4, 2),
            ("s-fn-churn-ample-cand-B1", 3, 1),
        ):
            ready = status(
                0,
                memory_pressure="warning",
                memory_governor__system_pressure="warning",
                streamed_experts__chunks=chunks,
                streamed_experts__cap_chunks=4,
            )
            churn_run(self.directory, label, done, [ready, status(960)])
            if level:
                self.run_meta(label, host_at_start={"pressure_level": level})
            run_ = matrix.Run(self.directory, label)
            matrix.validate(run_, self.spec, cell)
            voided[label] = run_.voided
        self.assertEqual(
            voided,
            {
                "sNFa": [],
                "s-fn-churn-ample-cand-A1": [
                    "ample start: macOS's memory pressure level was 2 before the run"
                ],
                "s-fn-churn-ample-cand-B1": [
                    "ample start: the experts were not at their cap at Ready"
                ],
            },
        )

    def with_cand_build(self, spec, build):
        trees = spec["trees"]
        return {
            **spec,
            "trees": {**trees, "cand": {**trees["cand"], "build_id": build}},
        }

    def test_analyze_judges_one_run_of_the_cand_build_under_test(self):
        # m6-fn-churn after a fix: cand's A1 measured the build before it,
        # A3 reran it on the build under test (a rerun takes the next repeat
        # number), and a copy of A3 sits in a second results directory. Only
        # A3 is judged, once; A1 and the copy are listed. B1, on the build
        # under test, was killed before it wrote its meta.json, after its
        # engine restarted: its .out names its build, and it fails G8.
        spec = matrix.load_spec(SPEC)
        cell = next(c for c in spec["cells"] if c["name"] == "m6-fn-churn")
        spec = {**self.with_cand_build(spec, "src-new"), "cells": [cell]}
        copy = self.directory / "copy"
        copy.mkdir()
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        for directory, label, build in (
            (self.directory, "m6-fn-churn-cand-A1", "src-old"),
            (self.directory, "m6-fn-churn-cand-A3", "src-new"),
            (copy, "m6-fn-churn-cand-A3", "src-new"),
        ):
            churn_run(directory, label, done, [self.schema7(0), self.schema7(960)])
            (directory / f"{label}.meta.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "cell": cell["name"],
                        "tree_name": "cand",
                        "order": [2, 0, 1],
                        "model": spec["models"]["fn-q2"]["model"],
                        "complete": True,
                        "build_id": build,
                    }
                )
            )
        killed = "m6-fn-churn-cand-B1"
        churn_run(
            self.directory,
            killed,
            done[:1],
            [status(0), status(300, transport__restarts=1)],
            complete=False,
        )
        (self.directory / f"{killed}.out").write_text(
            "root /models/fn tree /trees/cand binary 0123456789ab build src-new\n"
        )
        document = matrix.analyze(spec, [self.directory, copy])
        self.assertEqual(
            [v["rule"] for v in document["verdicts"] if v["rule"].startswith("G4")],
            ["G4 G8 G9 m6-fn-churn-cand-A3"],
        )
        self.assertIn(
            (f"G8 {killed}", stats.FAIL),
            [(v["rule"], v["verdict"]) for v in document["verdicts"]],
        )
        listed = [(u["label"], u["reasons"]) for u in document["invalid"]]
        self.assertIn(
            (
                "m6-fn-churn-cand-A1",
                ["build src-old, not the build under test src-new"],
            ),
            listed,
        )
        self.assertIn(
            ("m6-fn-churn-cand-A3", [f"also in {self.directory}, whose run is used"]),
            listed,
        )
        self.assertNotIn(("m6-fn-churn-cand-A1", ["not run"]), listed)

    def test_a_ttft_rerun_of_the_cand_build_under_test_is_judged(self):
        # m6-fn-ttft: cand-1 measured the build before a fix, and its rerun
        # took the next repeat number. G5 judges cand-2, slower than pass,
        # and lists cand-1.
        spec = self.with_cand_build(self.spec, "src-new")
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
        for label, ttft, repeat, build in (
            ("ref", 4000, None, None),
            ("t-cand-1", 4000, 1, "src-old"),
            ("t-cand-2", 6000, 2, "src-new"),
        ):
            write_jsonl(
                self.directory / f"{label}-n2.events.jsonl",
                [{"kind": "start", "n": 2, "rounds": 3}]
                + [
                    {
                        "kind": "round",
                        "n": 2,
                        "round": r,
                        "ttft_ms": [ttft] * 2,
                        "cached": [7968] * 2,
                    }
                    for r in range(3)
                ],
            )
            write_jsonl(
                self.directory / f"{label}-n2.status.jsonl", [status(0), status(30)]
            )
            (self.directory / f"{label}.out").write_text("DONE\n")
            if build:
                self.run_meta(
                    label, cell="t", tree_name="cand", repeat=repeat, build_id=build
                )
        verdicts, _, unused = matrix.ttft_cell(cell, spec, [self.directory])
        g5 = next(v for v in verdicts if v["rule"] == "G5 N=2")
        self.assertEqual(
            (g5["verdict"], g5["numbers"]["cand_median_worst_ms"]), (stats.FAIL, 6000)
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in unused],
            [("t-cand-1", ["build src-old, not the build under test src-new"])],
        )

    def test_an_abba_rerun_of_the_cand_build_under_test_is_judged(self):
        # s-serve1 after a fix: set 1 measured the build before it, and its
        # rerun took repeats 3 and 4, where cand is 10 % slower. Set 2 is
        # judged; set 1 is passed over, its runs listed.
        spec = self.with_cand_build(self.spec, "src-new")
        cell = {
            "name": "s1",
            "machine": "m5",
            "model": "fn-q2",
            "scenario": "serve1",
            "abba": {"reference": "pass", "sets": 1},
        }
        for tree, repeat, ttft, build in (
            ("pass", 1, 1000, "src-pass"),
            ("cand", 1, 1000, "src-old"),
            ("cand", 2, 1000, "src-old"),
            ("pass", 2, 1000, "src-pass"),
            ("pass", 3, 1000, "src-pass"),
            ("cand", 3, 1100, "src-new"),
            ("cand", 4, 1100, "src-new"),
            ("pass", 4, 1000, "src-pass"),
        ):
            label = f"s1-{tree}-{repeat}"
            write_jsonl(
                self.directory / f"{label}.events.jsonl",
                [
                    {"kind": "request", "size": 300, "sample": s, "ttft_ms": ttft}
                    for s in range(3)
                ],
            )
            (self.directory / f"{label}.out").write_text("DONE\n")
            self.run_meta(
                label, cell="s1", tree_name=tree, repeat=repeat, build_id=build
            )
        verdicts, unused = matrix.abba_cell(cell, spec, [self.directory])
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts if v["rule"][:2] == "G7"],
            [("G7 serve1_300_ttft_ms set 2", stats.FAIL)],
        )
        self.assertEqual(
            [u["label"] for u in unused],
            ["s1-cand-1", "s1-cand-2", "s1-pass-1", "s1-pass-2"],
        )

    def test_g7_single_judges_the_cand_build_under_test(self):
        # s-single after a fix: the 27B's repeat 1 passed on the build before
        # it, and its rerun on the build under test failed; the 35B's only
        # comparison is of the old build. G7 fails the 27B and is
        # INCONCLUSIVE for the 35B; both old comparisons are listed.
        spec = matrix.load_spec(SPEC)
        cell = next(c for c in spec["cells"] if c["name"] == "s-single")
        spec = {**self.with_cand_build(spec, "src-new"), "cells": [cell]}
        for label, build, passed in (
            ("s-single-cand-27b-q4-1", "src-old", True),
            ("s-single-cand-27b-q4-2", "src-new", False),
            ("s-single-cand-35b-q4-1", "src-old", True),
        ):
            rounds = [
                {
                    "version": version,
                    "identity": {
                        "build_id": build if version == "candidate" else "src-pass"
                    },
                }
                for version in ("baseline", "candidate", "candidate", "baseline")
            ]
            failures = [] if passed else ["partial_4k_hit_ttft_ms: FAIL"]
            (self.directory / f"{label}.backend-regression.json").write_text(
                json.dumps(
                    {
                        "rounds": rounds,
                        "comparison": {"failures": failures},
                        "pass": passed,
                    }
                )
            )
        document = matrix.analyze(spec, [self.directory])
        self.assertEqual(
            {v["rule"]: v["verdict"] for v in document["verdicts"]},
            {
                "G7 s-single-cand-27b-q4-2": stats.FAIL,
                "G7 35b-q4": stats.INCONCLUSIVE,
                "G7 flint": stats.INCONCLUSIVE,
            },
        )
        self.assertEqual(
            [u["label"] for u in document["invalid"]],
            ["s-single-cand-27b-q4-1", "s-single-cand-35b-q4-1"],
        )

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
        # server exited during the scenario fails G8. p-fn-churn-18g on the
        # M5: the owner touched the laptop at minute 3, and at minute 10 the
        # engine crashed and was relaunched. The run is listed, to be
        # repeated, and its crash still fails G8.
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
            self.directory,
            "crashed",
            done,
            [
                status(0),
                status(600, transport__restarts=1),
                status(960, transport__restarts=1),
            ],
        )
        self.run_meta("crashed")
        write_jsonl(
            self.directory / "crashed.idle.jsonl", [{"hid_idle_s": 4.0, "holders": []}]
        )
        churn_run(
            self.directory, "exited", done, [status(0), status(300)], complete=False
        )
        exit_error = "the server exited -9 during the scenario"
        self.run_meta("exited", complete=False, failure="server", error=exit_error)
        runs = [
            matrix.Run(self.directory, label, "c", "cand", "A")
            for label in ("refused", "called", "crashed", "exited")
        ]
        for run_ in runs:
            matrix.validate(run_, self.spec, cell)
        verdicts = matrix.churn_cell(cell, self.spec, runs, {}, PRICES)
        self.assertEqual(
            [(v["rule"], v["verdict"]) for v in verdicts],
            [("G8 crashed", stats.FAIL), ("G4 G8 G9 exited", stats.FAIL)],
        )
        self.assertEqual(
            verdicts[0]["numbers"],
            ["G8: the engine restarted (transport.restarts 0 -> 1)"],
        )
        self.assertIn(f"G8: {exit_error}", verdicts[1]["numbers"]["failures"])
        self.assertEqual(
            [(r.label, r.invalid, bool(r.voided)) for r in runs],
            [
                ("refused", [f"the run did not end with DONE: {refusal}"], True),
                ("called", ["the laptop was in use (HIDIdleTime 3.0 s)"], True),
                (
                    "crashed",
                    [
                        "the laptop was in use (HIDIdleTime 4.0 s)",
                        "the engine restarted (0 -> 1)",
                    ],
                    True,
                ),
                ("exited", [f"the run did not end with DONE: {exit_error}"], False),
            ],
        )
        # The same in a ttft, serve1 or warm cell.
        self.assertEqual(
            [
                (v["rule"], v["verdict"])
                for v in matrix.g8_verdict(cell, self.spec, runs[2])
            ],
            [("G8 crashed", stats.FAIL)],
        )
        self.assertEqual(matrix.g8_verdict(cell, self.spec, runs[1]), [])

    def test_a_request_status_counts_failed_makes_the_run_invalid(self):
        # m6-fn-churn cand: a request the engine failed counts in /status
        # requests.failed, though its client recorded it as done.
        cell = {"name": "c", "machine": "m6", "model": "fn-q2", "scenario": "churn"}
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        churn_run(
            self.directory,
            "failed",
            done,
            [
                status(0),
                status(500, requests__failed=1),
                status(960, requests__failed=1),
            ],
        )
        self.run_meta("failed")
        run_ = matrix.Run(self.directory, "failed", "c", "cand", "A")
        matrix.validate(run_, self.spec, cell)
        self.assertEqual(
            (run_.invalid, run_.voided), (["requests the server failed: 1"], [])
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
        # keeps counting output. cand did 10 % less work in the window, in
        # each order.
        spec = self.spec
        cell = next(c for c in spec["cells"] if c["name"] == "mini-35b-churn")
        self.assertEqual(matrix.work_second(cell), 600)
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        for reference, label, order in (
            ("mini35PSa", "mini-35b-churn-cand-A1", [2, 0, 1]),
            ("mini35PSb", "mini-35b-churn-cand-B1", [0, 1, 2]),
        ):
            churn_run(
                self.directory,
                reference,
                done,
                [
                    status(0),
                    status(600.5, rows=585 * 1000, disk__read_bytes=1000),
                    status(604.94, rows=585 * 1003, disk__read_bytes=1003),
                ],
            )
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
                        "order": order,
                    }
                )
            )
        runs = [
            r
            for r in matrix.load_runs(spec, [self.directory])
            if r.cell == cell["name"]
        ]
        self.assertEqual(sorted(r.tree for r in runs), ["cand", "cand", "pass", "pass"])
        work = next(
            v
            for v in matrix.point_gate(
                cell, runs, spec["machines"]["mini"], PRICES, self.gates, True
            )
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

    def test_each_candidate_takes_its_own_g4(self):
        # m6-fn-churn: every cand run holds G4; the floor = cap control's A1
        # has an 8 s client gap (G4c, the M6's floor is 7.3 s). r150's own
        # runs froze (9 s decode gaps). cand trails r150 by 3 % in work with
        # the same TTFT and latency: its own G4 makes the freeze its gain.
        def make(label, tree, order, work, chunks=(0, 100, 200), gap_ms=None):
            done = [
                {
                    "tag": f"w{w}-r1",
                    "start": 0,
                    "t": 200,
                    "ttft": 100,
                    "decode_tps": 30,
                    "chunks_ms": list(chunks),
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
                [status(900, rows=int(work * 585), **fields)],
            )
            r.tree, r.order = tree, order
            return r

        cell = {
            "name": "m",
            "machine": "m6",
            "release": {"throughput": "r150"},
            "candidates": ["cand"],
            "health": ["cand", "cand-floorcap"],
        }
        runs = (
            [
                make(f"r{o}{i}", "r150", o, 1000 + i, gap_ms=9000)
                for o in "AB"
                for i in range(2)
            ]
            + [make(f"c{o}{i}", "cand", o, 970 + i) for o in "AB" for i in range(2)]
            + [make("f1", "cand-floorcap", "A", 1000, chunks=(0, 8000))]
        )
        verdicts = matrix.churn_cell(cell, self.spec, runs, {}, PRICES)
        self.assertIn(
            ("G4 G8 G9 f1", stats.FAIL), [(v["rule"], v["verdict"]) for v in verdicts]
        )
        g1 = next(v for v in verdicts if v["rule"] == "G1 cand vs r150")
        self.assertTrue(g1["numbers"]["gain"], g1)
        self.assertEqual(g1["numbers"]["tolerance"], self.gates["trade_tolerance"])

    def test_a_cell_without_a_usable_candidate_run_says_so(self):
        # p-fn-churn-18g: the laptop's owner used the M5 Pro during cand's A
        # runs, the watchdog killed B1 before it wrote its meta.json, and B2
        # never ran. cand's gates are INCONCLUSIVE, naming its runs; the
        # killed run is listed, and B2 as not run.
        spec = matrix.load_spec(SPEC)
        cell = next(c for c in spec["cells"] if c["name"] == "p-fn-churn-18g")
        spec = {**spec, "cells": [cell]}
        model = spec["models"][cell["model"]]["model"]
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        for label in ("p-fn-churn-18g-cand-A1", "p-fn-churn-18g-cand-A2"):
            churn_run(self.directory, label, done, [status(0), status(960)])
            (self.directory / f"{label}.meta.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "cell": cell["name"],
                        "tree_name": "cand",
                        "order": [2, 0, 1],
                        "model": model,
                        "complete": True,
                    }
                )
            )
            write_jsonl(
                self.directory / f"{label}.idle.jsonl",
                [{"hid_idle_s": 3.0, "holders": []}],
            )
        churn_run(
            self.directory,
            "p-fn-churn-18g-cand-B1",
            done[:1],
            [status(0), status(400)],
            complete=False,
        )
        document = matrix.analyze(spec, [self.directory])
        verdicts = {
            v["rule"]: (v["verdict"], v["numbers"]) for v in document["verdicts"]
        }
        why = "no usable cand run: " + ", ".join(
            f"p-fn-churn-18g-cand-{run}" for run in ("A1", "A2", "B1", "B2")
        )
        for rule in (
            "G1 cand vs r150",
            "G2 cand vs r150",
            "G2 decode cand vs r150",
            "G3 cand vs main",
            "SSD disk_read_per_work",
            "SSD disk_written_per_work",
            "SSD expert_read_per_work",
        ):
            self.assertEqual(verdicts.get(rule), (stats.INCONCLUSIVE, why), rule)
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in document["invalid"]],
            [
                (
                    "p-fn-churn-18g-cand-A1",
                    ["the laptop was in use (HIDIdleTime 3.0 s)"],
                ),
                (
                    "p-fn-churn-18g-cand-A2",
                    ["the laptop was in use (HIDIdleTime 3.0 s)"],
                ),
                ("p-fn-churn-18g-cand-B1", ["the run did not end with DONE"]),
                ("p-fn-churn-18g-cand-B2", ["not run"]),
            ],
        )

    def test_an_ssd_cache_held_back_unlike_the_references_voids_the_run(self):
        # --max-cache-disk auto limits the SSD cache to a quarter of the
        # volume's free space beyond 10 GiB, so the free disk sets it. main's
        # sMNb held at most 10.71 GiB under 12.04 GiB (11.50 of 12.93 GB on
        # 62.6 GB free). On 50 GiB free a cand run's limit is 10 GiB: one that
        # fills it was held back where sMNb was not, and is voided; one that
        # held 8 GiB met neither limit, and counts. In the ample cell ntie's
        # sNFb filled its 10.49 GiB (11.26 GB): a cand run on a cleaned disk
        # holding 12.11 GiB under 14.90 GiB had room sNFb lacked, and is
        # voided; one that filled 10.49 GiB counts, though r150's s150Fb,
        # in none of that cell's gates, filled its 11.38 GiB.
        spec = matrix.load_spec(SPEC)
        names = ("s-fn-churn-18g", "s-fn-churn-ample")
        spec = {**spec, "cells": [c for c in spec["cells"] if c["name"] in names]}
        model = spec["models"]["fn-q2"]["model"]
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        for label, order, free, limit, held in (
            ("sMNb", None, 58.3, 12.04, 10.71),
            ("s-fn-churn-18g-cand-B1", [0, 1, 2], 50.0, 10.0, 9.95),
            ("s-fn-churn-18g-cand-B2", [0, 1, 2], 50.0, 10.0, 8.0),
            ("sNFb", None, 51.96, 10.49, 10.42),
            ("sMFb", None, 54.6, 11.14, 10.82),
            ("s150Fb", None, 55.5, 11.38, 11.34),
            ("s-fn-churn-ample-cand-A1", [2, 0, 1], 69.6, 14.9, 12.11),
            ("s-fn-churn-ample-cand-B1", [0, 1, 2], 51.96, 10.49, 10.45),
        ):
            samples = [
                status(
                    t,
                    disk__limit_bytes=limit * 2**30,
                    disk__used_bytes=used * 2**30,
                    disk__volume_available_bytes=free * 2**30,
                )
                for t, used in ((0, 0), (900, held))
            ]
            churn_run(self.directory, label, done, samples)
            if order:
                (self.directory / f"{label}.meta.json").write_text(
                    json.dumps(
                        {
                            "label": label,
                            "cell": label.rsplit("-cand-", 1)[0],
                            "tree_name": "cand",
                            "order": order,
                            "model": model,
                            "complete": True,
                        }
                    )
                )
        document = matrix.analyze(spec, [self.directory])
        self.assertEqual(
            [
                (u["label"], u["reasons"])
                for u in document["invalid"]
                if u["reasons"] != ["not run"]
            ],
            [
                (
                    "s-fn-churn-18g-cand-B1",
                    [
                        "SSD cache not comparable with main's sMNb: this run held "
                        "at most 9.95 GiB, filling its 10.00 GiB limit (50.00 GiB "
                        "free at Ready); sMNb held at most 10.71 GiB under a "
                        "12.04 GiB limit"
                    ],
                ),
                (
                    "s-fn-churn-ample-cand-A1",
                    [
                        "SSD cache not comparable with ntie's sNFb: this run held "
                        "at most 12.11 GiB under a 14.90 GiB limit (69.60 GiB "
                        "free at Ready); sNFb held at most 10.42 GiB, filling its "
                        "10.49 GiB limit"
                    ],
                ),
            ],
        )

    def test_scheduler_readouts_and_the_owners_conflict_report(self):
        # s-fn-churn-18g cand: 12 plans in 15 minutes left a request out for
        # the Index-page lane limit, past S's bound of 5 (amend/S.md D1);
        # each interval's committed rows per command give the unit sizes.
        counters = [
            # t, prefill_batches, prefill_rows, index_page_conflicts, prefill_units,
            # pass_commands, bound_cuts
            (0, 0, 0, 0, 0, 0, 0),
            (300, 1, 2048, 4, 1, 0, 0),
            (600, 3, 2048 + 2 * 8192, 9, 3, 0, 1),
            (900, 4, 2048 + 2 * 8192 + 4096, 12, 7, 1, 2),
        ]
        samples = [
            self.schema7(
                t,
                scheduler__prefill_batches=batches,
                scheduler__prefill_rows=rows,
                scheduler__index_page_conflicts=conflicts,
                scheduler__prefill_units=units,
                scheduler__pass_commands=passes,
                scheduler__bound_cuts=cuts,
            )
            for t, batches, rows, conflicts, units, passes, cuts in counters
        ]
        r = churn_run(self.directory, "s-cand", [], samples)
        r.tree, r.order = "cand", "A"
        cell = {"name": "s", "machine": "studio", "health": ["cand"]}
        verdicts = matrix.churn_cell(cell, self.spec, [r], {}, PRICES)
        readouts = next(v for v in verdicts if v["rule"] == "G4 G8 G9 s-cand")[
            "numbers"
        ]["readouts"]
        self.assertEqual(readouts["index_page_conflicts_per_15min"], 12)
        self.assertEqual(readouts["bound_cuts"], 2)
        self.assertEqual(
            readouts["rows_per_unit"],
            {"intervals": 3, "p10": 2048, "p50": 4096, "p90": 8192, "max": 8192},
        )
        self.assertEqual((readouts["pass_commands"], readouts["prefill_units"]), (1, 7))
        self.assertIn(
            ("index page conflicts s-cand", "report", {"per_15min": 12.0, "bound": 5}),
            [(v["rule"], v["verdict"], v["numbers"]) for v in verdicts],
        )

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
            matrix.abba_runs(cell, 1), (1000, 1010, 1005, 1002)
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

    def test_analyze_pools_the_cand_builds_it_is_given(self):
        # The M6 cells ran on the M3b+C build src-a; the release binary src-b
        # differs by a behaviour-neutral change, so its confirmation run
        # (one per order, AMENDMENTS "Final test start") is pooled with them.
        args = matrix.parse_args(
            ["analyze", "--spec", str(SPEC), "--results", ".", "--out", "o"]
            + ["--build", "cand=src-a,src-b"]
        )
        self.assertEqual(args.builds, {"cand": ["src-a", "src-b"]})
        spec = self.with_cand_build(self.spec, ["src-a", "src-b"])
        cell = next(c for c in spec["cells"] if c["name"] == "m6-fn-churn")
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        runs = []
        for label, build in (
            ("m6-fn-churn-cand-A1", "src-a"),
            ("m6-fn-churn-cand-A3", "src-b"),
            ("m6-fn-churn-cand-B1", "src-old"),
        ):
            churn_run(self.directory, label, done, [self.schema7(0)])
            (self.directory / f"{label}.meta.json").write_text(
                json.dumps({"label": label, "build_id": build})
            )
            runs.append(matrix.Run(self.directory, label, cell["name"], "cand"))
        kept, unused = matrix.usable(spec, cell, runs)
        self.assertEqual(
            [r.label for r in kept], ["m6-fn-churn-cand-A1", "m6-fn-churn-cand-A3"]
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in unused],
            [
                (
                    "m6-fn-churn-cand-B1",
                    ["build src-old, not the build under test src-a, src-b"],
                )
            ],
        )

    def test_g10_is_the_interval_rule(self):
        # m6-27b-churn: one cand and one pass run per order. cand's work at
        # 0.975 x pass lies inside the M6's +-3.6 % pooled resolution
        # (amend/V.md §4.2): INCONCLUSIVE for the owner, not FAIL; at 0.9 it
        # is surely worse, and a G4 failure fails it whatever the interval.
        cell = next(c for c in self.spec["cells"] if c["name"] == "m6-27b-churn")

        def runs(work):
            found = []
            for label, tree, order, rows in (
                ("pa", "pass", "A", 1000),
                ("pb", "pass", "B", 1000),
                ("ca", "cand", "A", work),
                ("cb", "cand", "B", work),
            ):
                done = [
                    {
                        "tag": f"w{w}-r1",
                        "start": 0,
                        "t": 100,
                        "ttft": 5,
                        "decode_tps": 30,
                    }
                    for w in range(3)
                ]
                r = churn_run(
                    self.directory, label, done, [status(900, rows=int(rows * 585))]
                )
                r.tree, r.order = tree, order
                found.append(r)
            return found

        def g10(work, g4_holds=True):
            return {
                v["rule"]: v["verdict"]
                for v in matrix.point_gate(
                    cell, runs(work), self.machine, PRICES, self.gates, g4_holds
                )
            }

        self.assertEqual(g10(975)["G10 work vs pass"], stats.INCONCLUSIVE)
        self.assertEqual(g10(900)["G10 work vs pass"], stats.FAIL)
        self.assertEqual(g10(1000, g4_holds=False)["G10 ttft vs pass"], stats.FAIL)

    def test_g10_ttft_takes_the_interval_over_round_worsts(self):
        # s-27b-ttft: cand's round worsts sit 3 % above pass's, within their
        # own round-to-round spread: INCONCLUSIVE for the owner, not FAIL.
        def ttft_run(label, worsts):
            write_jsonl(
                self.directory / f"{label}.jsonl",
                [
                    {
                        "kind": "round",
                        "n": 2,
                        "round": i,
                        "ttft_ms": [100, w],
                        "cached": [0, 0],
                        "queue_ms": [0, 0],
                    }
                    for i, w in enumerate(worsts)
                ],
            )
            (self.directory / f"{label}.out").write_text("DONE\n")

        ttft_run("p", [4000, 4200, 4400])
        ttft_run("c", [4120, 4326, 4532])
        cell = {
            "name": "t",
            "machine": "studio",
            "point_gate": "pass",
            "gate_n": [2],
            "recorded": {"pass": ["p"], "cand": ["c"]},
        }
        verdicts, _, _ = matrix.ttft_cell(cell, self.spec, [self.directory])
        self.assertEqual(verdicts[0]["rule"], "G10 N=2")
        self.assertEqual(verdicts[0]["verdict"], stats.INCONCLUSIVE, verdicts)

    def test_the_free_disk_window_keeps_the_ssd_cache_comparable(self):
        # s-fn-churn-ample: ntie's sNFb and sNFa filled their 10.49 GiB limits
        # (ranges up to 10.70 GiB) and main's sMFb held 10.82 GiB under 11.14
        # (range from 10.61): a cand run's limit must lie in [10.40, 10.92]
        # GiB, i.e. 51.6-53.7 GiB free with the 10 GiB floor their first
        # samples show. r150's s150Fb is in none of the cell's gates.
        spec = matrix.load_spec(SPEC)
        cell = next(c for c in spec["cells"] if c["name"] == "s-fn-churn-ample")
        spec = {**spec, "cells": [cell]}
        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        for label, free, limit, held in (
            ("sNFa", 51.92, 10.48, 10.32),
            ("sNFb", 51.96, 10.49, 10.42),
            ("sMFb", 54.6, 11.14, 10.82),
            ("s150Fb", 55.5, 11.38, 11.34),
        ):
            samples = [
                status(
                    t,
                    disk__limit_bytes=limit * 2**30,
                    disk__used_bytes=used * 2**30,
                    disk__volume_available_bytes=free * 2**30,
                )
                for t, used in ((0, 0), (900, held))
            ]
            churn_run(self.directory, label, done, samples)
        runs = matrix.load_runs(spec, [self.directory])
        window = matrix.ssd_window(cell, runs, "cand", spec["gates"]["tolerance"])
        self.assertEqual(window["references"], ["sMFb", "sNFa", "sNFb"])
        self.assertAlmostEqual(window["low"] / 2**30, 4 * 10.82 * 0.98**2 + 10, 2)
        self.assertAlmostEqual(window["high"] / 2**30, 4 * 10.48 / 0.98**2 + 10, 2)
        command = ["disk", "--spec", str(SPEC), "--results", str(self.directory)]
        command += ["--cell", "s-fn-churn-ample"]
        for free, code in ((56 * 2**30, 1), (52.5 * 2**30, 0), (47 * 2**30, 1)):
            with (
                mock.patch.object(matrix, "free_bytes", return_value=free),
                mock.patch("sys.stdout", new_callable=io.StringIO) as out,
            ):
                self.assertEqual(matrix.main(command), code, out.getvalue())
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(matrix.main(command[:-1] + ["122b-churn"]), 0)

    def test_prices_come_from_the_newest_complete_price_run(self):
        # p-35b-price-pass-1 failed at its first 14,336-token request: its
        # records stop at 2,048 tokens. Its rerun -2 completed; R is the
        # 14,336-token prompt's rate of -2, never -1's 2,048-token one.
        for label, sizes, complete in (
            ("p-35b-price-pass-1", (300, 2048), False),
            ("p-35b-price-pass-2", (300, 2048, 14336), True),
        ):
            write_jsonl(
                self.directory / f"{label}.events.jsonl",
                [
                    {
                        "kind": "request",
                        "size": size,
                        "prompt": size,
                        "ttft_ms": size / 2 + 500,
                        "decode_ms_per_token": 20,
                    }
                    for size in sizes
                ],
            )
            (self.directory / f"{label}.out").write_text("DONE\n" if complete else "")
        prices = matrix.serve1_prices("p-35b-price-pass-1", [self.directory])
        self.assertEqual(prices["source"], "p-35b-price-pass-2")
        self.assertAlmostEqual(prices["R"], 14336 / 7.668)
        (self.directory / "p-35b-price-pass-2.out").write_text("")
        self.assertIsNone(matrix.serve1_prices("p-35b-price-pass-1", [self.directory]))

    def test_a_job_without_history_is_late_at_twice_the_minutes_it_states(self):
        # M5 19-122b-yarn states 75 minutes and no yarn job finished before:
        # at 50 minutes it is not late (20 minutes per churn run would say
        # so); past 150 it is.
        queue = Path(tempfile.mkdtemp())
        (queue / "done").mkdir()
        (queue / "running").mkdir()
        job = queue / "running" / "19-122b-yarn-cand-1.sh"
        job.write_text("#!/bin/bash\n# Expected ~75 min (estimate)\n")
        (queue / "running" / "19-122b-yarn-cand-1.sh.log").write_text("")
        self.assertEqual(matrix.overruns(queue, now=time.time() + 50 * 60), [])
        self.assertEqual(len(matrix.overruns(queue, now=time.time() + 151 * 60)), 1)
        job.write_text("LABEL=s-fn-churn-18g-cand-A1; KIND=churn; EXPECT=20\n")
        self.assertEqual(len(matrix.overruns(queue, now=time.time() + 41 * 60)), 1)

    def test_a_single_comparison_its_job_voided_is_not_judged(self):
        # M5 p-single: the laptop was touched during the make, so the job
        # wrote LABEL.void beside the comparison it copied.
        cell = next(c for c in self.spec["cells"] if c["name"] == "p-single")
        spec = {**self.with_cand_build(self.spec, "src-new"), "cells": [cell]}
        label = "p-single-cand-27b-q4-1"
        rounds = [
            {"version": version, "identity": {"build_id": build}}
            for version, build in (("baseline", "p"), ("candidate", "src-new"))
        ]
        (self.directory / f"{label}.backend-regression.json").write_text(
            json.dumps({"rounds": rounds, "comparison": {"failures": []}, "pass": True})
        )
        (self.directory / f"{label}.void").write_text(
            "HIDIdleTime 30 s < the make's 1200 s\n"
        )
        document = matrix.analyze(spec, [self.directory])
        self.assertEqual(
            {v["rule"]: v["verdict"] for v in document["verdicts"]}["G7 27b-q4"],
            stats.INCONCLUSIVE,
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in document["invalid"]],
            [(label, ["void: HIDIdleTime 30 s < the make's 1200 s"])],
        )

    def test_check_fails_a_run_analyze_would_void(self):
        # Studio 20-s-fn-churn-18g-cand-A1 with mediaanalysisd at 117.8 % in
        # the top five (it came back on 11-02 20:25): the job's check fails
        # it at once, so it is rerun now, not after the batch's analyze. A
        # serve1 run of m6-serve1 is checked as one: no churn workers or
        # load window, but its voids and G8.
        model = self.spec["models"]["fn-q2"]["model"]
        busy = [[55828, 117.8, 0.5, "mediaanalysisd", False]]
        quiet = [[145, 11.9, 0.1, "WindowServer", False]]

        def meta(label, cell, scenario, rows):
            (self.directory / f"{label}.meta.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "cell": cell,
                        "tree_name": "cand",
                        "scenario": scenario,
                        "model": model,
                        "complete": True,
                        "pids": [700],
                        "ps": [{"rows": rows}],
                    }
                )
            )

        def check(label, machine, *more):
            command = ["check", str(self.directory), label, "--spec", str(SPEC)]
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = matrix.main(command + ["--machine", machine, *more])
            return code, out.getvalue()

        done = [
            {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
            for w in range(3)
        ]
        label = "s-fn-churn-18g-cand-A1"
        churn_run(self.directory, label, done, [self.schema7(0), self.schema7(960)])
        meta(label, "s-fn-churn-18g", "churn", busy)
        code, out = check(label, "studio", "--minutes", "15")
        self.assertEqual(code, 1, out)
        self.assertIn(
            "FAIL voided: a process this run did not start in the top five by "
            "CPU: mediaanalysisd at 117.8 %",
            out,
        )
        for label, rows, expected in (
            ("m6-serve1-cand-1", quiet, 0),
            ("m6-serve1-cand-2", busy, 1),
        ):
            write_jsonl(
                self.directory / f"{label}.events.jsonl",
                [{"kind": "request", "size": 300, "prompt": 300, "ttft_ms": 900}],
            )
            write_jsonl(self.directory / f"{label}.status.jsonl", [status(0)])
            (self.directory / f"{label}.out").write_text("DONE\n")
            meta(label, "m6-serve1", "serve1", rows)
            code, out = check(label, "m6")
            self.assertEqual(code, expected, out)
        self.assertIn("FAIL voided: a process this run did not start", out)

    def test_a_job_types_expected_time_is_its_first_measured_run(self):
        # M6: 02-m6-fn-churn-cand-floorcap-A1 ran before 01 completed and
        # failed in seconds (README: requeue it): it measured nothing, so 05
        # of its type is held to the 20 minutes it states, not to twice 5 s.
        # 01 waited 30 minutes for a build's slot and then ran 20: its type
        # is held to its run, not to its wait.
        queue = Path(tempfile.mkdtemp())
        (queue / "done").mkdir()
        (queue / "running").mkdir()
        failed = queue / "done" / "02-m6-fn-churn-cand-floorcap-A1.sh.log"
        failed.write_text("FAIL: no complete m6-fn-churn cand run\nEXIT=1\n")
        born = failed.stat().st_birthtime
        os.utime(failed, (born + 5, born + 5))
        job = queue / "running" / "05-m6-fn-churn-cand-floorcap-B2.sh"
        job.write_text("# Expected ~20 min\n")
        (queue / "running" / "05-m6-fn-churn-cand-floorcap-B2.sh.log").write_text("")
        self.assertEqual(matrix.overruns(queue, now=time.time() + 10 * 60), [])
        measured = queue / "done" / "01-m6-fn-churn-cand-A1.sh.log"
        measured.write_text("")
        born = measured.stat().st_birthtime
        measured.write_text(f"run started at epoch {int(born) + 1800}\nEXIT=0\n")
        os.utime(measured, (born + 3000, born + 3000))
        (queue / "running" / "04-m6-fn-churn-cand-B1.sh.log").write_text("")
        late = matrix.overruns(queue, now=time.time() + 41 * 60)
        self.assertEqual(
            [line.split(":")[0] for line in late],
            ["04-m6-fn-churn-cand-B1.sh", "05-m6-fn-churn-cand-floorcap-B2.sh"],
        )

    def test_the_ssd_rule_reports_its_ratios_without_reference_pairs(self):
        # p-35b-churn-18g, the only HiSparse-off pressure cell: r150 and main
        # recorded one run per order, so no reference pair gives a sigma.
        # cand's own pairs give one (as sigma_for pools them), and with one
        # cand run per order the machine's recorded SSD sigma does: the
        # ratios are reported and judged, never "no sigma". A reference
        # missing an order cannot clear a cand surely higher than the other.
        cell = next(c for c in self.spec["cells"] if c["name"] == "p-35b-churn-18g")
        machine = self.spec["machines"]["m5"]

        def run(label, tree, order, read):
            sample = status(
                900,
                rows=585 * 1000,
                disk__read_bytes=read,
                disk__written_bytes=read / 10,
                streamed_experts__prefill__read_bytes=read * 100,
            )
            r = churn_run(self.directory, label, [], [sample])
            r.tree, r.order = tree, order
            return r

        refs = [run(f"{t}{o}", t, o, 1e9) for t in ("r150", "main") for o in "AB"]
        pairs = [
            run(f"c{o}{i}", "cand", o, 1e9 * (1 + 0.01 * i))
            for o in "AB"
            for i in (0, 1)
        ]
        for runs in (refs + pairs, refs + pairs[::2]):
            verdicts = matrix.ssd_rule(cell, runs, machine, PRICES, "cand")
            self.assertEqual([v["verdict"] for v in verdicts], [stats.PASS] * 3)
            for v in verdicts:
                self.assertTrue(v["numbers"]["main"]["ratio"].startswith("1.0"), v)
        self.assertEqual(
            verdicts[0]["numbers"]["sigma"], machine["sigma"]["ssd"]["sigma"]
        )
        higher = [
            run(f"h{o}{i}", "cand", o, 1.5e9 * (1 + 0.01 * i))
            for o in "AB"
            for i in (0, 1)
        ]
        verdicts = matrix.ssd_rule(cell, refs + higher, machine, PRICES, "cand")
        self.assertEqual([v["verdict"] for v in verdicts], [stats.FAIL] * 3)
        verdicts = matrix.ssd_rule(cell, refs[:3] + higher, machine, PRICES, "cand")
        self.assertEqual([v["verdict"] for v in verdicts], [stats.INCONCLUSIVE] * 3)
        self.assertEqual(
            verdicts[0]["numbers"]["main"]["missing"], {"B": "no reference run"}
        )

    def test_one_order_alone_decides_no_gate(self):
        # m6-27b-churn plans one cand run per order. With B1 voided, order A
        # alone would decide G10; the orders are different workloads (the
        # Studio's 18G TTFT: 175 s in A, 136 s in B, DATA.md), so G10 is
        # INCONCLUSIVE and its numbers name the order it lacks.
        cell = next(c for c in self.spec["cells"] if c["name"] == "m6-27b-churn")
        found = []
        for label, tree, order, rows in (
            ("pa", "pass", "A", 1000),
            ("pb", "pass", "B", 1000),
            ("ca", "cand", "A", 900),
        ):
            done = [
                {"tag": f"w{w}-r1", "start": 0, "t": 100, "ttft": 5, "decode_tps": 30}
                for w in range(3)
            ]
            r = churn_run(
                self.directory, label, done, [status(900, rows=int(rows * 585))]
            )
            r.tree, r.order = tree, order
            found.append(r)
        verdicts = {
            v["rule"]: v
            for v in matrix.point_gate(
                cell, found, self.machine, PRICES, self.gates, True
            )
        }
        work = verdicts["G10 work vs pass"]
        self.assertEqual(work["verdict"], stats.INCONCLUSIVE, work)
        self.assertEqual(work["numbers"]["orders"], {"A": "1 vs 1"})
        self.assertEqual(work["numbers"]["missing"], {"B": "no candidate run"})
        result = stats.interval({"A": ([1.0], [1.0]), "B": ([], [1.0])}, 0.02, 10)
        self.assertIsNone(result["pooled"])
        self.assertEqual(result["missing"], {"B": ["candidate"]})

    def test_a_noisy_single_comparison_waits_for_its_rerun(self):
        # p-single 27B repeat 1: one metric's ABBA spread was past abba's 5 %
        # limit (it measured nothing: "rerun idle"), so G7 is INCONCLUSIVE,
        # not FAIL, until repeat 2 measures; repeat 2 passes and replaces it.
        # The 35B's FAIL stays beside its later PASS.
        cell = next(c for c in self.spec["cells"] if c["name"] == "p-single")
        spec = {**self.with_cand_build(self.spec, "src-new"), "cells": [cell]}
        rounds = [
            {"version": version, "identity": {"build_id": build}}
            for version, build in (
                ("baseline", "p"),
                ("candidate", "src-new"),
                ("candidate", "src-new"),
                ("baseline", "p"),
            )
        ]
        measured = {"verdict": "pass", "pass": True}

        def comparison(label, speed, failures):
            (self.directory / f"{label}.backend-regression.json").write_text(
                json.dumps(
                    {
                        "rounds": rounds,
                        "comparison": {
                            "speed": speed,
                            "failures": failures,
                            "pass": not failures,
                        },
                        "weights": {"pass": True},
                        "pass": not failures,
                    }
                )
            )

        comparison(
            "p-single-cand-27b-q4-1",
            {"decode": {"verdict": "inconclusive", "pass": False}, "short": measured},
            ["decode: INCONCLUSIVE (rerun idle): noise 5.83%"],
        )
        comparison(
            "p-single-cand-35b-q4-1",
            {"decode": {"verdict": "fail", "pass": False}},
            ["decode: FAIL: regression +4.00% (bound 2.00%)"],
        )
        comparison("p-single-cand-35b-q4-2", {"decode": measured}, [])

        def verdicts():
            document = matrix.analyze(spec, [self.directory])
            return document, {v["rule"]: v["verdict"] for v in document["verdicts"]}

        self.assertEqual(
            verdicts()[1],
            {
                "G7 p-single-cand-27b-q4-1": stats.INCONCLUSIVE,
                "G7 p-single-cand-35b-q4-1": stats.FAIL,
                "G7 p-single-cand-35b-q4-2": stats.PASS,
            },
        )
        comparison("p-single-cand-27b-q4-2", {"decode": measured}, [])
        document, found = verdicts()
        self.assertEqual(
            found,
            {
                "G7 p-single-cand-27b-q4-2": stats.PASS,
                "G7 p-single-cand-35b-q4-1": stats.FAIL,
                "G7 p-single-cand-35b-q4-2": stats.PASS,
            },
        )
        self.assertEqual(
            [(u["label"], u["reasons"]) for u in document["invalid"]],
            [
                (
                    "p-single-cand-27b-q4-1",
                    ["too noisy to measure, superseded by repeat 2"],
                )
            ],
        )

    def test_a_trade_carries_the_numbers_of_its_gains(self):
        # cand trails r150 by 3 % in work with a clearly better TTFT; the
        # trade's record names that ratio, and the concurrent TTFT figures
        # the machine's TTFT cell gave.
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
                self.directory, label, done, [status(900, rows=int(work * 585))]
            )
            r.tree, r.order = tree, order
            return r

        cell = {
            "name": "c",
            "machine": "m6",
            "release": {"throughput": "r150"},
            "health": [],
        }
        runs = [
            make(f"r{o}{i}", "r150", o, 1000 + i, 100) for o in "AB" for i in range(3)
        ] + [make(f"f{o}{i}", "cand", o, 970 + i, 60) for o in "AB" for i in range(2)]
        concurrent = {
            "gain": False,
            "cell": "m6-fn-ttft",
            "n": 4,
            "cand_worst_max_ms": 5300,
            "t150_worst_min_ms": 5100,
        }
        g1 = next(
            v
            for v in matrix.churn_cell(
                cell, self.spec, runs, {"m6": concurrent}, PRICES
            )
            if v["rule"].startswith("G1")
        )
        self.assertTrue(g1["trade"])
        self.assertTrue(g1["numbers"]["gains"]["ttft vs r150"].startswith("0.600"))
        self.assertEqual(g1["numbers"]["gains"]["concurrent_ttft"], concurrent)


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

    def test_a_single_job_copies_only_the_comparison_its_make_wrote(self):
        # m6-single reruns in a tree whose build/release holds an earlier
        # run's passing comparison. A make that fails before
        # backend_regression.py writes one leaves no copy; a failing
        # comparison it wrote is copied. The job runs under bash, as gpuq's
        # runner runs it, and under zsh, its own.
        spec = matrix.load_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            machines = spec["machines"]
            spec = {
                **spec,
                "machines": {
                    **machines,
                    "m6": {**machines["m6"], "lock": str(root / "gpu.lock")},
                },
            }
            tree, out, tools = root / "cand", root / "out", root / "bin"
            model = spec["models"]["27b-iq3"]["model"]
            results = (
                tree
                / "build/release"
                / model.replace("/", "--").replace(":", "--")
                / "backend-regression.json"
            )
            results.parent.mkdir(parents=True)
            out.mkdir()
            tools.mkdir()
            arguments = matrix.parse_args(
                [
                    "plan",
                    "--spec",
                    str(SPEC),
                    "--machine",
                    "m6",
                    "--tree",
                    f"cand={tree}",
                    "--tree",
                    f"pass={root / 'pass'}",
                    "--tools",
                    str(tree),
                    "--out",
                    str(out),
                    "--jobs",
                    str(root / "jobs"),
                    "--cells",
                    "m6-single",
                ]
            )
            job = next(j for j in matrix.plan(spec, arguments) if "27b-iq3" in j.name)
            copied = out / "m6-single-cand-27b-iq3-1.backend-regression.json"
            for shell in ("/bin/bash", "/bin/zsh"):
                for writes, code in ((None, 2), ('{"pass": false}', 1)):
                    results.write_text('{"pass": true}')
                    (tools / "make").write_text(
                        "#!/bin/sh\n"
                        + (f"echo '{writes}' > {results}\n" if writes else "")
                        + f"exit {code}\n"
                    )
                    (tools / "make").chmod(0o755)
                    copied.unlink(missing_ok=True)
                    ran = subprocess.run(
                        [shell, str(job)],
                        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertTrue(
                        ran.stdout.endswith(f"EXIT={code}\n"),
                        f"{shell}: {ran.stdout}{ran.stderr}",
                    )
                    if writes:
                        self.assertEqual(
                            json.loads(copied.read_text()), {"pass": False}, shell
                        )
                    else:
                        self.assertFalse(copied.exists(), shell)

    def test_analyze_takes_the_cand_build_under_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                "analyze",
                "--spec",
                str(SPEC),
                "--results",
                str(root),
                "--out",
                str(root / "verdict"),
            ]
            with (
                mock.patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                matrix.parse_args(command)
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                code = matrix.main(command + ["--build", "cand=src-new"])
            self.assertEqual(code, 0)
            document = json.loads((root / "verdict" / "verdict.json").read_text())
            self.assertEqual(document["builds"], {"cand": "src-new"})

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
        # Recorded references alone: every candidate cell is "not run", each
        # planned churn run is listed as not run, and every recorded run is
        # valid. verdict.md carries bin/cmp.py's and pertag.py's numbers of
        # each fixed-order set (cmp.txt, pertag.txt).
        document = matrix.analyze(matrix.load_spec(SPEC), [FIXTURES / "m6"])
        verdicts = {(v["cell"], v["rule"]): v["verdict"] for v in document["verdicts"]}
        self.assertEqual(verdicts[("m6-fn-churn", "cell")], "not run")
        self.assertEqual(verdicts[("m6-fn-ttft", "G5 N=2")], stats.INCONCLUSIVE)
        self.assertEqual(
            {tuple(u["reasons"]) for u in document["invalid"]}, {("not run",)}
        )
        self.assertIn(
            ("m6-fn-churn", "m6-fn-churn-cand-floorcap-B2"),
            {(u["cell"], u["label"]) for u in document["invalid"]},
        )
        markdown = matrix.verdict_markdown(document)
        self.assertIn("## Runs not used", markdown)
        self.assertEqual(
            [
                (s["cell"], s["order"], [row["label"] for row in s["cmp"]])
                for s in document["legacy"]
            ],
            [
                ("m6-fn-churn", "A", ["m150a", "m150a2", "mMNa", "mMNa2"]),
                ("m6-fn-churn", "B", ["m150b", "mMNb"]),
            ],
        )
        for line in (FIXTURES / "cmp.txt").read_text().splitlines():
            self.assertIn(
                ("    " if line.startswith(" ") else "### m6-fn-churn ") + line.strip(),
                markdown,
            )
        pertag = (FIXTURES / "pertag.txt").read_text().splitlines()[-4:]
        for line in pertag:
            self.assertIn(f"    {line}\n", markdown)


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
        # A request the server failed, a server gone by the end of the
        # scenario, or anything else that ends a scenario once the server is
        # ready, ends the run as the server's failure, which fails G8;
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
        # p-serve1 cand: alive on 127.0.0.1, the server refuses a request,
        # leaves its 14K request's stream to time out, or never goes idle.
        exited.clear()
        self.assertEqual(
            recorded("alive", connection_refused),
            (1, "server", "ConnectionRefusedError: [Errno 61] Connection refused"),
        )

        def wedged(*args, **kwargs):
            raise TimeoutError("timed out")

        self.assertEqual(
            recorded("wedged", wedged), (1, "server", "TimeoutError: timed out")
        )

        # A request it kept streaming past --request-seconds.
        def expired(*args, **kwargs):
            raise clients.Deadline(
                "requests still streaming 2137 s after their send: w1-r1"
            )

        self.assertEqual(
            recorded("expired", expired),
            (
                1,
                "server",
                "Deadline: requests still streaming 2137 s after their send: w1-r1",
            ),
        )

        def owing(*args, **kwargs):
            raise RuntimeError("server did not go idle")

        self.assertEqual(
            recorded("owing", owing),
            (1, "server", "RuntimeError: server did not go idle"),
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
                meta = json.loads((Path(directory) / "busy.meta.json").read_text())
                self.assertEqual((meta["complete"], meta["failure"]), (False, "port"))
        self.assertFalse(run.port_answers(port))


if __name__ == "__main__":
    unittest.main()
