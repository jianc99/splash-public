"""A --lazy-load server: it answers without an engine, and the first request
that needs one loads it, the requests that arrive meanwhile waiting for that
load, over the runtime and a stub engine."""

import concurrent.futures
import contextlib
import http.client
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import native_peer
from dev.tests.server import test_server_main
from dev.tests.server_fixtures import (
    READY,
    FakeFactory,
    FakeTokenizer,
    chat_body,
    console_written,
    main_args,
    make_frontend,
)
from install import clients, launcher
from server import backend as backend_api
from server import dialects
from server import protocol as wire
from server import runtime as engine_runtime
from server import server as api
from server.frontend import Frontend


def engine(process, message):
    """The stub engine's answers: its status, and three tokens a request."""
    if isinstance(message, wire.StatusRequestFrame):
        process.send(native_peer.status_event(message.correlation_id))
    elif isinstance(message, wire.RequestFrame):
        request_id = message.request_id
        process.send(wire.StartEvent(request_id, 0, 0))
        process.send(wire.TokensEvent(request_id, 0, (14, 15)))
        process.send(
            wire.DoneEvent(
                request_id,
                wire.FinishReason.STOP,
                len(message.prompt_tokens),
                2,
                100,
                200,
                350,
                (),
            )
        )


class LazyServer:
    """A server as main() makes it with --lazy-load, over a runtime whose
    engines are stubs: each starts without its Ready, which the test sends
    (ready())."""

    def __init__(self, test):
        self.factory = FakeFactory(handler=engine, initial_output=b"")
        self.runtime = engine_runtime.MultiplexedRuntime(
            process_factory=self.factory, pending_limit=4
        )
        self.tokenizer = FakeTokenizer()
        self.backend = backend_api.NativeBackend(
            self.runtime, self.tokenizer, request_logger=lambda _record: None
        )
        self.loads = 0

        def load(latencies):
            self.loads += 1
            if not self.runtime.wait_ready():
                raise engine_runtime.EngineUnhealthy("did not become ready")
            return make_frontend(
                self.tokenizer,
                self.backend,
                "test-model",
                128,
                30,
                2,
                vision=False,
                latencies=latencies,
            )

        self.unloaded = api.UnloadedApp(
            self.backend,
            "test-model",
            30,
            dialects.responses.ResponseStore(),
            load,
            vision=False,
        )
        self.server = api.FrontendServer(("127.0.0.1", 0), self.unloaded)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        test.enterContext(mock.patch.object(api, "print_status"))
        test.enterContext(mock.patch.object(backend_api, "print_status"))
        test.addCleanup(self.close)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        connection.request(
            method,
            path,
            None if body is None else json.dumps(body),
            {} if body is None else {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        if response.getheader("Content-Type") != "application/json":
            return response.status, payload
        return response.status, json.loads(payload)

    def ready(self, index=-1):
        self.factory.processes[index].send(READY)

    def wait_for(self, predicate, what):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"{what} within 5 s")
            time.sleep(0.01)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.backend.close()


class LazyLoadTests(unittest.TestCase):
    def test_a_lazy_server_answers_what_it_knows_without_an_engine(self):
        """The chat page, the probes, the model list, the stored responses and
        a release are answered without starting an engine: the status says
        unloaded, its weights released, and the model list has the names and
        the input the installation serves, its context null."""
        lazy = LazyServer(self)
        self.assertEqual(lazy.request("GET", "/health"), (200, {"status": "ok"}))
        self.assertEqual(lazy.request("GET", "/ready"), (200, {"status": "unloaded"}))
        status, body = lazy.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertFalse(body["ready"])
        self.assertEqual(body["weights"], {"released": True, "restoring": False})
        transport = body["transport"]
        self.assertTrue(transport["unloaded"])
        self.assertFalse(transport["recovering"])
        self.assertFalse(transport["status_stale"])
        self.assertNotIn("error", transport)
        self.assertEqual(body["instance"]["model"], "test-model")
        self.assertEqual(lazy.request("GET", "/metrics")[0], 200)
        status, models = lazy.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        (model,) = models["data"]
        self.assertEqual(model["id"], "test-model")
        self.assertIsNone(model["context_length"])
        self.assertIsNone(model["max_model_len"])
        self.assertIsNone(model["reasoning"])
        self.assertFalse(model["vision"])
        self.assertEqual(model["input_modalities"], ["text"])
        self.assertEqual(model["limits"]["images"], wire.MAX_IMAGE_SPANS)
        self.assertEqual(lazy.request("GET", "/v1/models/test-model")[1], model)
        self.assertEqual(lazy.request("GET", "/v1/responses/resp_1")[0], 404)
        self.assertEqual(lazy.request("DELETE", "/v1/responses/resp_1")[0], 404)
        self.assertEqual(
            lazy.request("POST", "/v1/memory/release"),
            (
                200,
                {"released": True, "weights": {"released": True, "restoring": False}},
            ),
        )
        self.assertEqual(lazy.factory.processes, [])
        self.assertIs(lazy.server.app, lazy.unloaded)

    def test_the_first_request_loads_the_engine_once_for_every_request_waiting(self):
        """Requests that need the engine wait for the one load the first of
        them starts, a token count among them; meanwhile the status says the
        weights are restoring and a release is refused as during a restore.
        Each is then served, and the server answers as any loaded one does."""
        lazy = LazyServer(self)
        paths = ["/v1/chat/completions"] * 3 + ["/v1/chat/completions/count_tokens"]
        with concurrent.futures.ThreadPoolExecutor(len(paths)) as pool:
            answers = [
                pool.submit(lazy.request, "POST", path, chat_body()) for path in paths
            ]
            lazy.wait_for(lambda: len(lazy.factory.processes) == 1, "no engine spawned")
            status, body = lazy.request("GET", "/status")
            self.assertTrue(body["transport"]["unloaded"])
            self.assertEqual(body["weights"], {"released": True, "restoring": True})
            status, body = lazy.request("POST", "/v1/memory/release")
            self.assertEqual((status, body["error"]["code"]), (409, "engine_busy"))
            self.assertFalse(any(answer.done() for answer in answers))
            lazy.ready()
            results = [answer.result(5) for answer in answers]
        for status, body in results[:3]:
            self.assertEqual(status, 200)
            message = body["choices"][0]["message"]
            self.assertEqual(message["reasoning_content"], "first second\n")
        self.assertEqual(results[3], (200, {"input_tokens": 2}))
        self.assertEqual(len(lazy.factory.processes), 1)
        self.assertEqual(lazy.loads, 1)
        self.assertIsInstance(lazy.server.app, Frontend)
        status, body = lazy.request("GET", "/status")
        self.assertFalse(body["transport"]["unloaded"])
        self.assertTrue(body["transport"]["ready"])
        self.assertEqual(
            body["capabilities"], {"max_context_tokens": READY.max_context_tokens}
        )
        # The requests measured while it loaded count on.
        self.assertEqual(
            body["latency"]["http_request"]["count"],
            lazy.unloaded.latencies.snapshot()["http_request"]["count"],
        )
        self.assertGreaterEqual(body["latency"]["http_request"]["count"], 4)
        self.assertEqual(lazy.request("GET", "/ready"), (200, {"status": "ready"}))
        (model,) = lazy.request("GET", "/v1/models")[1]["data"]
        self.assertEqual(model["context_length"], READY.max_context_tokens)
        self.assertEqual(
            lazy.request("POST", "/v1/chat/completions", chat_body())[0], 200
        )
        self.assertEqual(len(lazy.factory.processes), 1)

    def test_a_loaded_lazy_server_answers_as_one_loaded_at_start(self):
        """Once loaded, the model list and the status's shape are those of a
        server whose engine started with it."""
        lazy = LazyServer(self)
        loading = threading.Thread(
            target=lazy.request, args=("POST", "/tokenize", {"prompt": "hi"})
        )
        loading.start()
        lazy.wait_for(lambda: lazy.factory.processes, "no engine spawned")
        lazy.ready()
        loading.join(5)
        eager = api.FrontendServer(
            ("127.0.0.1", 0),
            make_frontend(
                lazy.tokenizer, lazy.backend, "test-model", 128, 30, 2, vision=False
            ),
        )
        self.addCleanup(eager.server_close)
        self.assertEqual(lazy.server.app.max_context, eager.app.max_context)
        lazy_status, eager_status = lazy.server.status(), eager.status()
        self.assertEqual(set(lazy_status), set(eager_status))
        self.assertEqual(lazy_status["transport"], eager_status["transport"])

    def test_a_failed_first_load_fails_its_requests_and_the_next_loads_again(self):
        """An engine that fails before its Ready fails every request waiting
        for it with the failure, which the status names; the server stays
        unloaded, and the next request that needs the engine starts another."""
        lazy = LazyServer(self)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            answers = [
                pool.submit(lazy.request, "POST", "/v1/chat/completions", chat_body())
                for _ in range(2)
            ]
            lazy.wait_for(lambda: len(lazy.factory.processes) == 1, "no engine spawned")
            lazy.factory.processes[0].kill()
            results = [answer.result(5) for answer in answers]
        for status, body in results:
            self.assertEqual(status, 503)
            self.assertEqual(body["error"]["code"], "runtime_unavailable")
        self.assertEqual(lazy.loads, 1)
        status, body = lazy.request("GET", "/status")
        self.assertTrue(body["transport"]["unloaded"])
        self.assertFalse(body["transport"]["recovering"])
        self.assertEqual(body["weights"], {"released": True, "restoring": False})
        self.assertEqual(body["transport"]["error"], results[0][1]["error"]["message"])
        self.assertEqual(lazy.request("GET", "/ready"), (200, {"status": "unloaded"}))
        self.assertIs(lazy.server.app, lazy.unloaded)
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            answer = pool.submit(
                lazy.request, "POST", "/v1/chat/completions", chat_body()
            )
            lazy.wait_for(lambda: len(lazy.factory.processes) == 2, "no second engine")
            lazy.ready()
            self.assertEqual(answer.result(5)[0], 200)
        self.assertEqual(lazy.loads, 2)
        self.assertNotIn("error", lazy.request("GET", "/status")[1]["transport"])


class LazyMainTests(unittest.TestCase):
    main_starting = test_server_main.ServerMainTests.main_starting

    def test_main_with_lazy_load_answers_before_it_starts_the_engine(self):
        """main() with --lazy-load starts no engine: it listens with the
        stand-in, its record says ready, and the stand-in's load is the rest
        of the start main() makes without the option, whose warm-up lines
        leave the record as it is."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        runtime = mock.Mock()
        runtime.capabilities = READY
        runtime.wait_ready.return_value = True
        seen = {}

        def serve():
            seen["record"] = json.loads(record.read_text())
            seen["app"] = server.app
            runtime.start.assert_not_called()
            runtime.wait_ready.assert_not_called()
            runtime.on_warming(wire.WarmingEvent(1, 9))
            seen["loaded"] = server.app.load(None)
            seen["after"] = json.loads(record.read_text())

        with contextlib.ExitStack() as stack:
            for patch in self.main_starting(
                main_args(lazy_load=True, serve_record=record), runtime
            ):
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(api, "check_capabilities"))
            console = stack.enter_context(mock.patch.object(api, "print_status"))
            frontend = api.Frontend
            server = api.FrontendServer.return_value
            server.address_family = None
            server.serve_forever.side_effect = serve
            api.main()
        self.assertEqual(seen["record"], {"pid": 4242, "phase": "ready"})
        self.assertIsInstance(seen["app"], api.UnloadedApp)
        self.assertIs(seen["loaded"], frontend.return_value)
        self.assertIsNone(frontend.call_args.kwargs["latencies"])
        runtime.wait_ready.assert_called_once_with()
        self.assertEqual(seen["after"], seen["record"])
        console.assert_any_call("Warming up · 1 of 9 steps")
        lines = [call.args[0] for call in console.call_args_list]
        self.assertIn(
            "Ready · test-model · loads on the first request · http://127.0.0.1:8000",
            lines,
        )
        self.assertIn(
            f"Loaded · test-model · context {READY.max_context_tokens // 1024}K"
            + ("" if READY.vision else " · text only"),
            lines,
        )


if __name__ == "__main__":
    unittest.main()


class LazyLauncherTests(unittest.TestCase):
    def test_an_agent_launcher_loads_the_model_before_it_reads_the_context(self):
        """`splash claude` against an unloaded server has it load its model,
        by a token count, then sets the agent up with the context the loaded
        engine serves."""
        lazy = LazyServer(self)
        port = lazy.server.server_port
        home = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"HOME": home}))
        os.environ.pop("SPLASH_API_KEY", None)
        self.enterContext(
            mock.patch.object(clients, "find_executable", return_value="/bin/claude")
        )
        execute = self.enterContext(mock.patch.object(launcher.os, "execvpe"))
        output = self.enterContext(mock.patch("sys.stdout", io.StringIO()))
        launch = threading.Thread(
            target=launcher.main, args=(["claude", "--port", str(port)],)
        )
        launch.start()
        lazy.wait_for(lambda: lazy.factory.processes, "no engine spawned")
        execute.assert_not_called()
        lazy.ready()
        launch.join(10)
        console_written()
        execute.assert_called_once()
        self.assertEqual(lazy.loads, 1)
        self.assertEqual(
            output.getvalue().splitlines()[:2],
            [
                "Loading test-model...",
                f"Starting claude: test-model · {READY.max_context_tokens:,} "
                "context tokens",
            ],
        )
