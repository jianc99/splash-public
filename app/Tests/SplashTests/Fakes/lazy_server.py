# A launcher that serves as `splash serve --lazy-load` does: ready at once with the model not
# loaded, then loading it, a load that failed, or loaded, as the test signals with a file in the
# runtime's folder (`loading`, `failed`, `loaded`); `/v1/models` gives the model's context and
# reasoning only once it is loaded (ServerTests). It writes the options it was given to `argv`.
import json
import os
import runpy
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(sys.argv[sys.argv.index("--port") + 1])
(root / "argv").write_text(json.dumps(sys.argv[1:]))
runpy.run_path(str(Path(__file__).with_name("app_bound.py")))["end_with_app"]()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        loaded = (root / "loaded").exists()
        if self.path == "/v1/models":
            model = {"input_modalities": ["text", "image", "pdf"], "context_length": None, "reasoning": None}
            if loaded:
                model |= {"context_length": 32768, "reasoning": {"thinks": True, "efforts": ["low", "high"], "default": "high"}}
            return self.reply({"data": [model]})
        if self.path == "/status":
            if loaded:
                return self.reply({"ready": True, "weights": {"released": False}, "capabilities": {"max_context_tokens": 32768},
                                   "transport": {"ready": True, "unloaded": False, "recovering": False}})
            transport = {"ready": False, "unloaded": True, "recovering": False}
            if (root / "failed").exists():
                transport["error"] = "the engine ran out of memory loading the model"
            return self.reply({"ready": False, "weights": {"released": True, "restoring": (root / "loading").exists()},
                               "transport": transport})
        self.reply({})


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
path = root / "build/runtime" / f"serve-{port}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"pid": os.getpid(), "phase": "ready"}))
server.serve_forever()
