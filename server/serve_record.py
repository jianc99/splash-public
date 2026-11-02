"""The serve record, `<runtime>/serve-PORT.json` beside the launcher's lock,
from which the desktop app words why a server it started did not start:
written by the server, atomically, as it starts and when its start fails."""

import json
import os
from pathlib import Path

from . import runtime as engine_runtime


def write(path: Path | None, *, pid: int, phase: str, failure: dict | None = None):
    """Replace the record at `path` (a sibling written, then renamed over it);
    nothing without a path. `phase` is "starting", which no stale record of
    this pid can be, or "failed", with `failure` (failure_of)."""
    if path is None:
        return
    record = {"pid": pid, "phase": phase}
    if failure is not None:
        record["failure"] = failure
    sibling = path.with_name(f".{path.name}.{pid}")
    sibling.write_text(json.dumps(record, separators=(",", ":")))
    os.replace(sibling, path)


def failure_of(error: BaseException) -> dict:
    """An engine's StartFailedEvent as the record's engine failure, with its
    kind, bound and numbers by name; any other error as a server failure."""
    if not isinstance(error, engine_runtime.EngineStartFailed):
        return {"source": "server", "kind": "server", "message": str(error)}
    event = error.event
    failure = {"source": "engine", "kind": event.kind.name.lower()}
    if event.required_bytes or event.budget_bytes or event.plan_context_tokens:
        failure |= {
            "bound": event.bound.name.lower(),
            "required_bytes": event.required_bytes,
            "budget_bytes": event.budget_bytes,
            "plan_context_tokens": event.plan_context_tokens,
        }
    return failure | {"message": str(error)}
