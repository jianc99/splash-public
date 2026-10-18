#!/usr/bin/env python3
"""stall-bench.py PORT OUTDIR SECONDS DOC [MODE]: staggered 4-lane starts against `splash serve` until SECONDS pass or the
engine fails. Each iteration: an essay (256 tokens) at 0 s, another (192) at 1 s, then two long-document prompts
(~8K and ~16K tokens, 64 tokens out) at 2 and 2.5 s, whose prefills overlap the essays' decode. Every prompt starts
with a fresh random tag (no prefix reuse). MODE "short" instead runs the failing bench's pattern: an essay (256
tokens) alone, then four essays (192 tokens each) at once. Writes OUTDIR/iterations.jsonl; stops at the first failed
request."""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

port, out, seconds, doc_path = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
mode = sys.argv[5] if len(sys.argv) > 5 else "long"
doc = open(doc_path, encoding="utf-8").read()
rng = random.Random(20261018)
model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]
TOPICS = ["the printing press", "ocean currents", "suspension bridges", "number systems", "coral reefs",
          "steam engines", "vaccines", "glaciers", "the silk road", "radio astronomy"]


def tag():
    return "[%08x] " % rng.getrandbits(32)


def ask(prompt, tokens, record):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            usage = json.load(response).get("usage", {})
            record.update(ok=True, prompt=usage.get("prompt_tokens"), completion=usage.get("completion_tokens"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        detail = ""
        if isinstance(error, urllib.error.HTTPError):
            try:
                detail = error.read().decode("utf-8", "replace")[:500]
            except OSError:
                pass
        record.update(ok=False, error=repr(error), detail=detail)
    record.update(start=start, seconds=time.time() - start)


end = time.time() + seconds
iteration = 0
with open(os.path.join(out, "iterations.jsonl"), "a") as log:
    while time.time() < end:
        iteration += 1
        at = rng.randrange(0, len(doc) - 70000)
        plan = [(0.0, tag() + "Write a long essay about " + rng.choice(TOPICS) + ".", 256),
                (1.0, tag() + "Write a detailed essay about " + rng.choice(TOPICS) + ".", 192),
                (2.0, tag() + "Summarize this document in one paragraph.\n\n" + doc[at:at + 31000], 64),
                (2.5, tag() + "List the main points of this document.\n\n" + doc[at:at + 62000], 64)]
        if mode == "short":
            single = {}
            ask(tag() + "Write a long, detailed essay about " + rng.choice(TOPICS) + ".", 256, single)
            if not single.get("ok"):
                print(f"single request FAILED {single.get('error')}", flush=True)
                sys.exit(1)
            plan = [(0.0, tag() + "Write a long, detailed essay about " + rng.choice(TOPICS) + ".", 192)
                    for _ in range(4)]
        records = [{} for _ in plan]
        threads = []
        t0 = time.time()
        for (delay, prompt, tokens), record in zip(plan, records):
            time.sleep(max(0.0, t0 + delay - time.time()))
            thread = threading.Thread(target=ask, args=(prompt, tokens, record))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        entry = {"iteration": iteration, "t0": t0, "wall": time.time() - t0, "requests": records}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        failed = [r for r in records if not r.get("ok")]
        print(f"[{time.strftime('%H:%M:%S')}] iteration {iteration}: {time.time() - t0:.1f} s"
              + (f" FAILED {failed[0].get('error')} {failed[0].get('detail', '')[:200]}" if failed else ""),
              flush=True)
        if failed:
            sys.exit(1)
print(f"no failure in {iteration} iterations", flush=True)
