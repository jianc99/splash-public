"""Summaries of policy-bench runs (dev/benchmarks/policy_bench.mm): medians across runs, run-to-run spread, the
device plan against the fastest measured configuration per shape and rows, and per-step totals.

  python3 dev/benchmarks/policy_summary.py RUN.csv... [--new NEW.csv...] [--spread 0.03] [--json OUT.json]

RUN.csv are the alternating runs of one build. Each configuration's time is the median over the runs of its
per-run median; its spread is (max - min) / median over the runs. With --new, NEW.csv are runs of a candidate
build over the same options: its plan per shape and rows (policy = 1 rows) is timed from the base runs when they
measured that configuration (same interleaved samples), else from the new runs, and compared with the base plan.

Per-step totals weight each production shape by its count of projections per decode step; emulated widths
(`emulated` > 0) are totalled separately per emulated core count.
"""

import argparse
import collections
import csv
import json
import statistics
import sys

Key = collections.namedtuple("Key", "device cores suite shape rows label")


def load(paths):
    runs = collections.defaultdict(list)
    meta, policy = {}, {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["suite"] == "bandwidth":
                    continue
                key = Key(
                    row["device"],
                    int(row["cores"]),
                    row["suite"],
                    row["shape"],
                    int(row["rows"]),
                    row["label"],
                )
                runs[key].append(float(row["median_ms"]))
                shape = (row["device"], int(row["cores"]), row["suite"], row["shape"])
                meta[shape] = row
                if row["policy"] == "1":
                    policy[shape + (int(row["rows"]),)] = row["label"]
    times = {k: statistics.median(v) for k, v in runs.items()}
    spread = {
        k: (max(v) - min(v)) / statistics.median(v) if len(v) > 1 else 0.0
        for k, v in runs.items()
    }
    counts = {k: len(v) for k, v in runs.items()}
    return times, spread, counts, meta, policy


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--new", nargs="*", default=[])
    ap.add_argument("--spread", type=float, default=0.03)
    ap.add_argument("--json")
    a = ap.parse_args()
    times, spread, counts, meta, policy = load(a.runs)
    ntimes, nspread, _, _, npolicy = load(a.new) if a.new else ({}, {}, {}, {}, {})

    by_case = collections.defaultdict(dict)
    for k, t in times.items():
        by_case[(k.device, k.cores, k.suite, k.shape, k.rows)][k.label] = t
    cases, totals = [], collections.defaultdict(lambda: collections.Counter())
    for case in sorted(
        by_case, key=lambda c: (c[0], c[1], c[2], meta[c[:4]]["emulated"], c[3], c[4])
    ):
        labels = by_case[case]
        m = meta[case[:4]]
        base_label = policy.get(case)
        if base_label is None:
            continue
        best_label = min(labels, key=labels.get)
        base = labels[base_label]
        entry = {
            "device": case[0],
            "cores": case[1],
            "suite": case[2],
            "shape": case[3],
            "rows": case[4],
            "model": m["model"],
            "kind": m["kind"],
            "count": int(m["count"]),
            "emulated": int(m["emulated"]),
            "policy": base_label,
            "policy_ms": base,
            "best": best_label,
            "best_ms": labels[best_label],
            "spread": spread[Key(*case[:5], base_label)],
            "runs": counts[Key(*case[:5], base_label)],
        }
        if a.new:
            new_label = npolicy.get(case)
            if new_label is not None:
                new_ms = labels.get(new_label, ntimes.get(Key(*case[:5], new_label)))
                entry.update({"new": new_label, "new_ms": new_ms})
        cases.append(entry)
        if entry["count"] and case[2] in ("affine", "gguf"):
            group = (case[0], case[1], case[2], m["model"], int(m["emulated"]), case[4])
            totals[group]["policy"] += entry["count"] * base
            totals[group]["best"] += entry["count"] * entry["best_ms"]
            if "new_ms" in entry and entry["new_ms"] is not None:
                totals[group]["new"] += entry["count"] * entry["new_ms"]

    print(
        "| device | cores | shape | rows | plan | ms | best | best ms | plan/best | spread |"
        + (" new plan | new ms | new/plan |" if a.new else "")
    )
    print(
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|"
        + ("---|---:|---:|" if a.new else "")
    )
    flagged = 0
    for e in cases:
        flag = " !" if e["spread"] > a.spread else ""
        flagged += bool(flag)
        line = (
            f"| {e['device']} | {e['cores']} | {e['shape']} | {e['rows']} | {e['policy']} | {e['policy_ms']:.4f} | "
            f"{e['best']} | {e['best_ms']:.4f} | {e['policy_ms'] / e['best_ms']:.3f} | {e['spread']:.1%}{flag} |"
        )
        if a.new:
            if e.get("new_ms") is not None:
                line += f" {e['new']} | {e['new_ms']:.4f} | {e['new_ms'] / e['policy_ms']:.3f} |"
            else:
                line += " - | - | - |"
        print(line)
    print(f"\n{flagged} plans with a run-to-run spread above {a.spread:.0%} (!).\n")
    print(
        "| device | cores | suite | model | emulated | rows | plan ms/step | best ms/step | plan/best |"
        + (" new ms/step | new/plan |" if a.new else "")
    )
    print(
        "|---|---:|---|---|---:|---:|---:|---:|---:|" + ("---:|---:|" if a.new else "")
    )
    for g in sorted(totals):
        t = totals[g]
        line = (
            f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {g[4] or '-'} | {g[5]} | {t['policy']:.3f} | "
            f"{t['best']:.3f} | {t['policy'] / t['best']:.3f} |"
        )
        if a.new and t["new"]:
            line += f" {t['new']:.3f} | {t['new'] / t['policy']:.3f} |"
        print(line)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "cases": cases,
                    "totals": [
                        dict(
                            zip(
                                (
                                    "device",
                                    "cores",
                                    "suite",
                                    "model",
                                    "emulated",
                                    "rows",
                                ),
                                g,
                            ),
                            **totals[g],
                        )
                        for g in sorted(totals)
                    ],
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
