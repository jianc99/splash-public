#!/usr/bin/env python3
"""Writes the KV selection traces that the hot block cache's replacement is
checked on (dev/tests/engine/slot_recycler_test.cpp): the blocks each layer of
a block-sparse target's decode selected, step by step, of real requests.

A raw trace is what a measurement build of Qwen3.8-Flash-Next wrote (a host
step after each attention layer's block lists, not part of Splash): records of
four little-endian uint32, a magic, the attention layer, the first position of
the step's verify rows and the entries, then the entries (block << 8 | a bit
per verify row that selects it); one record per layer and step of the decoding
request, whose steps follow each other, and a request's first record has a
position below the one before it. The committed traces are the P1 traces of
the HiSparse measurements (a novel at 126K and 252K tokens and SGLang's
sources at 126K, two questions each), layers 0 and 6 of each request's first
100 steps:

    python3 dev/tools/kv_selection_traces.py RAW.trace... \\
        dev/tests/fixtures/kv-selection-traces

With --layers all --steps 0 it writes every layer and step, to replay the whole
traces the same way.

A written trace (NAME.kvst, for RAW's NAME.trace) is the magic b"KVS1", then
unsigned LEB128 varints: the requests and the layers; per request its steps;
per step and layer the step's first position, its blocks' count and the blocks
in ascending order, the first as it is and each other as its distance from the
one before.
"""

import argparse
import struct
from pathlib import Path

MAGIC = 0x51534154
HEADER = struct.Struct("<4I")


def requests(raw: bytes) -> list[list[dict[int, tuple[int, list[int]]]]]:
    """Per request, its steps; per step, each layer's position and blocks."""
    result, offset, last = [], 0, None
    while offset + HEADER.size <= len(raw):
        magic, layer, first, count = HEADER.unpack_from(raw, offset)
        offset += HEADER.size
        if magic != MAGIC:
            raise ValueError(f"not a selection trace record at byte {offset}")
        entries = struct.unpack_from(f"<{count}I", raw, offset)
        offset += 4 * count
        if layer == 0:
            if last is None or first < last:
                result.append([])
            result[-1].append({})
            last = first
        result[-1][-1][layer] = (first, sorted(entry >> 8 for entry in entries))
    return result


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def encode(trace, layers: list[int], steps: int) -> bytes:
    out = bytearray(b"KVS1") + varint(len(trace)) + varint(len(layers))
    for request in trace:
        kept = request[:steps] if steps else request
        out += varint(len(kept))
        for step in kept:
            for layer in layers:
                first, blocks = step[layer]
                out += varint(first) + varint(len(blocks))
                previous = 0
                for block in blocks:
                    out += varint(block - previous)
                    previous = block
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", default="0,6", help="comma-separated, or all")
    parser.add_argument("--steps", type=int, default=100, help="0 for every step")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.raw:
        trace = requests(path.read_bytes())
        layers = (
            sorted(trace[0][0])
            if args.layers == "all"
            else [int(layer) for layer in args.layers.split(",")]
        )
        written = args.output / f"{path.stem}.kvst"
        written.write_bytes(encode(trace, layers, args.steps))
        print(f"{written}: {len(trace)} requests, {written.stat().st_size} bytes")


if __name__ == "__main__":
    main()
