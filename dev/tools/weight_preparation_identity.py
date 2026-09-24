#!/usr/bin/env python3
"""Fingerprint the code defining prepared bytes, independently of app releases."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .build_identity import source_digest, update_if_changed
else:
    from build_identity import source_digest, update_if_changed

INPUTS = {
    "AFFINE": (
        "runtime/model/AffineTarget.cpp",
        "runtime/model/SafetensorsCheckpoint.mm",
        "runtime/model/WeightStore.hpp",
    ),
    "GGUF": (
        "runtime/model/GgufPreparation.cpp",
        "runtime/model/GgufImage.cpp",
        "runtime/model/GgufImage.hpp",
        "runtime/metal/abi/Gguf.h",
        "runtime/metal/abi/QuantFormat.h",
        "runtime/metal/kernels/shared/gguf_repack.metal",
    ),
    "VISION": (
        "runtime/model/VisionPreparation.cpp",
        "runtime/model/SafetensorsCheckpoint.mm",
        "runtime/model/GgufFile.cpp",
        "runtime/model/GgufFile.hpp",
        "runtime/model/WeightStore.hpp",
    ),
}


def header(root: Path) -> bytes:
    lines = ["// Generated preparation identities; do not edit.", "#pragma once"]
    for name, inputs in INPUTS.items():
        digest = source_digest(root, inputs)
        lines.append(f'#define SPLASH_{name}_PREPARATION_ID "{digest}"')
    return ("\n".join(lines) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--header", type=Path)
    parser.add_argument("--inputs", action="store_true")
    args = parser.parse_args()
    if args.inputs:
        print(" ".join(sorted({path for paths in INPUTS.values() for path in paths})))
    elif args.header is not None:
        update_if_changed(args.header, header(args.root))
    else:
        parser.error("--header or --inputs is required")


if __name__ == "__main__":
    main()
