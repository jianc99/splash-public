#!/usr/bin/env python3
"""Export an existing verified Q4 draft as a Splash DFlash2 draft folder.

Copies only draft weights, without requantizing them, into DESTINATION: the
folder of the shared draft repository named after the base model
(install/families.py DRAFTS), or a directory for --draft-model.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from install import legacy, models  # noqa: E402


def export(package, config_path, destination):
    config = models.read_json(config_path)
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise models.ModelError("expected the original DFlash2 configuration")
    layers = config.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0:
        raise models.ModelError("invalid draft layer count")
    manifest = legacy.validate_manifest(package / "manifest.json")
    # The package names the DFlash2 checkpoint its draft was converted from.
    source = manifest.get("upstream", {}).get("draft", {})
    if not isinstance(source.get("repo_id"), str) or not models.is_hex_digest(
        source.get("revision"), 40
    ):
        raise models.ModelError(
            "the package names no upstream draft repository and commit"
        )
    records = {item["path"]: item for item in manifest["artifacts"]}
    names = ["model.bin", *(f"layer-{i}.bin" for i in range(layers))]
    if missing := [n for n in names if "draft/" + n not in records]:
        raise models.ModelError(
            "the package lists no draft weight " + ", ".join(missing)
        )
    legacy.verify_artifacts(
        package, {"artifacts": [records["draft/" + n] for n in names]}, full=True
    )
    for name in names:
        # Each file begins as the native loader requires (DFlashDraft.cpp):
        # the magic, then layer-N.bin's index N and kind 0 (a decoder layer),
        # or model.bin's layer count and kind 1 (the shared projections).
        expected = (layers, 1) if name == "model.bin" else (int(name[6:-4]), 0)
        with (package / "draft" / name).open("rb") as stream:
            magic, *header = struct.unpack("<8sII", stream.read(16))
        if magic != models.DRAFT_LAYER_MAGIC.encode() or tuple(header) != expected:
            raise models.ModelError("incompatible draft layout: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            shutil.copyfile(package / "draft" / name, destination / name)
        config["splash"] = {
            "format": models.DRAFT_LAYER_MAGIC,
            "source": {"repo": source["repo_id"], "revision": source["revision"]},
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(f"Exported {layers} unchanged draft layers to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.package, args.config, args.destination)
