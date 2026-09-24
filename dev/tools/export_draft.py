#!/usr/bin/env python3
"""Export an existing verified Q4 draft as a Splash DFlash2 draft folder.

Copies only draft weights, without requantizing them, into DESTINATION: the
folder of the shared draft repository named after the base model
(install/upstream.py DRAFTS), or a directory for --draft-model.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from install import models  # noqa: E402


def export(package, config_path, destination):
    config = models.read_json(config_path)
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise models.ModelError("expected the original DFlash2 configuration")
    layers = config.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0:
        raise models.ModelError("invalid draft layer count")
    manifest = models.validate_package_manifest(package / "manifest.json")
    records = {item["path"]: item for item in manifest["artifacts"]}
    names = ["model.bin", *(f"layer-{i}.bin" for i in range(layers))]
    for name in names:
        source = package / "draft" / name
        record = records.get("draft/" + name)
        if (
            not record
            or source.stat().st_size != record["size"]
            or models.sha256(source) != record["sha256"]
        ):
            raise models.ModelError("unverified draft weight: " + name)
        if name.startswith("layer-"):
            with source.open("rb") as stream:
                magic, layer, kind = struct.unpack("<8sII", stream.read(16))
            if (
                magic != models.DRAFT_LAYER_MAGIC.encode()
                or layer != int(name[6:-4])
                or kind != 0
            ):
                raise models.ModelError("incompatible draft layout: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            shutil.copyfile(package / "draft" / name, destination / name)
        # The package names the DFlash2 checkpoint its draft was converted from.
        source = manifest.get("upstream", {}).get("draft", {})
        config["splash"] = {
            "format": models.DRAFT_LAYER_MAGIC,
            "source": {
                "repo": source.get("repo_id"),
                "revision": source.get("revision"),
            },
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
