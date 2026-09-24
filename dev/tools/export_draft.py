#!/usr/bin/env python3
"""Export an existing verified Q4 draft as independent DFlash2 assets.

Copies only draft weights, without requantizing them. The destination is a
DFlash2 repository directory; its splash/ subdirectory holds native weights.
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
            if magic != b"MDFD0004" or layer != int(name[6:-4]) or kind != 0:
                raise models.ModelError("incompatible draft layout: " + name)
    output = destination / "splash"
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            shutil.copyfile(package / "draft" / name, output / name)
        config["splash"] = {"format": "MDFD0004"}
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Exported {layers} unchanged draft layers to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.package, args.config, args.destination)
