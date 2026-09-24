"""Exercise both vision readers against independently serialized tensor layouts."""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ALIGN = 16384


def packed(values, dtype):
    if dtype == "BF16":
        return b"".join(struct.pack("<f", x)[2:] for x in values)
    return struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)


def decoded(data, dtype):
    if dtype == "BF16":
        return [
            struct.unpack("<f", b"\0\0" + data[i : i + 2])[0]
            for i in range(0, len(data), 2)
        ]
    return list(
        struct.unpack(
            "<"
            + ("f" if dtype == "F32" else "e")
            * (len(data) // (4 if dtype == "F32" else 2)),
            data,
        )
    )


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(root, source, corrupt=None):
    tensors = {}
    sections = []

    def add(
        mlx, gguf, rows, columns=1, padded_rows=None, padded_columns=None, patch=False
    ):
        index = len(sections)
        dtype = "BF16" if source == "mlx" else ("F32", "BF16", "F16")[index % 3]
        if corrupt == "dtype" and (
            (source == "mlx" and index == 0)
            or (source == "gguf" and gguf == "mm.0.weight")
        ):
            dtype = "F32" if source == "mlx" else "Q4_0"
        storage = (
            "BF16"
            if source == "mlx"
            or (dtype == "BF16" and columns > 1 and "position" not in gguf)
            else "F32"
        )
        # Deliberately non-BF16 values, not a sequence of exactly representable integers.
        values = [
            ((i * 7 + index * 11) % 257 - 128) / 1031 for i in range(rows * columns)
        ]
        raw = packed(values, "F32" if dtype == "Q4_0" else dtype)
        rounded = decoded(raw, "F32" if dtype == "Q4_0" else dtype)
        if patch:
            # Oracle output order is [output, channel, temporal, patch-row, patch-col].
            if source == "mlx":
                tensors["vision_tower." + mlx] = ([rows, 2, 2, 2, 3], dtype, raw)
                converted = [
                    rounded[r * 24 + t * 12 + pixel * 3 + channel]
                    for r in range(rows)
                    for channel in range(3)
                    for t in range(2)
                    for pixel in range(4)
                ]
            else:
                # GGUF has one [output, channel, patch-row, patch-col] tensor per frame.
                tensors[gguf] = (
                    [2, 2, 3, rows],
                    dtype,
                    packed(rounded[: rows * 12], "F32" if dtype == "Q4_0" else dtype),
                )
                tensors[gguf + ".1"] = (
                    [2, 2, 3, rows],
                    dtype,
                    packed(rounded[rows * 12 :], "F32" if dtype == "Q4_0" else dtype),
                )
                converted = [
                    rounded[t * rows * 12 + r * 12 + channel * 4 + pixel]
                    for r in range(rows)
                    for channel in range(3)
                    for t in range(2)
                    for pixel in range(4)
                ]
        else:
            shape = (
                [rows]
                if columns == 1
                else [rows, columns]
                if source == "mlx"
                else [columns, rows]
            )
            tensors[("vision_tower." + mlx) if source == "mlx" else gguf] = (
                shape,
                dtype,
                raw,
            )
            converted = rounded
        if corrupt == "shape" and index == 0:
            name = next(iter(tensors))
            shape, dt, data = tensors[name]
            tensors[name] = ([math.prod(shape)], dt, data)
        output = []
        for row in range(padded_rows or rows):
            output.extend(
                converted[row * columns : (row + 1) * columns]
                if row < rows
                else [0] * columns
            )
            output.extend([0] * ((padded_columns or columns) - columns))
        sections.append(packed(output, storage))

    def affine(mlx, gguf, rows, columns, pr=None, pc=None, patch=False):
        add(mlx + ".weight", gguf + ".weight", rows, columns, pr, pc, patch)
        add(mlx + ".bias", gguf + ".bias", rows, padded_rows=pr)

    def norm(mlx, gguf):
        add(mlx + ".weight", gguf + ".weight", 8)
        add(mlx + ".bias", gguf + ".bias", 8)

    affine("patch_embed.proj", "v.patch_embd", 8, 24, patch=True)
    add("pos_embed.weight", "v.position_embd.weight", 4, 8)
    for layer in range(2):
        m, g = f"blocks.{layer}.", f"v.blk.{layer}."
        norm(m + "norm1", g + "ln1")
        affine(m + "attn.qkv", g + "attn_qkv", 24, 8)
        affine(m + "attn.proj", g + "attn_out", 8, 8)
        norm(m + "norm2", g + "ln2")
        affine(m + "mlp.linear_fc1", g + "ffn_up", 10, 8, pr=16)
        affine(m + "mlp.linear_fc2", g + "ffn_down", 8, 10, pc=16)
    norm("merger.norm", "v.post_ln")
    affine("merger.linear_fc1", "mm.0", 32, 32)
    affine("merger.linear_fc2", "mm.2", 8, 32)
    expected = bytearray(struct.pack("<8sII", b"MDFV0001", 2, int(source == "gguf")))
    expected.extend(bytes(ALIGN - len(expected)))
    for section in sections:
        expected.extend(section)
        expected.extend(bytes(-len(expected) % ALIGN))
    (root / "expected.bin").write_bytes(expected)
    if source == "mlx":
        header, data = {}, bytearray()
        for name, (shape, dtype, raw) in tensors.items():
            header[name] = {
                "shape": shape,
                "dtype": dtype,
                "data_offsets": [len(data), len(data) + len(raw)],
            }
            data.extend(raw)
        encoded = json.dumps(header).encode()
        (root / "model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + data
        )
        (root / "config.json").write_text("{}")
    else:
        metadata = {
            "general.architecture": "clip",
            "clip.projector_type": "qwen3vl_merger",
            "clip.vision.projection_dim": 8,
            "clip.vision.patch_size": 2,
            "clip.vision.embedding_length": 8,
            "clip.vision.feed_forward_length": 10,
            "clip.vision.block_count": 2,
            "clip.vision.attention.head_count": 2,
            "clip.vision.spatial_merge_size": 2,
            "clip.use_gelu": True,
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.image_mean": [0.5] * 3,
            "clip.vision.image_std": [0.5] * 3,
            "clip.vision.is_deepstack_layers": [False] * 2,
            # Unknown metadata must not make an otherwise supported projector fail.
            "clip.unused": ["ignored"],
        }
        if corrupt == "epsilon":
            metadata["clip.vision.attention.layer_norm_epsilon"] = 1e-5
        if corrupt == "deepstack":
            metadata["clip.vision.is_deepstack_layers"] = [True, False]

        def field(value):
            if isinstance(value, str):
                return 8, string(value)
            if isinstance(value, bool):
                return 7, bytes([value])
            if isinstance(value, int):
                return 4, struct.pack("<I", value)
            if isinstance(value, float):
                return 6, struct.pack("<f", value)
            dt, _ = field(value[0])
            return 9, struct.pack("<IQ", dt, len(value)) + b"".join(
                field(x)[1] for x in value
            )

        header = bytearray(
            struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
        )
        for name, value in metadata.items():
            dt, data = field(value)
            header.extend(string(name) + struct.pack("<I", dt) + data)
        data = bytearray()
        for name, (shape, dtype, raw) in tensors.items():
            header.extend(
                string(name)
                + struct.pack("<I", len(shape))
                + struct.pack("<" + "Q" * len(shape), *shape)
            )
            header.extend(
                struct.pack(
                    "<IQ", {"F32": 0, "F16": 1, "BF16": 30, "Q4_0": 2}[dtype], len(data)
                )
            )
            data.extend(raw)
            data.extend(bytes(-len(data) % 32))
        header.extend(bytes(-len(header) % 32))
        (root / "mmproj.gguf").write_bytes(header + data)


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="splash-vision-preparation-") as temp:
        root = Path(temp)
        for source in ("mlx", "gguf"):
            for case in (
                None,
                "shape",
                "dtype",
                *(("epsilon", "deepstack") if source == "gguf" else ()),
            ):
                directory = root / f"{source}-{case}"
                directory.mkdir()
                fixture(directory, source, case)
                env = {**os.environ, "SPLASH_WEIGHT_CACHE": str(directory / "cache")}
                result = subprocess.run(
                    [
                        binary,
                        source,
                        str(directory),
                        "tiny",
                        str(directory / "expected.bin"),
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                )
                if case:
                    assert result.returncode != 0, (source, case, result.stdout)
                    assert not list((directory / "cache").glob("*/weights")), (
                        source,
                        case,
                    )
                else:
                    assert result.returncode == 0, result.stderr
        print(
            "Vision source layouts, dtype preservation, warm reuse and invalid metadata PASS"
        )


if __name__ == "__main__":
    main()
