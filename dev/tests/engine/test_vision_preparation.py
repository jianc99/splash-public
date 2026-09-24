"""Prepare tiny MLX and GGUF vision towers and compare them with an independently
serialized packed file; check the exact-BF16 rule, the MLX cache identity and
that invalid sources fail with their message and publish nothing."""

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ALIGN = 16384
DTYPES = ("BF16", "F16", "F32")
GGML_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "BF16": 30}
VISION_SHARD = "model-00001-of-00002.safetensors"
TEXT_SHARD = "model-00002-of-00002.safetensors"
# SHA-256 of the file every fixture below prepares: the values do not depend
# on the source format or the dtypes. A change means the prepared bytes
# changed: that needs a new preparation identity, so cached files of the old
# layout are never served.
GOLDEN = "f1a165335c42384479f73d83e69146cfa46d4964d66ed13c663b158fed08cdd5"


def encode(values, dtype):
    if dtype == "BF16":
        return b"".join(struct.pack("<f", x)[2:] for x in values)
    code = {"F16": "e", "F32": "f", "F64": "d"}[dtype]
    return struct.pack("<" + code * len(values), *values)


def bfloat16(values):
    # The upper half of each value's F32 bits: the value itself when it is
    # exactly a BF16.
    return b"".join(struct.pack("<f", x)[2:] for x in values)


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def safetensors(path, tensors):
    header, data = {}, bytearray()
    for name, (shape, dtype, raw) in tensors.items():
        header[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def gguf(path, metadata, tensors):
    def field(value):
        if isinstance(value, str):
            return 8, string(value)
        if isinstance(value, bool):
            return 7, bytes([value])
        if isinstance(value, int):
            return 4, struct.pack("<I", value)
        if isinstance(value, float):
            return 6, struct.pack("<f", value)
        kind, _ = field(value[0])
        return 9, struct.pack("<IQ", kind, len(value)) + b"".join(
            field(x)[1] for x in value
        )

    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for name, value in metadata.items():
        kind, data = field(value)
        header.extend(string(name) + struct.pack("<I", kind) + data)
    data = bytearray()
    for name, (shape, dtype, raw) in tensors.items():
        header.extend(string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", GGML_TYPES[dtype], len(data)))
        data.extend(raw)
        data.extend(bytes(-len(data) % 32))
    header.extend(bytes(-len(header) % 32))
    path.write_bytes(header + data)


def fixture(root, source, shift=0, case=None):
    """Writes a tiny tower (depth 2, width 8, 2x2 patches) whose tensors cycle
    through BF16, F16 and F32, and the packed file it must prepare."""
    tensors = {}
    sections = []

    def add(mlx, gguf_name, rows, columns=1, padded_rows=None, padded_columns=None):
        index = len(sections)
        dtype = DTYPES[(index + shift) % 3]
        # Exact in every dtype: signed zero, infinity, 2^-24 (an F16
        # subnormal) and multiples of 1/64.
        values = [-0.0, math.inf, 2.0**-24] + [
            ((i * 7 + index * 11) % 255 - 127) / 64 for i in range(3, rows * columns)
        ]
        if case == "inexact-f32" and index == 0:
            values[5] = 1 + 2.0**-20
        if case == "inexact-f16" and index == 0:
            values[5] = 1 + 2.0**-10
        if index == 0:
            # One patch row is [frame, patch-row, patch-col, channel] in MLX and
            # [channel, patch-row, patch-col] per frame in GGUF. Packed rows are
            # [channel, frame, patch-row, patch-col].
            def at(row, frame, pixel, channel):
                return values[row * 24 + (frame * 4 + pixel) * 3 + channel]

            if source == "mlx":
                tensors["vision_tower." + mlx] = (
                    [rows, 2, 2, 2, 3],
                    dtype,
                    encode(values, dtype),
                )
            else:
                for frame, suffix in enumerate(("", ".1")):
                    frame_values = [
                        at(row, frame, pixel, channel)
                        for row in range(rows)
                        for channel in range(3)
                        for pixel in range(4)
                    ]
                    tensors[gguf_name + suffix] = (
                        [2, 2, 3, rows],
                        dtype,
                        encode(frame_values, dtype),
                    )
            values = [
                at(row, frame, pixel, channel)
                for row in range(rows)
                for channel in range(3)
                for frame in range(2)
                for pixel in range(4)
            ]
        else:
            name = "vision_tower." + mlx if source == "mlx" else gguf_name
            if columns == 1:
                shape = [rows]
            else:
                shape = [rows, columns] if source == "mlx" else [columns, rows]
            if case == "dtype" and mlx == "merger.linear_fc1.weight":
                dtype = "F64" if source == "mlx" else "Q4_0"
            raw = encode(values, "F32" if dtype == "Q4_0" else dtype)
            tensors[name] = (shape, dtype, raw)
        output = []
        for row in range(padded_rows or rows):
            output.extend(
                values[row * columns : (row + 1) * columns]
                if row < rows
                else [0.0] * columns
            )
            output.extend([0.0] * ((padded_columns or columns) - columns))
        sections.append(bfloat16(output))

    def affine(mlx, gguf_name, rows, columns, padded_rows=None, padded_columns=None):
        add(
            mlx + ".weight",
            gguf_name + ".weight",
            rows,
            columns,
            padded_rows,
            padded_columns,
        )
        add(mlx + ".bias", gguf_name + ".bias", rows, padded_rows=padded_rows)

    def norm(mlx, gguf_name):
        add(mlx + ".weight", gguf_name + ".weight", 8)
        add(mlx + ".bias", gguf_name + ".bias", 8)

    affine("patch_embed.proj", "v.patch_embd", 8, 24)
    add("pos_embed.weight", "v.position_embd.weight", 4, 8)
    for layer in range(2):
        m, g = f"blocks.{layer}.", f"v.blk.{layer}."
        norm(m + "norm1", g + "ln1")
        affine(m + "attn.qkv", g + "attn_qkv", 24, 8)
        affine(m + "attn.proj", g + "attn_out", 8, 8)
        norm(m + "norm2", g + "ln2")
        affine(m + "mlp.linear_fc1", g + "ffn_up", 10, 8, padded_rows=16)
        affine(m + "mlp.linear_fc2", g + "ffn_down", 8, 10, padded_columns=16)
    norm("merger.norm", "v.post_ln")
    affine("merger.linear_fc1", "mm.0", 32, 32)
    affine("merger.linear_fc2", "mm.2", 8, 32)
    first = next(iter(tensors))
    if case == "shape":
        shape, dtype, raw = tensors[first]
        tensors[first] = ([math.prod(shape)], dtype, raw)
    expected = bytearray(struct.pack("<8sII", b"MDFV0001", 2, 0))
    expected.extend(bytes(ALIGN - len(expected)))
    for section in sections:
        expected.extend(section)
        expected.extend(bytes(-len(expected) % ALIGN))
    (root / "expected.bin").write_bytes(expected)
    if source == "mlx":
        if case == "quantized":
            tensors["vision_tower.blocks.0.attn.qkv.scales"] = (
                [24, 1],
                "BF16",
                bytes(48),
            )
        safetensors(root / VISION_SHARD, tensors)
        text = {"language_model.model.norm.weight": ([8], "BF16", bytes(16))}
        safetensors(root / TEXT_SHARD, text)
        (root / "config.json").write_text("{}")
        return
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
    if case == "epsilon":
        metadata["clip.vision.attention.layer_norm_epsilon"] = 1e-5
    if case == "deepstack":
        metadata["clip.vision.is_deepstack_layers"] = [True, False]
    if case == "no-deepstack":
        del metadata["clip.vision.is_deepstack_layers"]
    if case == "unused":
        tensors["v.deepstack.0.fc1.weight"] = ([8, 8], "BF16", bytes(128))
    gguf(root / "mmproj.gguf", metadata, tensors)


def prepare(binary, directory, source, mode, expected=True):
    command = [binary, source, str(directory), "tiny", mode]
    if expected:
        command.append(str(directory / "expected.bin"))
    env = {**os.environ, "SPLASH_WEIGHT_CACHE": str(directory / "cache")}
    return subprocess.run(command, env=env, text=True, capture_output=True)


def published(directory):
    return list((directory / "cache").glob("*/weights"))


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="splash-vision-preparation-") as temp:
        root = Path(temp)
        for source in ("mlx", "gguf"):
            mlx = source == "mlx"
            file = VISION_SHARD if mlx else "mmproj.gguf"
            # Every tensor as BF16, F16 and F32; BF16 is copied and exact F32
            # or F16 values become their BF16 bits.
            for shift in range(3):
                directory = root / f"{source}-{shift}"
                directory.mkdir()
                fixture(directory, source, shift)
                result = prepare(binary, directory, source, "cold")
                assert result.returncode == 0, (source, shift, result.stderr)
                path = Path(result.stdout.split()[0])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == GOLDEN, (source, shift, digest)
            patch = (
                "vision_tower.patch_embed.proj.weight" if mlx else "v.patch_embd.weight"
            )
            merger = "vision_tower.merger.linear_fc1.weight" if mlx else "mm.0.weight"
            dtype = "F64" if mlx else "Q4_0"
            inexact = (
                f"vision tensor {patch} in {{}} is not exactly representable in BF16"
            )
            # The patch embedding is F32 with shift 2 and F16 with shift 1.
            rejected = {
                ("inexact-f32", 2): inexact,
                ("inexact-f16", 1): inexact,
                ("shape", 0): f"vision tensor shape mismatch: {patch}",
                ("dtype", 0): f"vision tensor {merger} in {{}} is {dtype}; "
                "preparation reads BF16, F16 or F32",
            }
            if mlx:
                rejected[("quantized", 0)] = (
                    "the MLX vision tower is quantized "
                    "(vision_tower.blocks.0.attn.qkv.scales in {}); "
                    "preparation needs BF16, F16 or F32 vision weights"
                )
            else:
                rejected |= {
                    ("epsilon", 0): "vision LayerNorm epsilon mismatch",
                    ("deepstack", 0): "vision deepstack layers are unsupported",
                    ("no-deepstack", 0): "vision metadata must list "
                    "clip.vision.is_deepstack_layers per block",
                    ("unused", 0): "mmproj tensors the vision tower does not use: "
                    "v.deepstack.0.fc1.weight ({})",
                }
            for (case, shift), message in rejected.items():
                directory = root / f"{source}-{case}"
                directory.mkdir()
                fixture(directory, source, shift, case)
                result = prepare(binary, directory, source, "cold")
                expected = message.format(directory / file)
                errors = result.stderr.strip().splitlines()
                assert result.returncode == 1 and errors[-1] == expected, (
                    source,
                    case,
                    result.stderr,
                )
                assert not published(directory), (source, case)

        # The MLX tower's identity is config.json and the shard holding it.
        directory = root / "mlx-0"
        cached = prepare(binary, directory, "mlx", "warm").stdout.split()
        safetensors(
            directory / TEXT_SHARD,
            {"language_model.model.norm.weight": ([8], "BF16", bytes(range(16)))},
        )
        result = prepare(binary, directory, "mlx", "warm")
        assert result.returncode == 0 and result.stdout.split() == cached, result.stderr
        (directory / "config.json").write_text('{"text_config": {}}')
        result = prepare(binary, directory, "mlx", "warm", expected=False)
        assert result.stderr.strip() == "unexpected warm conversion", result.stderr
        data = bytearray((directory / VISION_SHARD).read_bytes())
        data[-1] ^= 1
        (directory / VISION_SHARD).write_bytes(data)
        result = prepare(binary, directory, "mlx", "warm", expected=False)
        assert result.stderr.strip() == "unexpected warm conversion", result.stderr
        print(
            "Vision layouts from MLX and GGUF, exact BF16 conversion, MLX identity "
            "and rejected sources PASS"
        )


if __name__ == "__main__":
    main()
