"""Small standalone dense and MoE affine checkpoints, an independent
byte-layout oracle of every prepared image and their golden hashes."""

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dev.tests.fixture_files import (  # noqa: E402
    read_safetensors,
    safetensors_bytes,
    weight_file,
    write_safetensors,
)


def fixture(root, moe=False):
    tensors = {}
    quantization = {"bits": 4, "group_size": 64}

    def add(name, shape, dtype="BF16", data=None):
        size = math.prod(shape) * {"BF16": 2, "U32": 4, "F32": 4}[dtype]
        if data is None:
            # Byte i is (i * 31 + seed * 17) % 256, which repeats every 256.
            seed = len(tensors) + 1
            period = bytes((i * 31 + seed * 17) % 256 for i in range(256))
            data = (period * (size // 256 + 1))[:size]
        tensors[name] = (shape, dtype, data)
        return data

    def projection(name, rows, columns, bits=4, experts=1):
        lead = [experts] if experts > 1 else []
        add(name + ".weight", lead + [rows, columns * bits // 32], "U32")
        add(name + ".scales", lead + [rows, columns // 64])
        add(name + ".biases", lead + [rows, columns // 64])
        if bits != 4:
            quantization[name] = {"bits": bits, "group_size": 64}

    def packed(parts, rows, columns, bits=4, experts=1):
        # Per expert, each field's rows of the parts, then zero rows, in
        # [rows / 256][groups][256] tiles of the field's group bytes.
        result = bytearray()
        groups = columns // 64
        for expert in range(experts):
            for field, unit in (("weight", 8 * bits), ("scales", 2), ("biases", 2)):
                source = bytearray()
                for name in parts:
                    data = tensors[name + "." + field][2]
                    size = len(data) // experts
                    source += data[expert * size : (expert + 1) * size]
                source = source.ljust(rows * groups * unit, b"\0")
                for tile in range(0, rows, 256):
                    for group in range(groups):
                        for row in range(tile, tile + 256):
                            offset = (row * groups + group) * unit
                            result.extend(source[offset : offset + unit])
        return result

    expected = root / "expected"
    expected.mkdir()

    def image(name, magic, index, kind, sections):
        (expected / name).write_bytes(weight_file(magic, index, kind, sections))

    for layer in range(2):
        p = f"language_model.model.layers.{layer}."
        sections = [add(p + "input_layernorm.weight", [256])]
        if layer == 0:
            g = p + "linear_attn."
            names = [
                g + s for s in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            ]
            for name, rows in zip(names, (512, 256, 4, 4)):
                projection(name, rows, 256)
            sections.append(packed(names, 1024, 256))
            sections.append(add(g + "conv1d.weight", [512, 4, 1]))
            if moe:
                # BF16 decay logarithms: 0.5, -1, 2 and 0.
                logarithms = (0.5, -1, 2, 0)
                add(g + "A_log", [4], data=bytes.fromhex("003f80bf00400000"))
            else:
                logarithms = (0, 0, 0, 0)
                add(g + "A_log", [4], "F32", bytes(16))
            # The decay -exp(A_log), rounded once from double to F32.
            sections.append(struct.pack("<4f", *(-math.exp(x) for x in logarithms)))
            sections.append(add(g + "dt_bias", [4]))
            sections.append(add(g + "norm.weight", [64]))
            projection(g + "out_proj", 256, 256)
            sections.append(packed([g + "out_proj"], 256, 256))
        else:
            a = p + "self_attn."
            names = [a + s for s in ("q_proj", "k_proj", "v_proj")]
            for name, rows in zip(names, (512, 128, 128)):
                projection(name, rows, 256)
            sections.append(packed(names, 768, 256))
            sections.append(add(a + "q_norm.weight", [64]))
            sections.append(add(a + "k_norm.weight", [64]))
            projection(a + "o_proj", 256, 256)
            sections.append(packed([a + "o_proj"], 256, 256))
        sections.append(add(p + "post_attention_layernorm.weight", [256]))
        if moe:
            # The 8-bit router and shared-expert scalar gate; the gate's one row is
            # padded to a 256-row tile.
            projection(p + "mlp.gate", 256, 256, bits=8)
            sections.append(packed([p + "mlp.gate"], 256, 256, bits=8))
            for name in ("gate_proj", "up_proj", "down_proj"):
                name = p + "mlp.switch_mlp." + name
                projection(name, 256, 256, experts=256)
                sections.append(packed([name], 256, 256, experts=256))
            for name in ("gate_proj", "up_proj", "down_proj"):
                name = p + "mlp.shared_expert." + name
                projection(name, 256, 256)
                sections.append(packed([name], 256, 256))
            projection(p + "mlp.shared_expert_gate", 1, 256, bits=8)
            sections.append(packed([p + "mlp.shared_expert_gate"], 256, 256, bits=8))
        else:
            for name, rows, columns in (
                ("gate_proj", 512, 256),
                ("up_proj", 512, 256),
                ("down_proj", 256, 512),
            ):
                name = p + "mlp." + name
                projection(name, rows, columns)
                sections.append(packed([name], rows, columns))
        magic = "MDFM0001" if moe else "MDFL0006"
        image(f"layer-{layer}.bin", magic, layer, layer, sections)
    norm = add("language_model.model.norm.weight", [256])
    projection("language_model.lm_head", 256, 256)
    image(
        "head.bin",
        "MDFM0002" if moe else "MDFL0002",
        2,
        2,
        [norm, packed(["language_model.lm_head"], 256, 256)],
    )
    projection("language_model.model.embed_tokens", 256, 256)
    image(
        "embedding.bin",
        "MDFE0001",
        256,
        256,
        [
            tensors["language_model.model.embed_tokens." + field][2]
            for field in ("weight", "scales", "biases")
        ],
    )
    config = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "vocab_size": 256,
        "head_dim": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "intermediate_size": 512,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    }
    if moe:
        del config["intermediate_size"]
        config |= {
            "model_type": "qwen3_5_moe_text",
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
        }
    (root / "config.json").write_text(
        json.dumps({"text_config": config, "quantization": quantization})
    )
    write_safetensors(root / "model.safetensors", tensors)


def prepare(binary, metallib, root, kind, golden):
    command = [str(binary.resolve()), str(metallib.resolve()), str(root), kind]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    hashes = {
        name: digest
        for marker, name, digest in (
            line.split()
            for line in result.stdout.splitlines()
            if line.startswith("prepared ")
        )
    }
    assert hashes == golden, (kind, hashes)
    print(result.stdout.splitlines()[-1])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("goldens", type=Path)
    args = parser.parse_args()
    goldens = json.loads(args.goldens.read_text())["affine_images"]
    with tempfile.TemporaryDirectory(prefix="splash-affine-preparation-") as directory:
        root = Path(directory) / "moe"
        root.mkdir()
        fixture(root, moe=True)
        prepare(args.binary, args.metallib, root, "moe", goldens["moe"])
        root = Path(directory) / "dense"
        root.mkdir()
        fixture(root)
        command = prepare(args.binary, args.metallib, root, "dense", goldens["dense"])
        # Raw checkpoints need a different normalization convention. Refuse
        # their unsanitized convolution layout before publishing any weights.
        source = root / "model.safetensors"
        header, payload = read_safetensors(source)
        header["language_model.model.layers.0.linear_attn.conv1d.weight"]["shape"] = [
            512,
            1,
            4,
        ]
        before = set((root / "cache").glob("*/weights"))
        source.write_bytes(safetensors_bytes(header, payload))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode != 0 and "conv1d.weight" in result.stderr, (
            result.stderr
        )
        assert set((root / "cache").glob("*/weights")) == before
        print("affine preparation: raw checkpoint rejected before conversion PASS")


if __name__ == "__main__":
    main()
