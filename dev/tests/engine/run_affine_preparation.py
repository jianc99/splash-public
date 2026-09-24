"""Small standalone affine checkpoints, an independent byte-layout oracle and
golden hashes of the prepared images."""

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

ALIGN = 16384

# SHA-256 of every prepared image of the two fixtures. A change means the
# prepared bytes changed: that needs a new preparation identity, so cached
# images of the old layout are never served.
GOLDEN = {
    "dense": {
        "target/layer-0.bin": "475768a2f36a25870f844e3c99d5cd1b85c00c8c3569f3f0abc3a1bffc14fd0b",
        "target/layer-1.bin": "228844988bdd3ecf8cd32395b2a1fd3820f4bd7f91c68683f5b633e1c1919d1f",
        "target/head.bin": "3dde9dffefd58b1b0bb06e445b4347eb2cf5412b203f1b2f6ebe00d681dc1c8f",
        "target/embedding.bin": "5bfea1223081c0b006d617ee0fc044517937a4cd9fe9503421537e8d8c459211",
    },
    "moe": {
        "target/layer-0.bin": "5b9926e5cbd89f8c772ed8308abc1e4dae140a67e2c44cddbd4ce8e688dcda61",
        "target/layer-1.bin": "76e65f55a4b9ed13103583bd44da12fe2dc4258584b82a7df3f0f7cb0aebae69",
        "target/head.bin": "25556e9c9b5c9629e2707cd4f90f82008a722cf82a64ec0dc97665ae242aefc6",
        "target/embedding.bin": "3623464c3b923b290f556b1a66612f9012c0e249d3d49982b13300003dd81edf",
    },
}


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

    def packed(parts, rows, columns):
        result = bytearray()
        for field, unit in (("weight", 32), ("scales", 2), ("biases", 2)):
            source = b"".join(tensors[name + "." + field][2] for name in parts)
            source = source.ljust(rows * columns // 64 * unit, b"\0")
            for tile in range(0, rows, 256):
                for group in range(columns // 64):
                    for row in range(tile, tile + 256):
                        offset = (row * (columns // 64) + group) * unit
                        result.extend(source[offset : offset + unit])
        return result

    expected = root / "expected"
    expected.mkdir()

    def image(name, magic, index, kind, sections):
        data = bytearray(struct.pack("<8sII", magic.encode(), index, kind))
        data.extend(bytes(ALIGN - len(data)))
        for section in sections:
            data.extend(section)
            data.extend(bytes((-len(data)) % ALIGN))
        (expected / name).write_bytes(data)

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
                add(g + "A_log", [4], data=bytes.fromhex("003f80bf00400000"))
            else:
                add(g + "A_log", [4], "F32", bytes(16))
                sections.append(struct.pack("<4f", -1, -1, -1, -1))
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
            projection(p + "mlp.gate", 256, 256, bits=8)
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection(p + "mlp.switch_mlp." + name, 256, 256, experts=256)
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection(p + "mlp.shared_expert." + name, 256, 256)
            projection(p + "mlp.shared_expert_gate", 1, 256, bits=8)
            continue
        for name, rows, columns in (
            ("gate_proj", 512, 256),
            ("up_proj", 512, 256),
            ("down_proj", 256, 512),
        ):
            name = p + "mlp." + name
            projection(name, rows, columns)
            sections.append(packed([name], rows, columns))
        image(f"layer-{layer}.bin", "MDFL0006", layer, layer, sections)
    norm = add("language_model.model.norm.weight", [256])
    projection("language_model.lm_head", 256, 256)
    image(
        "head.bin",
        "MDFL0002",
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
    header, payload = {}, bytearray()
    for name, (shape, dtype, data) in tensors.items():
        header[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )


def prepare(binary, metallib, root, kind):
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
    assert hashes == GOLDEN[kind], (kind, hashes)
    print(result.stdout.splitlines()[-1])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("metallib", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="splash-affine-preparation-") as directory:
        root = Path(directory) / "moe"
        root.mkdir()
        fixture(root, moe=True)
        prepare(args.binary, args.metallib, root, "moe")
        root = Path(directory) / "dense"
        root.mkdir()
        fixture(root)
        command = prepare(args.binary, args.metallib, root, "dense")
        # Raw checkpoints need a different normalization convention. Refuse
        # their unsanitized convolution layout before publishing any weights.
        source = root / "model.safetensors"
        data = source.read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + length])
        header["language_model.model.layers.0.linear_attn.conv1d.weight"]["shape"] = [
            512,
            1,
            4,
        ]
        encoded = json.dumps(header).encode()
        before = set((root / "cache").glob("*/weights"))
        source.write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + data[8 + length :]
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode != 0 and "conv1d.weight" in result.stderr, (
            result.stderr
        )
        assert set((root / "cache").glob("*/weights")) == before
        print("affine preparation: raw checkpoint rejected before conversion PASS")


if __name__ == "__main__":
    main()
