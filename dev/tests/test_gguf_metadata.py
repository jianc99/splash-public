import argparse
import json
import struct
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoTokenizer

from install import gguf, models, upstream


def write_gguf(path, values, tensors=()):
    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    def typed(value):
        if isinstance(value, str):
            return 8, string(value)
        if isinstance(value, bool):
            return 7, struct.pack("<?", value)
        if isinstance(value, int):
            return 4, struct.pack("<I", value)
        if isinstance(value, float):
            return 6, struct.pack("<f", value)
        if isinstance(value, list):
            kind = typed(value[0])[0] if value else 4
            return 9, struct.pack("<IQ", kind, len(value)) + b"".join(
                typed(x)[1] for x in value
            )
        raise AssertionError(value)

    data = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(values))
    for key, value in values.items():
        kind, encoded = typed(value)
        data += string(key) + struct.pack("<I", kind) + encoded
    for name, kind in tensors:
        data += string(name) + struct.pack("<IQIQ", 1, 256, kind, 0)
    path.write_bytes(data)
    return path


def fixture(*, native=False):
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens += ["ab", "<|endoftext|>", "<|im_end|>", "<think>"]
    types = [1] * 257 + [3, 3, 4]
    if native:
        padding = 248320 - len(tokens)
        tokens += [f"[unused{i}]" for i in range(padding)]
        types += [5] * padding
    return {
        "general.architecture": "qwen35moe",
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.block_count": 40,
        "qwen35moe.context_length": 262144,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 258,
        "tokenizer.ggml.bos_token_id": 257,
        "tokenizer.ggml.padding_token_id": 257,
        "tokenizer.chat_template": "{% for message in messages %}{{ message.content }}<|im_end|>{% endfor %}",
    }


def vision_fixture():
    return {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.use_gelu": True,
        "clip.vision.block_count": 27,
        "clip.vision.embedding_length": 1152,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4304,
        "clip.vision.projection_dim": 2048,
        "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.image_size": 768,
        "clip.vision.is_deepstack_layers": [False] * 27,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }


class GgufMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values):
        return gguf.Metadata(write_gguf(self.root / "model.gguf", values))

    def test_reader_stops_before_weights_and_checks_bounds(self):
        path = write_gguf(self.root / "small.gguf", {"key": "value"})
        expected = path.stat().st_size
        with path.open("ab") as stream:
            stream.write(b"tensor payload is not metadata")
        with mock.patch.object(gguf.Metadata, "MAX_BYTES", expected):
            self.assertEqual(gguf.Metadata(path).values, {"key": "value"})
        raw = path.read_bytes()[:expected]
        for size in range(len(raw)):
            path.write_bytes(raw[:size])
            with self.assertRaises(models.ModelError):
                gguf.Metadata(path)
        path.write_bytes(raw)
        with mock.patch.object(gguf.Metadata, "MAX_BYTES", 24):
            with self.assertRaisesRegex(models.ModelError, "size limit"):
                gguf.Metadata(path)
        path.write_bytes(raw[:4] + struct.pack(">I", 3) + raw[8:])
        with self.assertRaises(models.ModelError):
            gguf.Metadata(path)

    def test_reader_rejects_duplicate_keys_unknown_types_and_nested_arrays(self):
        path = write_gguf(self.root / "bad.gguf", {"key": "value"})
        raw = path.read_bytes()
        path.write_bytes(raw[:16] + struct.pack("<Q", 2) + raw[24:] * 2)
        with self.assertRaisesRegex(models.ModelError, "duplicate"):
            gguf.Metadata(path)
        for kind, data in (
            (99, b""),
            (9, struct.pack("<IQ", 9, 1)),
            (9, struct.pack("<IQ", 8, gguf.Metadata.MAX_ITEMS + 1)),
            (8, struct.pack("<Q", 1) + b"\xff"),
        ):
            path.write_bytes(raw[:35] + struct.pack("<I", kind) + data)
            with self.assertRaises(models.ModelError):
                gguf.Metadata(path)

    def test_bpe_ids_special_tokens_normalization_and_local_reload(self):
        metadata = self.metadata(fixture())
        files = gguf.tokenizer_files(metadata)
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with mock.patch(
            "socket.socket.connect", side_effect=AssertionError("network access")
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                self.root / "tokenizer", local_files_only=True
            )
        backend = Tokenizer.from_str(files["tokenizer/tokenizer.json"].decode())
        tokens = metadata.values["tokenizer.ggml.tokens"]
        self.assertEqual(
            backend.get_vocab(), {token: i for i, token in enumerate(tokens)}
        )
        self.assertEqual(
            tokenizer.encode("ab<think><|im_end|>", add_special_tokens=False),
            [256, 259, 258],
        )
        self.assertEqual(tokenizer.encode("ab"), [256])  # No implicit BOS/EOS.
        self.assertEqual(
            tokenizer.decode([256, 259, 258], skip_special_tokens=True), "ab<think>"
        )
        self.assertEqual(tokenizer.encode("e\u0301"), tokenizer.encode("é"))
        # Combining marks stay with letters; Qwen2's older pattern splits these.
        for word in ("क्", "a\u035c"):
            self.assertEqual(len(backend.pre_tokenizer.pre_tokenize_str(word)), 1)
        self.assertEqual(len(backend.pre_tokenizer.pre_tokenize_str("123")), 3)
        text = "日本語 中文 👩🏽‍💻 \n\t 0123"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "ab"}], tokenize=True, return_dict=False
            ),
            [256, 258],
        )
        self.assertEqual(
            files["tokenizer/chat_template.jinja"].decode(),
            metadata.values["tokenizer.chat_template"],
        )

    def test_invalid_tokenizer_metadata_is_rejected(self):
        for key, value in (
            ("tokenizer.ggml.pre", "unknown"),
            ("tokenizer.ggml.model", "llama"),
            ("tokenizer.ggml.tokens", ["a", "a"]),
            ("tokenizer.ggml.token_type", [1]),
            ("tokenizer.ggml.merges", ["a"]),
            ("tokenizer.ggml.merges", ["a absent"]),
            ("tokenizer.ggml.eos_token_id", 99999),
            ("tokenizer.ggml.add_bos_token", True),
            ("tokenizer.ggml.add_eos_token", True),
            ("tokenizer.chat_template", ""),
        ):
            with self.subTest(key=key, value=value):
                values = fixture()
                values[key] = value
                with self.assertRaises(models.ModelError):
                    gguf.tokenizer_files(self.metadata(values))

    def test_unloadable_tensor_types_are_rejected_from_the_header(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        loadable = {
            "token_embd.weight": 12,  # Q4_K
            "output.weight": 14,  # Q6_K
            "blk.0.ffn_down_exps.weight": 23,  # IQ4_XS
            "blk.0.attn_norm.weight": 0,  # F32
            # The MTP layer (block 40) is never loaded, so its type does not matter.
            "blk.40.ffn_up_exps.weight": 16,  # IQ2_XXS
        }
        path = write_gguf(self.root / "ok.gguf", values, loadable.items())
        gguf.require_loadable(gguf.Metadata(path, tensors=True))
        for changes, reason in (
            ({"blk.3.ffn_gate_exps.weight": 18}, "IQ3_XXS [(]1 tensors[)]"),
            (
                {"blk.0.attn_qkv.weight": 39, "blk.1.attn_qkv.weight": 39},
                "MXFP4 [(]2 tensors[)]",
            ),
            ({"token_embd.weight": 23}, "IQ4_XS"),
            ({"blk.0.attn_q.weight": 30}, "BF16"),
        ):
            with self.subTest(reason=reason):
                path = write_gguf(
                    self.root / "bad.gguf", values, (loadable | changes).items()
                )
                with self.assertRaisesRegex(
                    models.ModelError, "cannot load: .*" + reason
                ):
                    gguf.require_loadable(gguf.Metadata(path, tensors=True))
        # Without tensors=True the reader never reads the tensor table.
        self.assertEqual(gguf.Metadata(path).tensors, {})

    def test_loadable_types_are_the_native_formats(self):
        # The installer's list must be the loader's own: kQuantFormats' GGML types
        # plus F32 (runtime/metal/abi/QuantFormat.h).
        import re

        header = (
            Path(__file__).resolve().parents[2] / "runtime/metal/abi/QuantFormat.h"
        ).read_text()
        table = header.split("kQuantFormats[GGUF_FMT_COUNT] = {", 1)[1].split("};", 1)[
            0
        ]
        native = {gguf.TENSOR_TYPES[int(n)] for n in re.findall(r"\{(\d+),", table)}
        self.assertEqual(native | {"F32"}, gguf.LOADABLE_TYPES)
        self.assertLessEqual(gguf.EMBEDDING_TYPES, gguf.LOADABLE_TYPES)

    def test_moe_config_states_its_experts_and_identifies_the_family(self):
        config = gguf.model_config(self.metadata(fixture(native=True)))
        text = config["text_config"]
        self.assertEqual((text["num_experts"], text["num_experts_per_tok"]), (256, 8))
        self.assertEqual(upstream.family_for(config).name, "Qwen3.6-35B-A3B")

    def test_config_uses_metadata_and_subtracts_only_mtp_layers(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(text["num_hidden_layers"], 40)
        self.assertEqual(text["hidden_size"], 2048)
        self.assertEqual(text["max_position_embeddings"], 262144)
        values["qwen35moe.nextn_predict_layers"] = 41
        with self.assertRaises(models.ModelError):
            gguf.model_config(self.metadata(values))
        vision = self.metadata(vision_fixture())
        self.assertEqual(gguf.vision_config(vision)["num_position_embeddings"], 2304)
        upstream._validate_processor(gguf.processor_config(vision))

    def test_metadata_cache_hit_integrity_and_atomic_failure(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        files = upstream._gguf_metadata(cache, path, None)
        with mock.patch.object(
            gguf, "Metadata", side_effect=AssertionError("reparsed")
        ):
            self.assertEqual(upstream._gguf_metadata(cache, path, None), files)
        files["tokenizer/tokenizer.json"].write_text("corrupt")
        with self.assertRaisesRegex(models.ModelError, "metadata changed"):
            upstream._gguf_metadata(cache, path, None)
        with mock.patch(
            "install.upstream.os.rename", side_effect=OSError("interrupted")
        ):
            with self.assertRaises(OSError):
                upstream._gguf_metadata(self.root / "interrupted", path, None)
        self.assertEqual(list((self.root / "interrupted/.metadata").iterdir()), [])
        self.assertTrue(upstream._gguf_metadata(self.root / "interrupted", path, None))

    def test_concurrent_preparation_and_source_change(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        barrier = threading.Barrier(2)
        original = gguf.tokenizer_files

        def synchronized(metadata):
            barrier.wait(timeout=5)
            return original(metadata)

        with mock.patch.object(gguf, "tokenizer_files", side_effect=synchronized):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        lambda _: upstream._gguf_metadata(cache, path, None), range(2)
                    )
                )
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list((cache / ".metadata").iterdir())), 1)
        values = fixture()
        values["tokenizer.chat_template"] = "updated template"
        write_gguf(path, values)
        changed = upstream._gguf_metadata(cache, path, None)
        self.assertNotEqual(changed, results[0])
        self.assertEqual(
            changed["tokenizer/chat_template.jinja"].read_text(), "updated template"
        )

    def test_gguf_only_repository_assembly_never_resolves_other_target_sources(self):
        target = self.root / "target"
        target.mkdir()
        write_gguf(target / "model-Q4_K_M.gguf", fixture(native=True))
        write_gguf(target / "mmproj-F32.gguf", vision_fixture())
        # Conflicting sidecars must not override the selected GGUF's metadata.
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (target / name).write_text("invalid sidecar")
        draft = self.root / "draft/Qwen3.6-35B-A3B"
        draft.mkdir(parents=True)
        (draft / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "hidden_size": 2048,
                    "num_hidden_layers": 6,
                    "splash": {"format": "MDFD0004"},
                }
            )
        )
        for name in ("model.bin", *(f"layer-{i}.bin" for i in range(6))):
            (draft / name).write_bytes(b"draft")
        source = upstream.Repository(target)
        draft_repo = upstream.Repository(draft.parent)
        for language_only in (True, False):
            args = argparse.Namespace(
                model="unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
                models=self.root / "models",
                language_only=language_only,
            )
            with mock.patch.object(
                upstream, "Repository", return_value=draft_repo
            ) as resolve:
                with mock.patch.object(
                    source, "download", wraps=source.download
                ) as download:
                    upstream.prepare(args, repo=source)
                (moe,) = (f for f in upstream.FAMILIES if f.name == "Qwen3.6-35B-A3B")
                resolve.assert_called_once_with(upstream.DRAFTS, moe.draft.revision)
                expected = {"model-Q4_K_M.gguf"} | (
                    set() if language_only else {"mmproj-F32.gguf"}
                )
                download.assert_called_once_with(expected)
            root = models.installed_root(
                args.models, args.model, language_only=language_only
            )
            upstream.verify(root, full=True)
            config = models.read_json(root / "config.json")
            self.assertEqual(config["text_config"]["num_hidden_layers"], 40)
            self.assertEqual("vision_config" in config, not language_only)
            self.assertEqual((root / "vision").exists(), not language_only)
