import argparse
import contextlib
import io
import shutil
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoTokenizer

from dev.tests.test_upstream import MOE, draft_dir
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
        "qwen35moe.full_attention_interval": 4,
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


def loadable_tensors(values, directory):
    """A tensor table the native loader accepts for the header values: each
    tensor it reads, quantized as Q4_K, Q8_0 (GDN alpha and beta) or F32."""
    codes = {name: code for code, name in gguf.TENSOR_TYPES.items()}
    header = gguf.Metadata(write_gguf(directory / "header.gguf", values))
    return {
        name: codes[next(t for t in ("Q4_K", "Q8_0", "F32") if t in types)]
        for name, types in gguf.loaded_tensors(header).items()
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

    def test_screening_accepts_each_tensor_as_the_native_loader_reads_it(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        tensors = loadable_tensors(values, self.root)
        # Layer 3 is full attention; the others around it GDN.
        self.assertIn("blk.3.attn_q.weight", tensors)
        self.assertNotIn("blk.2.attn_q.weight", tensors)
        # GDN alpha and beta may also both be F32, and the MTP layer (block
        # 40) is never loaded, so its types do not matter.
        tensors |= {"blk.1.ssm_alpha.weight": 0, "blk.1.ssm_beta.weight": 0}
        tensors |= {"blk.40.ffn_up_exps.weight": 16, "blk.0.ffn_down_exps.weight": 23}
        path = write_gguf(self.root / "ok.gguf", values, tensors.items())
        gguf.require_loadable(gguf.Metadata(path, tensors=True))
        f32 = {name: 0 for name in tensors}
        for changes, reason in (
            (
                {"blk.4.ffn_gate_exps.weight": 18},
                "ffn_gate_exps.weight IQ3_XXS [(]1 tensor[)]",
            ),
            (
                {"blk.0.attn_qkv.weight": 39, "blk.1.attn_qkv.weight": 39},
                "attn_qkv.weight MXFP4 [(]2 tensors[)]",
            ),
            ({"token_embd.weight": 23}, "token_embd.weight IQ4_XS"),
            ({"blk.3.attn_q.weight": 30}, "attn_q.weight BF16"),
            # F32 only where the loader reads floats: not a projection, not
            # a quantized router or norm, not half an alpha/beta pair.
            ({"blk.3.attn_q.weight": 0}, "attn_q.weight F32"),
            ({"blk.0.ffn_gate_inp.weight": 8}, "ffn_gate_inp.weight Q8_0"),
            ({"output_norm.weight": 1}, "output_norm.weight F16"),
            (
                {"blk.1.ssm_alpha.weight": 8},
                "ssm_alpha.weight and ssm_beta.weight of different types",
            ),
            # An all-F32 file, whose types the loader reads somewhere.
            (f32, "attn_output.weight F32 [(]10 tensors[)]"),
        ):
            with self.subTest(reason=reason):
                path = write_gguf(
                    self.root / "bad.gguf", values, (tensors | changes).items()
                )
                with self.assertRaisesRegex(
                    models.ModelError, "cannot load: .*" + reason
                ):
                    gguf.require_loadable(gguf.Metadata(path, tensors=True))
        # Every tensor the loader reads must be present.
        missing = {n: k for n, k in tensors.items() if n != "blk.7.attn_k.weight"}
        path = write_gguf(self.root / "bad.gguf", values, missing.items())
        with self.assertRaisesRegex(models.ModelError, "attn_k.weight missing"):
            gguf.require_loadable(gguf.Metadata(path, tensors=True))
        # Without tensors=True the reader never reads the tensor table.
        self.assertEqual(gguf.Metadata(path).tensors, {})

    def test_loadable_types_are_the_native_formats(self):
        # The installer's list must be the loader's own: kQuantFormats' GGML
        # types (runtime/metal/abi/QuantFormat.h).
        import re

        header = (
            Path(__file__).resolve().parents[2] / "runtime/metal/abi/QuantFormat.h"
        ).read_text()
        table = header.split("kQuantFormats[GGUF_FMT_COUNT] = {", 1)[1].split("};", 1)[
            0
        ]
        native = {gguf.TENSOR_TYPES[int(n)] for n in re.findall(r"\{(\d+),", table)}
        self.assertEqual(native, gguf.QUANTIZED_TYPES)
        self.assertLessEqual(gguf.EMBEDDING_TYPES, gguf.QUANTIZED_TYPES)

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

    def derived(self, models_root, path):
        """_gguf_metadata as installations call it, under the lock."""
        models_root.mkdir(parents=True, exist_ok=True)
        with models.installation_lock(models_root):
            return upstream._gguf_metadata(models_root, path, None)

    def test_vision_projector_is_chosen_by_its_header(self):
        tensors = (("v.blk.0.attn_qkv.weight", 30), ("v.patch_embd.weight", 0))

        def projectors(**files):
            root = self.root / "projectors"
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir()
            for name, (values, kinds) in files.items():
                write_gguf(root / (name + ".gguf"), values, kinds)
            return upstream.Repository.local_directory(root)

        bf16 = (vision_fixture(), tensors)
        f32 = (vision_fixture(), [(name, 0) for name, _ in tensors])
        f16 = (vision_fixture(), [(tensors[0][0], 1), tensors[1]])
        text = (fixture(), tensors)
        # Other publishers' names; F16 and non-vision files never count.
        repo = projectors(
            **{"mmproj-Model-bf16": bf16, "mmproj-Model-f16": f16, "mmproj-x": text}
        )
        name, header = upstream.select_vision(repo)
        self.assertEqual(name, "mmproj-Model-bf16.gguf")
        self.assertEqual(header.values["general.architecture"], "clip")
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"mmproj-f16": f16, "mmproj-f32": f32})
            )[0],
            "mmproj-f32.gguf",
        )
        with self.assertRaisesRegex(
            models.ModelError,
            r"no BF16 or F32 vision projector \(mmproj-f16.gguf \(clip: F16, F32\); "
            r"mmproj-x.gguf \(qwen35moe: BF16, F32\)\); use --language-only",
        ):
            upstream.select_vision(projectors(**{"mmproj-f16": f16, "mmproj-x": text}))
        with self.assertRaisesRegex(
            models.ModelError, "several BF16 vision projectors"
        ):
            upstream.select_vision(projectors(**{"mmproj-a": bf16, "mmproj-b": bf16}))
        with self.assertRaisesRegex(models.ModelError, "no mmproj"):
            upstream.select_vision(projectors())

    def test_metadata_cache_hit_integrity_and_atomic_failure(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        key, files = self.derived(cache, path)
        self.assertEqual(files["config.json"].parent.name, key)
        expected = files["tokenizer/tokenizer.json"].read_bytes()
        with mock.patch.object(
            gguf, "Metadata", side_effect=AssertionError("reparsed")
        ):
            self.assertEqual(self.derived(cache, path), (key, files))
        # A damaged entry is derived again, not left to block installation.
        for damage in (
            lambda: files["tokenizer/tokenizer.json"].write_text("corrupt"),
            lambda: files["config.json"].unlink(),
            lambda: (files["config.json"].parent / "files.json").write_text("{}"),
        ):
            damage()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.derived(cache, path), (key, files))
            self.assertIn("Deriving damaged GGUF metadata again", output.getvalue())
            self.assertEqual(files["tokenizer/tokenizer.json"].read_bytes(), expected)
        with mock.patch(
            "install.upstream.os.rename", side_effect=OSError("interrupted")
        ):
            with self.assertRaises(OSError):
                self.derived(self.root / "interrupted", path)
        self.assertEqual(list((self.root / "interrupted/.metadata").iterdir()), [])
        self.assertTrue(self.derived(self.root / "interrupted", path))

    def test_concurrent_preparation_derives_once_and_follows_the_source(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        with (
            mock.patch.object(
                gguf, "tokenizer_files", wraps=gguf.tokenizer_files
            ) as derive,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(lambda _: self.derived(cache, path), range(2)))
        # The lock serializes installations: the second reuses the entry.
        derive.assert_called_once()
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list((cache / ".metadata").iterdir())), 1)
        values = fixture()
        values["tokenizer.chat_template"] = "updated template"
        write_gguf(path, values)
        _, changed = self.derived(cache, path)
        self.assertNotEqual(changed, results[0][1])
        self.assertEqual(
            changed["tokenizer/chat_template.jinja"].read_text(), "updated template"
        )

    def test_gguf_only_repository_assembly_never_resolves_other_target_sources(self):
        target = self.root / "target"
        target.mkdir()
        values = fixture(native=True)
        write_gguf(
            target / "model-Q4_K_M.gguf",
            values,
            loadable_tensors(values, self.root).items(),
        )
        write_gguf(
            target / "mmproj-F32.gguf", vision_fixture(), [("v.patch_embd.weight", 0)]
        )
        # Conflicting sidecars must not override the selected GGUF's metadata.
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (target / name).write_text("invalid sidecar")
        source = upstream.Repository.local_directory(target)
        draft_repo = upstream.Repository.local_directory(
            draft_dir(self.root / "draft", MOE)
        )
        for language_only in (True, False):
            args = argparse.Namespace(
                model="unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
                models=self.root / "models",
                revision=None,
                language_only=language_only,
                draft_model=None,
            )
            with mock.patch.object(
                upstream.Repository, "resolve", side_effect=[source, draft_repo]
            ) as resolve:
                with mock.patch.object(
                    source, "download", wraps=source.download
                ) as download:
                    upstream.prepare(args)
                self.assertEqual(
                    resolve.call_args_list,
                    [
                        mock.call(
                            "unsloth/Qwen3.6-35B-A3B-GGUF", None, installation=mock.ANY
                        ),
                        mock.call(upstream.DRAFTS, MOE.draft.revision),
                    ],
                )
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
            self.assertFalse((root / "processor").exists())
        # An image normalization the server does not implement is rejected
        # from the projector's header, before any download.
        values = vision_fixture()
        values["clip.vision.image_mean"] = [0.48, 0.46, 0.41]
        write_gguf(target / "mmproj-F32.gguf", values, [("v.patch_embd.weight", 0)])
        args.models = self.root / "rejected"
        with (
            mock.patch.object(upstream.Repository, "resolve", return_value=source),
            mock.patch.object(source, "download") as download,
            self.assertRaisesRegex(models.ModelError, "vision preprocessing"),
        ):
            upstream.prepare(args)
        download.assert_not_called()

    def test_a_new_metadata_adapter_rebuilds_the_metadata_locally(self):
        target = self.root / "target"
        target.mkdir()
        values = fixture(native=True)
        write_gguf(
            target / "model-Q4_K_M.gguf",
            values,
            loadable_tensors(values, self.root).items(),
        )
        source = upstream.Repository.local_directory(target)
        draft = draft_dir(self.root / "draft", MOE)
        args = argparse.Namespace(
            model="unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
            models=self.root / "models",
            revision=None,
            language_only=True,
            draft_model=str(draft),
        )
        resolve = upstream.Repository.resolve

        def hub(name, *arguments, **options):
            # The target's Hub resolution; the draft is a local directory.
            if name == "unsloth/Qwen3.6-35B-A3B-GGUF":
                return source
            return resolve(name, *arguments, **options)

        with (
            mock.patch.object(upstream.Repository, "resolve", side_effect=hub),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            upstream.prepare(args)
        root = models.installed_root(
            args.models, args.model, language_only=True, draft_model=str(draft)
        )
        installed = upstream.verify(root)["metadata"]
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        with (
            mock.patch.object(upstream.gguf, "__file__", str(adapter)),
            # The unchanged target is re-assembled without the Hub.
            mock.patch.object(
                upstream.Repository, "resolve", side_effect=hub
            ) as resolved,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertTrue(upstream.prepare(args))
        resolved.assert_called_once_with(
            "unsloth/Qwen3.6-35B-A3B-GGUF", None, installation=mock.ANY
        )
        self.assertIn("the GGUF metadata adapter changed", output.getvalue())
        rebuilt = upstream.verify(root)["metadata"]
        self.assertNotEqual(rebuilt, installed)
        self.assertEqual((root / "config.json").resolve().parent.name, rebuilt)
