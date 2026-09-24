import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from install import models, upstream


class UpstreamTest(unittest.TestCase):
    def test_pairing_is_independent_of_quantizer_and_precision(self):
        for repo in (
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "someone/Qwen3.6-35B-A3B-8bit",
        ):
            self.assertEqual(
                upstream.match_family(repo).draft_model,
                "incoai/Qwen3.6-35B-A3B-DFlash2",
            )
        self.assertEqual(
            upstream.match_family("user/custom", ["Qwen/Qwen3.8-27B"]).draft_layers, 5
        )
        for name, bases in (
            ("user/unknown", []),
            ("user/Qwen3.8-27Bigger", []),
            ("user/Qwen3.8-27B-4bit", ["Qwen/Qwen3.6-35B-A3B"]),
        ):
            with self.subTest(name=name), self.assertRaises(models.ModelError):
                upstream.match_family(name, bases)

    def test_gguf_selection_never_guesses_a_quantization(self):
        files = {
            "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "Qwen3.6-35B-A3B-Q8_0.gguf",
            "mmproj-BF16.gguf",
        }
        self.assertEqual(
            upstream.select_gguf(files, "UD-Q4_K_M"), "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
        )
        self.assertEqual(upstream.select_vision(files), "mmproj-BF16.gguf")
        for variant in (None, "Q4", "missing"):
            with self.assertRaises(models.ModelError):
                upstream.select_gguf(files, variant)
        with self.assertRaises(models.ModelError):
            upstream.select_vision({"mmproj-Q8_0.gguf"})

    def test_configuration_is_checked_before_weight_downloads(self):
        family = upstream.FAMILIES[0]
        with self.assertRaises(models.ModelError):
            upstream.validate_config(
                {"text_config": {"model_type": family.text_type}}, family
            )

    def test_missing_metadata_never_falls_back_to_another_repository(self):
        required = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        }
        for weight in ("model.safetensors", "model.gguf"):
            for missing in (*sorted(required), None):
                with self.subTest(weight=weight, missing=missing):
                    with tempfile.TemporaryDirectory() as temporary:
                        args = argparse.Namespace(
                            model="user/Qwen3.6-35B-A3B-custom",
                            models=Path(temporary) / "models",
                            language_only=False,
                        )
                        source = mock.Mock()
                        source.base_models = []
                        source.files = {weight} | (
                            required - {missing} if missing else set()
                        )
                        with mock.patch.object(upstream, "Repository") as resolve:
                            with self.assertRaisesRegex(
                                models.ModelError,
                                "must come from the target repository",
                            ):
                                upstream.prepare(args, repo=source)
                            resolve.assert_not_called()
                        source.file.assert_not_called()
                        source.download.assert_not_called()
                        self.assertFalse(args.models.exists())

    def test_source_assembly_has_no_package_or_fixed_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target-repo"
            draft = root / "draft-repo"
            target.mkdir()
            (draft / "splash").mkdir(parents=True)
            family = upstream.FAMILIES[1]
            config = {
                "text_config": {
                    "model_type": family.text_type,
                    "hidden_size": family.hidden_size,
                    "num_hidden_layers": family.layers,
                    "vocab_size": 248320,
                },
                "quantization": {"bits": 4, "group_size": 64},
            }
            (target / "config.json").write_text(json.dumps(config))
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            ):
                (target / name).write_text("{}")
            (draft / "splash/config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "hidden_size": family.hidden_size,
                        "num_hidden_layers": family.draft_layers,
                        "splash": {"format": "MDFD0004"},
                    }
                )
            )
            for name in (
                "model.bin",
                *(f"layer-{i}.bin" for i in range(family.draft_layers)),
            ):
                (draft / "splash" / name).write_bytes(b"draft")
            args = argparse.Namespace(
                model="mlx-community/Qwen3.6-35B-A3B-4bit",
                models=root / "models",
                revision=None,
                language_only=True,
                draft_model=None,
            )
            source = upstream.Repository(target)
            draft_repo = upstream.Repository(draft)
            with mock.patch.object(
                upstream, "Repository", return_value=draft_repo
            ) as resolve:
                self.assertTrue(upstream.prepare(args, repo=source))
                resolve.assert_called_once_with(family.draft_model)
            installed = models.installed_root(
                args.models, args.model, language_only=True
            )
            record = upstream.verify(installed)
            for component in ("target", "config", "tokenizer"):
                self.assertEqual(record["sources"][component], source.identity())
            for name in ("tokenizer.json", "tokenizer_config.json"):
                self.assertEqual(
                    (installed / "tokenizer" / name).resolve(),
                    (target / name).resolve(),
                )
            self.assertEqual(record["vision_format"], "none")
            self.assertFalse((installed / "manifest.json").exists())
            self.assertFalse((installed / "vision").exists())
            self.assertEqual(
                (installed / "target/model.safetensors").resolve(),
                (target / "model.safetensors").resolve(),
            )
            self.assertEqual((installed / "draft/model.bin").read_bytes(), b"draft")
            with mock.patch.object(upstream, "Repository", return_value=draft_repo):
                upstream.prepare(args, repo=source)
            self.assertEqual(upstream.verify(installed), record)
            (target / "tokenizer.json").write_text("changed size")
            with self.assertRaises(models.ModelError):
                upstream.verify(installed)

    def test_verify_checks_full_content_and_rejects_same_size_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(b"abcd")
            assembly = root / "assembly"
            assembly.mkdir()
            (assembly / "weight").symlink_to(source)
            record = {"version": 1, "files": {"weight": upstream._file_record(source)}}
            (assembly / "model.json").write_text(json.dumps(record))
            upstream.verify(assembly, full=True)
            source.write_bytes(b"abce")
            with self.assertRaises(models.ModelError):
                upstream.verify(assembly)
            stat = source.stat()
            record["files"]["weight"].update(
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns
            )
            (assembly / "model.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(models.ModelError, "hash mismatch"):
                upstream.verify(assembly, full=True)

    def test_hub_resolution_uses_one_snapshot_without_requiring_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = SimpleNamespace(
                sha="a" * 40,
                siblings=[SimpleNamespace(rfilename="config.json")],
                card_data=None,
            )
            with (
                mock.patch("huggingface_hub.HfApi") as api,
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                    return_value=str(root / "config.json"),
                ) as file,
                mock.patch(
                    "huggingface_hub.snapshot_download", return_value=str(root)
                ) as snapshot,
            ):
                api.return_value.model_info.return_value = info
                repo = upstream.Repository("mlx-community/Qwen3.8-27B-4bit")
                api.return_value.model_info.assert_called_once_with(
                    repo.name, revision=None
                )
                repo.file("config.json")
                self.assertEqual(file.call_args.kwargs["revision"], info.sha)
                repo.download({"config.json"})
                self.assertEqual(snapshot.call_args.kwargs["revision"], info.sha)

    def test_selection_paths_do_not_conflict(self):
        root = Path("/models")
        repo = "mlx-community/Qwen3.8-27B-4bit"
        paths = {
            models.installed_root(root, repo),
            models.installed_root(root, repo, language_only=True),
            models.installed_root(root, repo, revision="old"),
            models.installed_root(root, repo, draft_model="mine/draft"),
        }
        self.assertEqual(len(paths), 4)
