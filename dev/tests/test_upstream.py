import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from install import models, upstream

DENSE, MOE = upstream.FAMILIES


def text_config(family, **changes):
    return dict(family.signature) | changes


def mlx_target(root, family, *, changes=None):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": text_config(family, **(changes or {})),
        "quantization": {"bits": 4, "group_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (root / name).write_text("{}")
    return root


def draft_dir(root, family):
    folder = root / family.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": dict(family.signature)["hidden_size"],
                "num_hidden_layers": family.draft.layers,
                "splash": {"format": "MDFD0004"},
            }
        )
    )
    for name in ("model.bin", *(f"layer-{i}.bin" for i in range(family.draft.layers))):
        (folder / name).write_bytes(b"draft")
    return root


def hub_repository(cache, repo_id, commit, build):
    """A Repository whose files sit in a Hub-cache snapshot, as a download leaves them."""
    snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / commit
    build(snapshot)
    repo = upstream.Repository(snapshot)
    repo.name, repo.revision, repo.local = repo_id, commit, False
    return repo


def arguments(root, model, **options):
    return argparse.Namespace(
        model=model,
        models=root / "models",
        revision=options.get("revision"),
        language_only=options.get("language_only", True),
        draft_model=options.get("draft_model"),
        update=options.get("update", False),
    )


class UpstreamTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_family_is_identified_by_architecture_not_name(self):
        for family in upstream.FAMILIES:
            self.assertIs(
                upstream.family_for({"text_config": text_config(family)}), family
            )
        # A differing field is another architecture, whatever the repository is called.
        for changes in (
            {"num_hidden_layers": 48},
            {"vocab_size": 151936},
            {"head_dim": 128},
            {"model_type": "qwen3_moe"},
            {"num_experts": 128},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(
                    models.ModelError, "no supported model has this architecture"
                ),
            ):
                family = MOE if "num_experts" in changes else DENSE
                upstream.family_for({"text_config": text_config(family, **changes)})
        with self.assertRaises(models.ModelError):
            upstream.family_for({"hidden_size": 5120})

    def test_gguf_selection_is_exact_and_ignores_subfolders(self):
        files = {
            "Qwen3.8-27B-UD-Q4_K_M.gguf",
            "Qwen3.8-27B-Q4_0.gguf",
            "Qwen3.8-27B-Q8_0.gguf",
            "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
            "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        }
        self.assertEqual(
            upstream.select_gguf(files, "UD-Q4_K_M"), "Qwen3.8-27B-UD-Q4_K_M.gguf"
        )
        self.assertEqual(upstream.select_gguf(files, "q4_0"), "Qwen3.8-27B-Q4_0.gguf")
        # Q4_K_M names the UD file when it is the only one; a plain file wins over it.
        self.assertEqual(
            upstream.select_gguf(files, "Q4_K_M"), "Qwen3.8-27B-UD-Q4_K_M.gguf"
        )
        both = files | {"Qwen3.8-27B-Q4_K_M.gguf"}
        self.assertEqual(
            upstream.select_gguf(both, "Q4_K_M"), "Qwen3.8-27B-Q4_K_M.gguf"
        )
        self.assertEqual(upstream.select_vision(files), "mmproj-BF16.gguf")
        for variant in (None, "Q4", "BF16", "missing"):
            with (
                self.subTest(variant=variant),
                self.assertRaisesRegex(models.ModelError, "Qwen3.8-27B-Q8_0.gguf"),
            ):
                upstream.select_gguf(files, variant)
        with self.assertRaisesRegex(models.ModelError, "--language-only"):
            upstream.select_vision({"mmproj-Q8_0.gguf"})

    def test_architecture_is_checked_before_weight_downloads(self):
        target = mlx_target(
            self.root / "target", DENSE, changes={"num_hidden_layers": 48}
        )
        source = upstream.Repository(target)
        with (
            mock.patch.object(source, "download") as download,
            mock.patch.object(upstream, "Repository") as resolve,
            self.assertRaisesRegex(models.ModelError, "no supported model"),
        ):
            upstream.prepare(arguments(self.root, "someone/renamed-27b"), repo=source)
        download.assert_not_called()
        resolve.assert_not_called()

    def test_missing_metadata_never_falls_back_to_another_repository(self):
        required = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        }
        for missing in (*sorted(required), None):
            with self.subTest(missing=missing):
                args = arguments(self.root, "user/custom", language_only=False)
                source = mock.Mock()
                source.files = {"model.safetensors"} | (
                    required - {missing} if missing else set()
                )
                with mock.patch.object(upstream, "Repository") as resolve:
                    with self.assertRaisesRegex(
                        models.ModelError, "must come from the target repository"
                    ):
                        upstream.prepare(args, repo=source)
                    resolve.assert_not_called()
                source.file.assert_not_called()
                source.download.assert_not_called()
                self.assertFalse(args.models.exists())

    def test_source_assembly_pairs_the_draft_by_architecture(self):
        target = mlx_target(self.root / "target", MOE)
        source = upstream.Repository(target)
        draft = upstream.Repository(draft_dir(self.root / "draft", MOE))
        # The name says nothing about the model; the configuration does.
        args = arguments(self.root, "someone/my-favourite-model")
        with mock.patch.object(upstream, "Repository", return_value=draft) as resolve:
            self.assertTrue(upstream.prepare(args, repo=source))
        resolve.assert_called_once_with(upstream.DRAFTS, MOE.draft.revision)
        installed = models.installed_root(args.models, args.model, language_only=True)
        record = upstream.verify(installed)
        self.assertEqual(record["family"], MOE.name)
        self.assertEqual(
            record["sources"], {"target": source.identity(), "draft": draft.identity()}
        )
        self.assertEqual(record["vision_format"], "none")
        self.assertFalse((installed / "manifest.json").exists())
        self.assertFalse((installed / "vision").exists())
        for name in ("tokenizer.json", "tokenizer_config.json"):
            self.assertEqual(
                (installed / "tokenizer" / name).resolve(), (target / name).resolve()
            )
        self.assertEqual(
            (installed / "target/model.safetensors").resolve(),
            (target / "model.safetensors").resolve(),
        )
        self.assertEqual((installed / "draft/model.bin").read_bytes(), b"draft")
        (target / "tokenizer.json").write_text("changed size")
        with self.assertRaises(models.ModelError):
            upstream.verify(installed)

    def test_installed_model_starts_without_the_hub(self):
        source = upstream.Repository(mlx_target(self.root / "target", DENSE))
        draft = upstream.Repository(draft_dir(self.root / "draft", DENSE))
        args = arguments(self.root, "mlx-community/Qwen3.8-27B-4bit")
        with mock.patch.object(upstream, "Repository", return_value=draft):
            upstream.prepare(args, repo=source)
        offline = mock.Mock(side_effect=AssertionError("the Hub was contacted"))
        with mock.patch.object(upstream, "Repository", offline):
            self.assertTrue(upstream.prepare(args))
        offline.assert_not_called()
        # --update resolves again, and so does an installation that no longer verifies.
        for update, damage in ((True, False), (False, True)):
            with self.subTest(update=update, damage=damage):
                if damage:
                    (self.root / "target/model.safetensors").write_text("replaced")
                resolve = mock.Mock(side_effect=[source, draft])
                with mock.patch.object(upstream, "Repository", resolve):
                    upstream.prepare(
                        argparse.Namespace(**vars(args) | {"update": update})
                    )
                self.assertEqual(resolve.call_count, 2)
                upstream.verify(
                    models.installed_root(args.models, args.model, language_only=True)
                )

    def test_hub_snapshots_are_pinned_and_old_pins_retired(self):
        cache = self.root / "hub"
        model = "mlx-community/Qwen3.8-27B-4bit"
        draft = hub_repository(
            cache, upstream.DRAFTS, "d" * 40, lambda p: draft_dir(p, DENSE)
        )
        args = arguments(self.root, model, update=True)
        installed = models.installed_root(args.models, model, language_only=True)
        owner_refs = None
        for commit in ("a" * 40, "b" * 40):
            source = hub_repository(
                cache, model, commit, lambda p: mlx_target(p, DENSE)
            )
            with mock.patch.object(upstream, "Repository", return_value=draft):
                upstream.prepare(args, repo=source)
            refs = sorted(
                (cache / "models--mlx-community--Qwen3.8-27B-4bit/refs/splash").glob(
                    "*/*"
                )
            )
            self.assertEqual([ref.name for ref in refs], [commit])
            owner_refs = refs[0].parent
            draft_refs = sorted(
                (cache / "models--incoai-internal--Splash-DFlash2/refs/splash").glob(
                    "*/*"
                )
            )
            self.assertEqual([ref.name for ref in draft_refs], ["d" * 40])
        self.assertEqual(owner_refs.name, draft_refs[0].parent.name)
        self.assertEqual(
            upstream.verify(installed)["sources"]["target"]["revision"], "b" * 40
        )

    def test_damaged_assembly_is_rebuilt(self):
        source = upstream.Repository(mlx_target(self.root / "target", DENSE))
        draft = upstream.Repository(draft_dir(self.root / "draft", DENSE))
        args = arguments(self.root, "mlx-community/Qwen3.8-27B-4bit", update=True)
        with mock.patch.object(upstream, "Repository", return_value=draft):
            upstream.prepare(args, repo=source)
        assembly = next((args.models / ".resolved").iterdir())
        (assembly / "tokenizer/tokenizer.json").unlink()
        with mock.patch.object(upstream, "Repository", return_value=draft):
            upstream.prepare(args, repo=source)
        upstream.verify(assembly)

    def test_verify_checks_full_content_and_rejects_same_size_changes(self):
        source = self.root / "source"
        source.write_bytes(b"abcd")
        assembly = self.root / "assembly"
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

    def test_hub_resolution_pins_one_commit(self):
        info = SimpleNamespace(
            sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json")]
        )
        with (
            mock.patch("huggingface_hub.HfApi") as api,
            mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(self.root / "config.json"),
            ) as file,
            mock.patch(
                "huggingface_hub.snapshot_download", return_value=str(self.root)
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

    def test_unreachable_hub_uses_the_cache_or_explains_access(self):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        request = httpx.Request(
            "GET", "https://huggingface.co/api/models/owner/private"
        )
        denied = HfHubHTTPError(
            "401 Client Error", response=httpx.Response(401, request=request)
        )
        with (
            mock.patch("huggingface_hub.HfApi") as api,
            mock.patch(
                "huggingface_hub.snapshot_download", side_effect=OSError("not cached")
            ),
        ):
            api.return_value.model_info.side_effect = denied
            with self.assertRaisesRegex(
                models.ModelError, "cannot resolve owner/private: .*HF_TOKEN"
            ):
                upstream.Repository("owner/private")

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
