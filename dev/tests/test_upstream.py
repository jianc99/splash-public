import argparse
import contextlib
import errno
import fcntl
import io
import json
import shutil
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
        self.cache = self.root / "hub"
        for patch in (
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch("huggingface_hub.get_token", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def hub(self, repo_id, commit, build):
        """repo_id at commit as a download leaves it in the Hub cache: a
        snapshot of only the files downloaded."""
        build(self.cache / models.hub_folder_name(repo_id) / "snapshots" / commit)
        return upstream.Repository.cached(repo_id, commit)

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
        self.assertEqual(
            upstream.select_vision({"mmproj-F16.gguf", "mmproj-F32.gguf"}),
            "mmproj-F32.gguf",
        )
        # F16 has already rounded small weights of the BF16 tower.
        for projectors in ({"mmproj-Q8_0.gguf"}, {"mmproj-F16.gguf"}):
            with (
                self.subTest(projectors=projectors),
                self.assertRaisesRegex(models.ModelError, "--language-only"),
            ):
                upstream.select_vision(projectors)

    def test_architecture_is_checked_before_weight_downloads(self):
        target = mlx_target(
            self.root / "target", DENSE, changes={"num_hidden_layers": 48}
        )
        source = upstream.Repository.local_directory(target)
        with (
            mock.patch.object(source, "download") as download,
            mock.patch.object(upstream.Repository, "resolve") as resolve,
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
                with mock.patch.object(upstream.Repository, "resolve") as resolve:
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
        source = upstream.Repository.local_directory(target)
        draft = upstream.Repository.local_directory(draft_dir(self.root / "draft", MOE))
        # The name says nothing about the model; the configuration does.
        args = arguments(self.root, "someone/my-favourite-model")
        with mock.patch.object(
            upstream.Repository, "resolve", return_value=draft
        ) as resolve:
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

    def test_mlx_vision_links_only_the_shards_holding_the_tower(self):
        target = mlx_target(self.root / "target", DENSE)
        (target / "model.safetensors").unlink()
        shards = {
            "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
        }
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": shards})
        )
        for name in set(shards.values()):
            (target / name).write_text(name)
        (target / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "patch_size": 16,
                    "temporal_patch_size": 2,
                    "merge_size": 2,
                    "image_mean": [0.5] * 3,
                    "image_std": [0.5] * 3,
                }
            )
        )
        source = upstream.Repository.local_directory(target)
        draft = upstream.Repository.local_directory(
            draft_dir(self.root / "draft", DENSE)
        )
        args = arguments(
            self.root, "mlx-community/Qwen3.8-27B-4bit", language_only=False
        )
        with mock.patch.object(upstream.Repository, "resolve", return_value=draft):
            self.assertTrue(upstream.prepare(args, repo=source))
        installed = models.installed_root(args.models, args.model)
        self.assertEqual(upstream.verify(installed)["vision_format"], "safetensors")
        self.assertEqual(
            sorted(p.name for p in (installed / "vision").iterdir()),
            ["config.json", "model-00001-of-00002.safetensors"],
        )
        self.assertEqual(
            sorted(p.name for p in (installed / "target").iterdir()),
            ["config.json", *sorted(set(shards.values()))],
        )
        # A checkpoint without the tower cannot serve images.
        del shards["vision_tower.blocks.0.attn.qkv.weight"]
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": shards})
        )
        with self.assertRaisesRegex(models.ModelError, "no vision tower"):
            upstream.prepare(
                arguments(self.root, "someone/text-model", language_only=False),
                repo=upstream.Repository.local_directory(target),
            )

    def test_installed_model_starts_without_the_hub(self):
        source = upstream.Repository.local_directory(
            mlx_target(self.root / "target", DENSE)
        )
        draft = upstream.Repository.local_directory(
            draft_dir(self.root / "draft", DENSE)
        )
        args = arguments(self.root, "mlx-community/Qwen3.8-27B-4bit")
        with mock.patch.object(upstream.Repository, "resolve", return_value=draft):
            upstream.prepare(args, repo=source)
        offline = mock.Mock(side_effect=AssertionError("the Hub was contacted"))
        with mock.patch.object(upstream.Repository, "resolve", offline):
            self.assertTrue(upstream.prepare(args))
        offline.assert_not_called()
        # --update resolves again, and so does an installation that no longer verifies.
        for update, damage in ((True, False), (False, True)):
            with self.subTest(update=update, damage=damage):
                if damage:
                    (self.root / "target/model.safetensors").write_text("replaced")
                resolve = mock.Mock(side_effect=[source, draft])
                with mock.patch.object(upstream.Repository, "resolve", resolve):
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
        draft = self.hub(upstream.DRAFTS, "d" * 40, lambda p: draft_dir(p, DENSE))
        args = arguments(self.root, model, update=True)
        installed = models.installed_root(args.models, model, language_only=True)
        owner_refs = None
        for commit in ("a" * 40, "b" * 40):
            source = self.hub(model, commit, lambda p: mlx_target(p, DENSE))
            with mock.patch.object(upstream.Repository, "resolve", return_value=draft):
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

    def test_pins_are_required_before_publishing_and_repaired_on_start(self):
        cache = self.root / "hub"
        model = "mlx-community/Qwen3.8-27B-4bit"
        source = self.hub(model, "a" * 40, lambda p: mlx_target(p, DENSE))
        draft = self.hub(upstream.DRAFTS, "d" * 40, lambda p: draft_dir(p, DENSE))
        args = arguments(self.root, model)
        installed = models.installed_root(args.models, model, language_only=True)

        def pins():
            return sorted(ref.name for ref in cache.glob("*/refs/splash/*/*"))

        with (
            mock.patch.object(upstream.Repository, "resolve", return_value=draft),
            mock.patch.object(
                models, "retain_ref", side_effect=PermissionError(errno.EACCES, "no")
            ),
            self.assertRaises(PermissionError),
        ):
            upstream.prepare(args, repo=source)
        self.assertFalse(installed.exists())
        retain = models.retain_ref

        def locked(*arguments):
            # Pins change only under the installation lock.
            with (args.models / ".install.lock").open("a+b") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return retain(*arguments)

        with (
            mock.patch.object(upstream.Repository, "resolve", return_value=draft),
            mock.patch.object(models, "retain_ref", side_effect=locked) as pinned,
        ):
            upstream.prepare(args, repo=source)
        self.assertEqual(pinned.call_count, 2)
        self.assertEqual(pins(), ["a" * 40, "d" * 40])
        # A verified start restores lost pins, without the Hub.
        offline = mock.Mock(side_effect=AssertionError("the Hub was contacted"))
        for ref in cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with (
            mock.patch.object(upstream.Repository, "resolve", offline),
            mock.patch.object(models, "retain_ref", side_effect=locked),
        ):
            self.assertTrue(upstream.prepare(args))
        self.assertEqual(pins(), ["a" * 40, "d" * 40])
        # A read-only cache leaves the verified installation usable.
        for ref in cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        errors = io.StringIO()
        with (
            mock.patch.object(upstream.Repository, "resolve", offline),
            mock.patch.object(
                models.os, "link", side_effect=OSError(errno.EROFS, "read only")
            ),
            contextlib.redirect_stderr(errors),
        ):
            self.assertTrue(upstream.prepare(args))
        self.assertIn("external cache pruning", errors.getvalue())
        self.assertEqual(pins(), [])

    def test_damaged_assembly_is_rebuilt(self):
        source = upstream.Repository.local_directory(
            mlx_target(self.root / "target", DENSE)
        )
        draft = upstream.Repository.local_directory(
            draft_dir(self.root / "draft", DENSE)
        )
        args = arguments(self.root, "mlx-community/Qwen3.8-27B-4bit", update=True)
        with mock.patch.object(upstream.Repository, "resolve", return_value=draft):
            upstream.prepare(args, repo=source)
        assembly = next((args.models / ".resolved").iterdir())
        (assembly / "tokenizer/tokenizer.json").unlink()
        with mock.patch.object(upstream.Repository, "resolve", return_value=draft):
            upstream.prepare(args, repo=source)
        upstream.verify(assembly)

    def test_verify_checks_full_content_and_rejects_same_size_changes(self):
        source = self.root / "source"
        source.write_bytes(b"abcd")
        assembly = self.root / "assembly"
        assembly.mkdir()
        (assembly / "weight").symlink_to(source)
        local = {"repo": str(self.root), "revision": None}
        record = {
            "version": 1,
            "sources": {"target": local, "draft": local},
            "files": {"weight": upstream._file_record(source)},
        }
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
            repo = upstream.Repository.resolve("mlx-community/Qwen3.8-27B-4bit")
            api.return_value.model_info.assert_called_once_with(
                repo.name, revision=None
            )
            repo.file("config.json")
            self.assertEqual(file.call_args.kwargs["revision"], info.sha)
            repo.download({"config.json"})
            self.assertEqual(snapshot.call_args.kwargs["revision"], info.sha)

    def test_only_an_absolute_path_is_a_local_repository(self):
        # A relative path in the working directory is still a Hub repository ID.
        (self.root / "mlx-community/Qwen3.8-27B-4bit").mkdir(parents=True)
        info = SimpleNamespace(sha="a" * 40, siblings=[])
        with (
            contextlib.chdir(self.root),
            mock.patch("huggingface_hub.HfApi") as api,
        ):
            api.return_value.model_info.return_value = info
            repo = upstream.Repository.resolve(
                "mlx-community/Qwen3.8-27B-4bit", "branch"
            )
        api.return_value.model_info.assert_called_once_with(
            "mlx-community/Qwen3.8-27B-4bit", revision="branch"
        )
        self.assertEqual((repo.root, repo.revision), (None, "a" * 40))
        local = upstream.Repository.resolve(
            str(self.root / "mlx-community/Qwen3.8-27B-4bit")
        )
        self.assertEqual(local.root, self.root / "mlx-community/Qwen3.8-27B-4bit")
        self.assertIsNone(local.revision)
        with self.assertRaisesRegex(models.ModelError, "draft directory not found"):
            upstream.Repository.resolve(str(self.root / "deleted-draft"))

    def test_unreachable_hub_uses_the_cache_or_explains_access(self):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        request = httpx.Request(
            "GET", "https://huggingface.co/api/models/owner/private"
        )
        denied = HfHubHTTPError(
            "401 Client Error", response=httpx.Response(401, request=request)
        )
        with mock.patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.side_effect = denied
            with self.assertRaisesRegex(
                models.ModelError,
                "cannot resolve owner/private: 401 Client Error; set HF_TOKEN .*; "
                "neither this installation nor the Hub cache records a commit "
                "for the default branch",
            ):
                upstream.Repository.resolve("owner/private")
            # A branch the cache recorded resolves to its cached snapshot.
            self.hub("owner/private", "c" * 40, lambda p: mlx_target(p, DENSE))
            refs = self.cache / "models--owner--private/refs"
            refs.mkdir()
            (refs / "main").write_text("c" * 40)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                repo = upstream.Repository.resolve("owner/private")
            self.assertEqual(repo.revision, "c" * 40)
            self.assertIn("model.safetensors", repo.files)
            self.assertIn("cached snapshot " + "c" * 40, output.getvalue())
            (refs / "main").write_text("d" * 40)
            with self.assertRaisesRegex(
                models.ModelError, "the Hub cache has no snapshot of " + "d" * 40
            ):
                upstream.Repository.resolve("owner/private")

    def test_offline_rebuild_uses_the_recorded_or_pinned_snapshot(self):
        from huggingface_hub.errors import IncompleteSnapshotError, OfflineModeIsEnabled

        model = "mlx-community/Qwen3.8-27B-4bit"
        commit = "a" * 40
        # Partial snapshots, holding only what the installation downloaded.
        source = self.hub(model, commit, lambda p: mlx_target(p, DENSE))
        draft = self.hub(
            upstream.DRAFTS, DENSE.draft.revision, lambda p: draft_dir(p, DENSE)
        )
        args = arguments(self.root, model)
        root = models.installed_root(args.models, model, language_only=True)
        with mock.patch.object(
            upstream.Repository, "resolve", side_effect=[source, draft]
        ):
            upstream.prepare(args)
        output = io.StringIO()
        with (
            mock.patch("huggingface_hub.HfApi") as api,
            # What huggingface_hub reports for a partial snapshot offline.
            mock.patch(
                "huggingface_hub.snapshot_download",
                side_effect=IncompleteSnapshotError("incomplete", snapshot_path=""),
            ),
            contextlib.redirect_stdout(output),
        ):
            api.return_value.model_info.side_effect = OfflineModeIsEnabled("offline")
            # A damaged assembly is rebuilt from the commit it recorded,
            (root / "target/model.safetensors").unlink()
            self.assertTrue(upstream.prepare(args))
            upstream.verify(root)
            # a deleted one from the installation's pins,
            shutil.rmtree(root.resolve())
            self.assertTrue(upstream.prepare(args))
            self.assertEqual(
                upstream.verify(root)["sources"]["target"]["revision"], commit
            )
            # and a new selection from the commit it names.
            selection = arguments(self.root, model, revision=commit)
            self.assertTrue(upstream.prepare(selection))
            self.assertEqual(
                upstream.verify(
                    models.installed_root(
                        args.models, model, revision=commit, language_only=True
                    )
                )["sources"],
                upstream.verify(root)["sources"],
            )
            with self.assertRaisesRegex(
                models.ModelError,
                f"cannot resolve {model}: offline; neither this installation nor "
                "the Hub cache records a commit for v2",
            ):
                upstream.prepare(arguments(self.root, model, revision="v2"))
        self.assertIn(
            f"Using the cached snapshot {commit} of {model}", output.getvalue()
        )

    def test_hub_failures_during_installation_are_model_errors(self):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        def status(code):
            request = httpx.Request("GET", "https://huggingface.co/owner/model")
            return HfHubHTTPError(
                f"{code} Client Error", response=httpx.Response(code, request=request)
            )

        info = SimpleNamespace(
            sha="a" * 40, siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf")]
        )
        for failure, hint in (
            (httpx.ConnectError("connection reset"), False),
            (status(401), True),
            (status(404), False),
        ):
            with (
                self.subTest(failure=failure),
                mock.patch("huggingface_hub.get_token", return_value=None),
                mock.patch("huggingface_hub.HfApi") as api,
                mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None),
                mock.patch("huggingface_hub.HfFileSystem") as filesystem,
            ):
                api.return_value.model_info.return_value = info
                # The GGUF header read, before any download.
                filesystem.return_value.open.side_effect = failure
                with self.assertRaises(models.ModelError) as raised:
                    upstream.prepare(arguments(self.root, "owner/model:Q4_K_M"))
            message = str(raised.exception)
            self.assertTrue(message.startswith("cannot install owner/model:Q4_K_M: "))
            self.assertIn(str(failure), message)
            self.assertEqual("hf auth login" in message, hint)
        # A download that fails mid-transfer is reported without a traceback.
        info.siblings = [
            SimpleNamespace(rfilename=name)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
            )
        ]
        errors = io.StringIO()
        with (
            mock.patch("huggingface_hub.get_token", return_value=None),
            mock.patch("huggingface_hub.HfApi") as api,
            mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=httpx.ReadTimeout("timed out"),
            ),
            contextlib.redirect_stderr(errors),
        ):
            api.return_value.model_info.return_value = info
            code = models.main(
                [
                    "--models",
                    str(self.root / "models"),
                    "--model",
                    "owner/model",
                    "--language-only",
                    "prepare",
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(
            errors.getvalue(), "error: cannot install owner/model: timed out\n"
        )

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
