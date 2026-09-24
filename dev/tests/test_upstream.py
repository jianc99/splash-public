import argparse
import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import HfHubHTTPError, IncompleteSnapshotError
from huggingface_hub.hf_api import RepoSibling

from install import models, upstream

DENSE = next(f for f in upstream.FAMILIES if f.name == "Qwen3.8-27B")
MOE = next(f for f in upstream.FAMILIES if f.name == "Qwen3.6-35B-A3B")
MODEL = "mlx-community/Qwen3.8-27B-4bit"


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


def arguments(root, model=MODEL, **options):
    return argparse.Namespace(
        model=model,
        models=root / "models",
        revision=options.get("revision"),
        language_only=options.get("language_only", True),
        draft_model=options.get("draft_model"),
    )


def http_error(status):
    request = httpx.Request("GET", "https://huggingface.co/api/models/owner/model")
    return HfHubHTTPError(
        f"{status} Client Error", response=httpx.Response(status, request=request)
    )


class FakeHub:
    """The Hub as huggingface_hub presents it to the installer: repositories
    whose branches name commits, and downloads that fill the test's Hub cache
    with snapshots of only the files requested."""

    def __init__(self, test):
        self.cache = test.cache
        self.remote = test.root / "remote"
        self.branches = {}
        # What the next Hub request raises: resolution, or downloads.
        self.failure = self.download_failure = None
        self.requests, self.downloads = [], []
        for name, replacement in (
            ("huggingface_hub.HfApi", lambda **options: self),
            ("huggingface_hub.snapshot_download", self.snapshot_download),
            ("huggingface_hub.hf_hub_download", self.hf_hub_download),
            ("huggingface_hub.HfFileSystem", lambda: self),
            ("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None),
        ):
            patch = mock.patch(name, replacement)
            patch.start()
            test.addCleanup(patch.stop)

    def publish(self, repo_id, commit, build, branch="main"):
        build(self.remote / repo_id / commit)
        self.branches[repo_id, branch] = commit

    def model_info(self, repo_id, *, revision=None, files_metadata, timeout):
        self.requests.append((repo_id, revision))
        assert files_metadata and timeout == upstream.HUB_TIMEOUT
        if self.failure:
            raise self.failure
        commit = revision
        if not models.is_hex_digest(revision, 40):
            commit = self.branches.get((repo_id, revision or "main"))
        root = self.remote / repo_id / str(commit)
        if not root.is_dir():
            raise http_error(404)
        return SimpleNamespace(
            sha=commit,
            siblings=[
                RepoSibling(
                    rfilename=name,
                    size=(root / name).stat().st_size,
                    blob_id=hashlib.sha1((root / name).read_bytes()).hexdigest(),
                )
                for name in sorted(upstream._listing(root))
            ],
        )

    def fetch(self, repo_id, name, revision):
        if self.download_failure:
            raise self.download_failure
        snapshot = self.cache / models.hub_folder_name(repo_id) / "snapshots"
        path = snapshot / revision / name
        # As huggingface_hub does, a cached file is returned as it is.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / repo_id / revision / name, path)
            self.downloads.append(f"{repo_id}/{name}")
        return path

    def snapshot_download(self, repo_id, *, revision, allow_patterns, max_workers):
        for name in allow_patterns:
            self.fetch(repo_id, name, revision)
        snapshot = self.cache / models.hub_folder_name(repo_id) / "snapshots"
        return str(snapshot / revision)

    def hf_hub_download(self, repo_id, filename, *, revision):
        return str(self.fetch(repo_id, filename, revision))

    def open(self, path, mode, *, revision, block_size):
        owner, name, filename = path.split("/", 2)
        return (self.remote / owner / name / revision / filename).open(mode)


class UpstreamTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        for patch in (
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            mock.patch("huggingface_hub.get_token", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def hub(self, repo_id, commit, build):
        """repo_id at commit as a download leaves it in the Hub cache: a
        snapshot of only the files downloaded."""
        build(self.cache / models.hub_folder_name(repo_id) / "snapshots" / commit)
        return upstream.Repository.cached(repo_id, commit)

    def fake_hub(self, *, target=DENSE, commit="a" * 40):
        """A Hub publishing MODEL at commit on main and the drafts' repository
        at every family's pinned commit."""
        hub = FakeHub(self)
        hub.publish(MODEL, commit, lambda p: mlx_target(p, target))
        for family in upstream.FAMILIES:
            hub.publish(
                upstream.DRAFTS, family.draft.revision, lambda p: draft_dir(p, family)
            )
        return hub

    @staticmethod
    def prepare(args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = upstream.prepare(args)
        return result, output.getvalue()

    def pins(self):
        return sorted(ref.name for ref in self.cache.glob("*/refs/splash/*/*"))

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
        for variant in (None, "Q4", "BF16", "missing"):
            with (
                self.subTest(variant=variant),
                self.assertRaisesRegex(models.ModelError, "Qwen3.8-27B-Q8_0.gguf"),
            ):
                upstream.select_gguf(files, variant)

    def test_architecture_is_checked_before_weight_downloads(self):
        hub = FakeHub(self)
        hub.publish(
            "someone/renamed-27b",
            "a" * 40,
            lambda p: mlx_target(p, DENSE, changes={"num_hidden_layers": 48}),
        )
        with self.assertRaisesRegex(models.ModelError, "no supported model"):
            self.prepare(arguments(self.root, "someone/renamed-27b"))
        self.assertEqual(hub.requests, [("someone/renamed-27b", None)])
        self.assertEqual(hub.downloads, ["someone/renamed-27b/config.json"])

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
                source = mock.Mock(unavailable=None)
                source.files = {"model.safetensors"} | (
                    required - {missing} if missing else set()
                )
                with mock.patch.object(
                    upstream.Repository, "resolve", return_value=source
                ) as resolve:
                    with self.assertRaisesRegex(
                        models.ModelError, "must come from the target repository"
                    ):
                        self.prepare(args)
                resolve.assert_called_once_with(
                    "user/custom", None, installation=mock.ANY
                )
                source.file.assert_not_called()
                source.download.assert_not_called()
                self.assertFalse(args.models.exists())

    def test_source_assembly_pairs_the_draft_by_architecture(self):
        hub = FakeHub(self)
        hub.publish(
            "someone/my-favourite-model", "a" * 40, lambda p: mlx_target(p, MOE)
        )
        hub.publish(upstream.DRAFTS, MOE.draft.revision, lambda p: draft_dir(p, MOE))
        # The name says nothing about the model; the configuration does.
        args = arguments(self.root, "someone/my-favourite-model")
        self.assertTrue(self.prepare(args)[0])
        self.assertEqual(
            hub.requests,
            [
                ("someone/my-favourite-model", None),
                (upstream.DRAFTS, MOE.draft.revision),
            ],
        )
        installed = models.installed_root(args.models, args.model, language_only=True)
        record = upstream.verify(installed)
        self.assertEqual(record["family"], MOE.name)
        self.assertEqual(
            record["sources"],
            {
                "target": {"repo": args.model, "revision": "a" * 40},
                "draft": {"repo": upstream.DRAFTS, "revision": MOE.draft.revision},
            },
        )
        self.assertEqual(record["vision_format"], "none")
        self.assertFalse((installed / "manifest.json").exists())
        self.assertFalse((installed / "vision").exists())
        snapshot = self.cache / "models--someone--my-favourite-model/snapshots"
        for name in ("tokenizer.json", "tokenizer_config.json"):
            self.assertEqual(
                (installed / "tokenizer" / name).readlink(),
                snapshot / ("a" * 40) / name,
            )
        self.assertEqual((installed / "draft/model.bin").read_bytes(), b"draft")
        (snapshot / ("a" * 40) / "tokenizer.json").write_text("changed size")
        with self.assertRaises(models.ModelError):
            upstream.verify(installed)

    def test_mlx_vision_links_only_the_shards_holding_the_tower(self):
        shards = {
            "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
        }

        def target(root):
            mlx_target(root, DENSE)
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": shards})
            )
            for name in set(shards.values()):
                (root / name).write_text(name)
            (root / "preprocessor_config.json").write_text(
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

        hub = self.fake_hub()
        hub.publish(MODEL, "a" * 40, target)
        args = arguments(self.root, language_only=False)
        self.assertTrue(self.prepare(args)[0])
        installed = models.installed_root(args.models, args.model)
        self.assertEqual(upstream.verify(installed)["vision_format"], "safetensors")
        self.assertFalse((installed / "processor").exists())
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
        hub.publish("someone/text-model", "b" * 40, target)
        with self.assertRaisesRegex(models.ModelError, "no vision tower"):
            self.prepare(
                arguments(self.root, "someone/text-model", language_only=False)
            )

    def test_unchanged_commit_starts_with_one_request_and_no_download(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        self.prepare(args)
        hub.requests.clear(), hub.downloads.clear()
        result, output = self.prepare(args)
        self.assertTrue(result)
        self.assertEqual(hub.requests, [(MODEL, None)])
        self.assertEqual(hub.downloads, [])
        self.assertIn("is already installed", output)

    def test_moved_commit_is_installed_and_published_atomically(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        root = models.installed_root(args.models, MODEL, language_only=True)
        _, output = self.prepare(args)
        self.assertIn("Fetching 4 file(s), 0.00 GB, from " + MODEL, output)
        old = root.resolve()
        hub.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
        hub.requests.clear(), hub.downloads.clear()
        rename = upstream.os.rename

        def publish(stage, destination):
            # The old assembly stays in use until the new one is complete.
            self.assertEqual(root.resolve(), old)
            rename(stage, destination)

        with mock.patch.object(upstream.os, "rename", side_effect=publish):
            result, output = self.prepare(args)
        self.assertTrue(result)
        self.assertIn(f"{MODEL} moved from {'a' * 12} to {'b' * 12}.", output)
        self.assertEqual(
            upstream.verify(root)["sources"]["target"]["revision"], "b" * 40
        )
        self.assertEqual(hub.requests, [(MODEL, None)])
        self.assertNotIn(f"{upstream.DRAFTS}/{DENSE.name}/model.bin", hub.downloads)
        self.assertEqual(self.pins(), sorted(["b" * 40, DENSE.draft.revision]))

    def test_unreachable_hub_starts_the_installed_assembly(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        self.prepare(args)
        for failure in (
            httpx.ConnectError("[Errno 8] nodename nor servname provided"),
            httpx.ConnectTimeout(""),
            http_error(500),
            http_error(401),
            http_error(403),
            http_error(404),
        ):
            with self.subTest(failure=failure):
                hub.failure = failure
                result, output = self.prepare(args)
                self.assertTrue(result)
                self.assertIn(
                    f"Could not reach the Hub ({models.hub_reason(failure)}); "
                    f"using the installed {MODEL}@{'a' * 12}.",
                    output,
                )
                self.assertIn("is already installed", output)
        # Without an installation, the failure is the error.
        with self.assertRaisesRegex(
            models.ModelError,
            "cannot resolve someone/other: 404 Client Error; neither this "
            "installation nor the Hub cache records a commit for the default branch",
        ):
            self.prepare(arguments(self.root, "someone/other"))

    def test_commit_and_offline_selections_make_no_request(self):
        hub = self.fake_hub()
        pinned = arguments(self.root, revision="a" * 40)
        self.prepare(pinned)
        self.prepare(arguments(self.root))
        hub.requests.clear(), hub.downloads.clear()
        self.assertTrue(self.prepare(pinned)[0])
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            self.assertTrue(self.prepare(arguments(self.root))[0])
        self.assertEqual((hub.requests, hub.downloads), ([], []))

    def test_new_commit_that_cannot_be_installed_keeps_the_installed_one(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        root = models.installed_root(args.models, MODEL, language_only=True)
        self.prepare(args)
        installed = root.resolve()
        # The new commit's architecture is rejected before any weight download,
        hub.publish(
            MODEL, "b" * 40, lambda p: mlx_target(p, DENSE, changes={"head_dim": 128})
        )
        result, output = self.prepare(args)
        self.assertTrue(result)
        self.assertIn(
            f"keeping the installed {MODEL}@{'a' * 12}; cannot install "
            f"{MODEL}@{'b' * 40}: no supported model has this architecture",
            output,
        )
        # and a download that fails keeps it too.
        hub.publish(MODEL, "c" * 40, lambda p: mlx_target(p, DENSE))
        hub.download_failure = httpx.ReadTimeout("timed out")
        result, output = self.prepare(args)
        self.assertTrue(result)
        self.assertIn(
            f"cannot install {MODEL}@{'c' * 40}: cannot install {MODEL}: timed out",
            output,
        )
        self.assertEqual(root.resolve(), installed)
        self.assertEqual(self.pins(), sorted(["a" * 40, DENSE.draft.revision]))
        # Nothing installed: the rejection is the error.
        with self.assertRaisesRegex(models.ModelError, "timed out"):
            self.prepare(arguments(self.root, revision="c" * 40))

    def test_a_release_pinning_another_draft_reassembles_the_installed_target(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        root = models.installed_root(args.models, MODEL, language_only=True)
        self.prepare(args)
        moved = dataclasses.replace(
            DENSE, draft=upstream.Draft("e" * 40, DENSE.draft.layers)
        )
        hub.publish(upstream.DRAFTS, "e" * 40, lambda p: draft_dir(p, DENSE))
        hub.requests.clear(), hub.downloads.clear()
        with mock.patch.object(upstream, "FAMILIES", (moved, MOE)):
            result, output = self.prepare(args)
            self.assertTrue(result)
            self.assertIn(f"pins the {DENSE.name} draft at {'e' * 12}", output)
            record = upstream.verify(root)
            self.assertEqual(
                record["sources"],
                {
                    "target": {"repo": MODEL, "revision": "a" * 40},
                    "draft": {"repo": upstream.DRAFTS, "revision": "e" * 40},
                },
            )
            # Only the new draft is fetched; the target is not downloaded again.
            self.assertEqual(hub.requests, [(MODEL, None), (upstream.DRAFTS, "e" * 40)])
            self.assertTrue(
                all(name.startswith(upstream.DRAFTS) for name in hub.downloads)
            )
            self.assertEqual(self.pins(), sorted(["a" * 40, "e" * 40]))
            # A draft that cannot be fetched keeps the installed one.
            newer = dataclasses.replace(
                DENSE, draft=upstream.Draft("f" * 40, DENSE.draft.layers)
            )
        with mock.patch.object(upstream, "FAMILIES", (newer, MOE)):
            result, output = self.prepare(args)
        self.assertTrue(result)
        self.assertIn(
            f"Warning: cannot fetch the {DENSE.name} draft {'f' * 12}; keeping the "
            f"installed one: cannot resolve {upstream.DRAFTS}: 404 Client Error",
            output,
        )
        self.assertEqual(
            upstream.verify(root)["sources"]["draft"]["revision"], "e" * 40
        )

    def test_hub_snapshots_are_pinned_and_old_pins_retired(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        installed = models.installed_root(args.models, MODEL, language_only=True)
        for commit in ("a" * 40, "b" * 40):
            hub.publish(MODEL, commit, lambda p: mlx_target(p, DENSE))
            self.prepare(args)
            refs = sorted(
                (
                    self.cache / "models--mlx-community--Qwen3.8-27B-4bit/refs/splash"
                ).glob("*/*")
            )
            self.assertEqual([ref.name for ref in refs], [commit])
            draft_refs = sorted(
                (
                    self.cache / "models--incoai-internal--Splash-DFlash2/refs/splash"
                ).glob("*/*")
            )
            self.assertEqual([ref.name for ref in draft_refs], [DENSE.draft.revision])
        self.assertEqual(refs[0].parent.name, draft_refs[0].parent.name)
        self.assertEqual(
            upstream.verify(installed)["sources"]["target"]["revision"], "b" * 40
        )

    def test_pins_are_required_before_publishing_and_repaired_on_start(self):
        self.fake_hub()
        args = arguments(self.root)
        installed = models.installed_root(args.models, MODEL, language_only=True)
        expected = sorted(["a" * 40, DENSE.draft.revision])
        with (
            mock.patch.object(
                models, "retain_ref", side_effect=PermissionError(errno.EACCES, "no")
            ),
            self.assertRaises(PermissionError),
        ):
            self.prepare(args)
        self.assertFalse(installed.exists())
        retain = models.retain_ref

        def locked(*arguments):
            # Pins change only under the installation lock.
            with (args.models / ".install.lock").open("a+b") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return retain(*arguments)

        with mock.patch.object(models, "retain_ref", side_effect=locked) as pinned:
            self.prepare(args)
        self.assertEqual(pinned.call_count, 2)
        self.assertEqual(self.pins(), expected)
        # A verified start restores lost pins.
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with mock.patch.object(models, "retain_ref", side_effect=locked):
            self.assertTrue(self.prepare(args)[0])
        self.assertEqual(self.pins(), expected)
        # A read-only cache leaves the verified installation usable.
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        errors = io.StringIO()
        with (
            mock.patch.object(
                models.os, "link", side_effect=OSError(errno.EROFS, "read only")
            ),
            contextlib.redirect_stderr(errors),
        ):
            self.assertTrue(self.prepare(args)[0])
        self.assertIn("external cache pruning", errors.getvalue())
        self.assertEqual(self.pins(), [])

    def test_damaged_assembly_is_rebuilt(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        self.prepare(args)
        assembly = next((args.models / ".resolved").iterdir())
        (assembly / "tokenizer/tokenizer.json").unlink()
        hub.downloads.clear()
        _, output = self.prepare(args)
        self.assertIn("Reinstalling " + MODEL, output)
        upstream.verify(assembly)
        # The cached snapshot is complete; nothing is downloaded again.
        self.assertNotIn(f"{MODEL}/model.safetensors", hub.downloads)

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
        hub = FakeHub(self)
        hub.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE), branch="v2")
        repo = upstream.Repository.resolve(MODEL, "v2")
        self.assertEqual(hub.requests, [(MODEL, "v2")])
        self.assertEqual(repo.revision, "a" * 40)
        self.assertEqual(
            repo.file("config.json"),
            self.cache
            / "models--mlx-community--Qwen3.8-27B-4bit/snapshots"
            / ("a" * 40)
            / "config.json",
        )
        self.assertEqual(
            set(repo.download({"model.safetensors"})), {"model.safetensors"}
        )
        hub.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE), branch="v2")
        # The branch moved; the resolved repository still reads its commit.
        self.assertEqual(
            json.loads(repo.file("config.json").read_text())["quantization"]["bits"], 4
        )
        self.assertEqual(repo.revision, "a" * 40)

    def test_only_an_absolute_path_is_a_local_repository(self):
        # A relative path in the working directory is still a Hub repository ID.
        (self.root / MODEL).mkdir(parents=True)
        hub = FakeHub(self)
        hub.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE), branch="branch")
        with contextlib.chdir(self.root):
            repo = upstream.Repository.resolve(MODEL, "branch")
        self.assertEqual(hub.requests, [(MODEL, "branch")])
        self.assertEqual((repo.root, repo.revision), (None, "a" * 40))
        local = upstream.Repository.resolve(str(self.root / MODEL))
        self.assertEqual((local.root, local.revision), (self.root / MODEL, None))
        with self.assertRaisesRegex(models.ModelError, "draft directory not found"):
            upstream.Repository.resolve(str(self.root / "deleted-draft"))

    def test_unreachable_hub_uses_the_cache_or_explains_access(self):
        hub = FakeHub(self)
        hub.failure = http_error(401)
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
        repo = upstream.Repository.resolve("owner/private")
        self.assertEqual(repo.revision, "c" * 40)
        self.assertIn("model.safetensors", repo.files)
        self.assertIn("401 Client Error", repo.unavailable)
        (refs / "main").write_text("d" * 40)
        with self.assertRaisesRegex(
            models.ModelError, "the Hub cache has no snapshot of " + "d" * 40
        ):
            upstream.Repository.resolve("owner/private")

    def test_offline_rebuild_uses_the_recorded_or_pinned_snapshot(self):
        hub = self.fake_hub()
        args = arguments(self.root)
        root = models.installed_root(args.models, MODEL, language_only=True)
        self.prepare(args)
        hub.requests.clear()
        # What huggingface_hub reports for a partial snapshot offline.
        incomplete = IncompleteSnapshotError("incomplete", snapshot_path="")
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            mock.patch("huggingface_hub.snapshot_download", side_effect=incomplete),
        ):
            # A damaged assembly is rebuilt from the commit it recorded,
            (root / "target/model.safetensors").unlink()
            result, output = self.prepare(args)
            self.assertTrue(result)
            self.assertIn(
                "Could not reach the Hub (HF_HUB_OFFLINE is set); installing "
                f"{MODEL}@{'a' * 12} from the Hub cache.",
                output,
            )
            upstream.verify(root)
            # a deleted one from the installation's pins,
            shutil.rmtree(root.resolve())
            self.assertTrue(self.prepare(args)[0])
            self.assertEqual(
                upstream.verify(root)["sources"]["target"]["revision"], "a" * 40
            )
            # and a new selection from the commit it names.
            self.assertTrue(self.prepare(arguments(self.root, revision="a" * 40))[0])
            with self.assertRaisesRegex(
                models.ModelError,
                f"cannot resolve {MODEL}: HF_HUB_OFFLINE is set; neither this "
                "installation nor the Hub cache records a commit for v2",
            ):
                self.prepare(arguments(self.root, revision="v2"))
        self.assertEqual(hub.requests, [])

    def test_hub_failures_during_installation_are_model_errors(self):
        def gguf_repository(root):
            root.mkdir(parents=True, exist_ok=True)
            (root / "m-Q4_K_M.gguf").touch()

        for failure, hint in (
            (httpx.ConnectError("connection reset"), False),
            (http_error(401), True),
            (http_error(404), False),
        ):
            with self.subTest(failure=failure):
                hub = FakeHub(self)
                hub.publish("owner/model", "a" * 40, gguf_repository)
                # The GGUF header read, before any download.
                with (
                    mock.patch.object(hub, "open", side_effect=failure),
                    self.assertRaises(models.ModelError) as raised,
                ):
                    self.prepare(arguments(self.root, "owner/model:Q4_K_M"))
                message = str(raised.exception)
                self.assertTrue(
                    message.startswith("cannot install owner/model:Q4_K_M: "), message
                )
                self.assertIn(str(failure), message)
                self.assertEqual("hf auth login" in message, hint)
        # A download that fails mid-transfer is reported without a traceback.
        hub = self.fake_hub()
        hub.download_failure = httpx.ReadTimeout("timed out")
        errors = io.StringIO()
        with (
            contextlib.redirect_stderr(errors),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = models.main(
                [
                    "--models",
                    str(self.root / "fresh"),
                    "--model",
                    MODEL,
                    "--language-only",
                    "prepare",
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(
            errors.getvalue(), f"error: cannot install {MODEL}: timed out\n"
        )

    def test_selection_paths_do_not_conflict(self):
        root = Path("/models")
        paths = {
            models.installed_root(root, MODEL),
            models.installed_root(root, MODEL, language_only=True),
            models.installed_root(root, MODEL, revision="old"),
            models.installed_root(root, MODEL, draft_model="mine/draft"),
        }
        self.assertEqual(len(paths), 4)
