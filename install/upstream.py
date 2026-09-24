"""Resolve supported upstream models and their drafts into a local assembly.

The Hub owns downloads and snapshots. This module selects components and links
immutable files; native source adapters own tensor validation and preparation.
No published Splash target package, quantization catalog or pinned revision is
required. The small registry pairs base models, independently of weight format.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __package__:
    from . import models
else:
    import models


@dataclass(frozen=True)
class ModelFamily:
    name: str
    text_type: str
    hidden_size: int
    layers: int
    draft_layers: int

    @property
    def draft_model(self):
        return f"incoai/{self.name}-DFlash2"


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "qwen3_5_text", 5120, 64, 5),
    ModelFamily("Qwen3.6-35B-A3B", "qwen3_5_moe_text", 2048, 40, 6),
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def match_family(repo_id, base_models=()):
    if isinstance(base_models, str):
        base_models = [base_models]
    if not isinstance(base_models, (list, tuple)) or not all(
        isinstance(x, str) for x in base_models
    ):
        raise models.ModelError("invalid upstream base_model metadata")
    matches = set()
    for name in (repo_id, *base_models):
        basename = name.rsplit("/", 1)[-1].lower()
        for family in FAMILIES:
            prefix = family.name.lower()
            if basename == prefix or basename.startswith(prefix + "-"):
                matches.add(family)
    if len(matches) != 1:
        raise models.ModelError(
            "model has no unambiguous supported DFlash2 pairing: " + repo_id
        )
    return matches.pop()


def validate_config(config, family):
    text = config.get("text_config", {})
    expected = {
        "model_type": family.text_type,
        "hidden_size": family.hidden_size,
        "num_hidden_layers": family.layers,
        "vocab_size": 248320,
    }
    if not isinstance(text, dict) or any(text.get(k) != v for k, v in expected.items()):
        raise models.ModelError("upstream configuration does not match " + family.name)


def select_gguf(files, variant):
    candidates = sorted(
        name
        for name in files
        if name.endswith(".gguf") and not Path(name).name.lower().startswith("mmproj")
    )
    if variant:
        candidates = [
            name
            for name in candidates
            if Path(name).stem.lower().endswith("-" + variant.lower())
        ]
    if len(candidates) != 1:
        raise models.ModelError(
            "select one GGUF with --model OWNER/REPO:VARIANT (for example :UD-Q4_K_M); split GGUF files are not supported"
        )
    return candidates[0]


def select_vision(files):
    for dtype in ("BF16", "F16", "F32"):
        matches = [
            name
            for name in files
            if Path(name).name.lower() == f"mmproj-{dtype}.gguf".lower()
        ]
        if len(matches) == 1:
            return matches[0]
    raise models.ModelError("model repository has no supported BF16/F16/F32 mmproj")


class Repository:
    def __init__(self, name, revision=None):
        from huggingface_hub import HfApi, ModelCard, snapshot_download

        self.name = str(name)
        self.base_models = []
        local = Path(name).expanduser()
        if local.is_dir():
            self.root = local.resolve()
            self.revision = None
            self.files = {
                p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
            }
            return
        models.validate_repo_id(name)
        try:
            info = HfApi().model_info(name, revision=revision)
        except Exception:
            # Hub cache remains usable without network. A missing snapshot is
            # an error; it never substitutes a different requested revision.
            try:
                self.root = Path(
                    snapshot_download(name, revision=revision, local_files_only=True)
                )
            except Exception:
                raise models.ModelError(
                    f"cannot resolve {name}; check the repository, access and connection"
                ) from None
            self.revision = self.root.name
            if (self.root / "README.md").is_file():
                self.base_models = (
                    ModelCard.load(self.root / "README.md")
                    .data.to_dict()
                    .get("base_model", [])
                )
            self.files = {
                p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
            }
        else:
            self.root = None
            self.revision = info.sha
            self.files = {item.rfilename for item in info.siblings}
            card_data = getattr(info, "card_data", None)
            card = card_data.to_dict() if card_data else {}
            self.base_models = card.get("base_model", [])
        for name in self.files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise models.ModelError("invalid repository filename")

    def file(self, name):
        if name not in self.files:
            raise models.ModelError(f"missing {name} in {self.name}")
        if self.root is not None:
            return self.root / name
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.name, name, revision=self.revision))

    def download(self, names):
        if self.root is None:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    self.name,
                    revision=self.revision,
                    allow_patterns=sorted(names),
                    max_workers=4,
                )
            )
            return {name: root / name for name in names}
        return {name: self.file(name) for name in names}

    def identity(self):
        return {"repo": self.name, "revision": self.revision}


def _weight_files(repo):
    if "model.safetensors.index.json" in repo.files:
        index = models.read_json(repo.file("model.safetensors.index.json"))
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise models.ModelError("invalid safetensors shard index")
        if not all(isinstance(name, str) for name in weights.values()):
            raise models.ModelError("invalid safetensors shard filename")
        names = set(weights.values())
    elif "model.safetensors" in repo.files:
        names = {"model.safetensors"}
    else:
        raise models.ModelError("model has no safetensors checkpoint")
    if not all(
        isinstance(name, str)
        and name in repo.files
        and "/" not in name
        and name.endswith(".safetensors")
        for name in names
    ):
        raise models.ModelError("checkpoint has missing or unsupported shards")
    return names


def _draft_files(repo, family):
    names = {
        "splash/config.json",
        "splash/model.bin",
        *(f"splash/layer-{i}.bin" for i in range(family.draft_layers)),
    }
    if not names <= repo.files:
        raise models.ModelError(
            f"{repo.name} does not contain the prepared DFlash2 weights (splash/); use a draft repository with these assets"
        )
    config = models.read_json(repo.file("splash/config.json"))
    if (
        config.get("architectures") != ["DFlash2DraftModel"]
        or config.get("hidden_size") != family.hidden_size
        or config.get("num_hidden_layers") != family.draft_layers
        or config.get("splash", {}).get("format") != "MDFD0004"
    ):
        raise models.ModelError(
            "draft configuration is incompatible with " + family.name
        )
    return names


def _validate_processor(config):
    expected = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise models.ModelError("unsupported vision preprocessing configuration")


def _link(stage, relative, source):
    path = stage / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(source.absolute())


def prepare(args, repo=None):
    """Return False for a legacy package; otherwise publish a source assembly."""
    repo_id, variant = models.split_model_id(args.model)
    revision = getattr(args, "revision", None)
    language_only = getattr(args, "language_only", False)
    draft_override = getattr(args, "draft_model", None)
    root = models.installed_root(
        args.models.resolve(),
        args.model,
        revision=revision,
        language_only=language_only,
        draft_model=draft_override,
    )
    # Old installed packages need no network lookup or migration.
    if (root / "manifest.json").exists() and not (root / "model.json").exists():
        if revision or language_only or draft_override:
            raise models.ModelError(
                "source selection options require an upstream model ID"
            )
        return False
    repo = repo or Repository(repo_id, revision)
    if "manifest.json" in repo.files:
        if revision or language_only or draft_override:
            raise models.ModelError(
                "source selection options require an upstream model ID"
            )
        return False
    family = match_family(repo_id, repo.base_models)
    gguf = variant is not None or not any(
        name.endswith(".safetensors") for name in repo.files
    )
    # Cache the card so renamed repositories also resolve while offline.
    if "README.md" in repo.files:
        repo.file("README.md")
    files = {}
    if gguf:
        target_name = select_gguf(repo.files, variant)
        vision_name = select_vision(repo.files) if not language_only else None
        sources = repo.download(
            {target_name} | ({vision_name} if vision_name else set())
        )
        files["target/" + Path(target_name).name] = sources[target_name]
        vision_path = sources[vision_name] if vision_name else None
        if vision_path:
            files["vision/mmproj.gguf"] = vision_path
        files.update(
            _gguf_metadata(args.models.resolve(), sources[target_name], vision_path)
        )
    else:
        required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        if not language_only:
            required.add("preprocessor_config.json")
        missing = required - repo.files
        if missing:
            raise models.ModelError(
                f"target repository {repo_id} is missing: {', '.join(sorted(missing))}. "
                "Configuration, tokenizer and processor must come from the target repository."
            )
        files["config.json"] = repo.file("config.json")
        for name in TOKENIZER_FILES:
            if name in repo.files:
                files["tokenizer/" + name] = repo.file(name)
        if not language_only:
            for name in PROCESSOR_FILES:
                if name in repo.files:
                    files["processor/" + name] = repo.file(name)
    config = models.read_json(files["config.json"])
    validate_config(config, family)
    if not gguf:
        quant = config.get("quantization", config.get("quantization_config", {}))
        if (
            not isinstance(quant, dict)
            or quant.get("mode", "affine") != "affine"
            or quant.get("bits") != 4
            or quant.get("group_size") != 64
        ):
            raise models.ModelError(
                "this model requires an MLX affine 4-bit/group-64 checkpoint or a supported GGUF"
            )
    if not language_only:
        _validate_processor(
            models.read_json(files["processor/preprocessor_config.json"])
        )
    draft = Repository(draft_override or family.draft_model)
    draft_names = _draft_files(draft, family)
    print(
        f"Loading {args.model}; draft {draft.name}; vision {'disabled' if language_only else 'enabled'}.",
        flush=True,
    )
    if not gguf:
        for name, path in repo.download(_weight_files(repo)).items():
            files["target/" + name] = path
            if not language_only:
                files["vision/" + name] = path
        files["target/config.json"] = files["config.json"]
        if not language_only:
            files["vision/config.json"] = files["config.json"]
    files["tokenizer/config.json"] = files["config.json"]
    for name, path in draft.download(draft_names).items():
        files["draft/" + Path(name).name] = path
    record = {
        "version": 1,
        "model": args.model,
        "target_format": "gguf" if gguf else "mlx-affine",
        "vision_format": "none" if language_only else "gguf" if gguf else "safetensors",
        "sources": {
            "target": repo.identity(),
            "config": repo.identity(),
            "tokenizer": repo.identity(),
            "draft": draft.identity(),
        },
        "files": {name: _file_record(path) for name, path in sorted(files.items())},
    }
    encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
    cache = args.models.resolve() / ".resolved"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / hashlib.sha256(encoded.encode()).hexdigest()
    with models.installation_lock(args.models.resolve()):
        if not destination.exists():
            stage = Path(tempfile.mkdtemp(prefix=".loading-", dir=cache))
            try:
                for name, path in files.items():
                    _link(stage, name, path)
                (stage / "model.json").write_text(encoded)
                os.rename(stage, destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        verify(destination)
        models.install_snapshot(destination, root)
    return True


def _gguf_metadata(models_root, target, vision):
    if __package__:
        from . import gguf
    else:
        import gguf
    from importlib.metadata import version

    source_paths = [target] + ([vision] if vision else [])
    identity = {
        "sources": [_file_record(path) for path in source_paths],
        "adapter": models.sha256(Path(gguf.__file__)),
        "tokenizers": version("tokenizers"),
    }
    key = hashlib.sha256(gguf.json_bytes(identity)).hexdigest()
    cache = models_root / ".metadata"
    destination = cache / key
    names = {
        "config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja",
    }
    if vision:
        names.add("processor/preprocessor_config.json")
    if not destination.exists():
        metadata = gguf.Metadata(target)
        vision_metadata = gguf.Metadata(vision) if vision else None
        contents = gguf.tokenizer_files(metadata)
        contents["config.json"] = gguf.json_bytes(
            gguf.model_config(metadata, vision_metadata)
        )
        if vision_metadata:
            contents["processor/preprocessor_config.json"] = gguf.json_bytes(
                gguf.processor_config(vision_metadata)
            )
        if [_file_record(path) for path in source_paths] != identity["sources"]:
            raise models.ModelError("GGUF source changed while reading metadata")
        cache.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".loading-", dir=cache))
        try:
            for name, data in contents.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            hashes = {
                name: hashlib.sha256(data).hexdigest()
                for name, data in contents.items()
            }
            (stage / "files.json").write_bytes(gguf.json_bytes(hashes))
            with models.installation_lock(models_root):
                if not destination.exists():
                    os.rename(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    hashes = models.read_json(destination / "files.json")
    if set(hashes) != names:
        raise models.ModelError("invalid prepared GGUF metadata record")
    for name in names:
        if models.sha256(destination / name) != hashes[name]:
            raise models.ModelError("prepared GGUF metadata changed: " + name)
    return {name: destination / name for name in names}


def _file_record(path):
    stat = path.stat()
    resolved = path.resolve()
    digest = resolved.name if resolved.parent.name == "blobs" else ""
    if len(digest) not in (40, 64) or any(c not in "0123456789abcdef" for c in digest):
        digest = models.sha256(path)
    return {
        "path": str(path.absolute()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "digest": digest,
    }


def verify(root, *, full=False):
    record = models.read_json(root / "model.json")
    if record.get("version") != 1 or not isinstance(record.get("files"), dict):
        raise models.ModelError("invalid resolved model record")
    for name, entry in record["files"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or type(entry.get("bytes")) is not int
            or not isinstance(entry.get("digest"), str)
        ):
            raise models.ModelError("invalid resolved model file entry")
        path = root / relative
        stat = path.stat()
        if (
            not path.is_symlink()
            or str(path.readlink()) != entry["path"]
            or stat.st_size != entry["bytes"]
            or stat.st_mtime_ns != entry.get("mtime_ns")
            or stat.st_ctime_ns != entry.get("ctime_ns")
        ):
            raise models.ModelError("resolved model file changed: " + name)
        if full:
            expected = entry["digest"]
            if len(expected) == 64:
                actual = models.sha256(path)
            elif len(expected) == 40:
                digest = hashlib.sha1(f"blob {stat.st_size}\0".encode())
                with path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                actual = digest.hexdigest()
            else:
                raise models.ModelError("invalid source digest")
            if actual != expected:
                raise models.ModelError("source content hash mismatch: " + name)
    return record
