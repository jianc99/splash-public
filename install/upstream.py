"""Resolve supported upstream models and their drafts into a local assembly.

The Hub owns downloads and snapshots; this module selects components and links
immutable files, and the native source adapters own tensor validation and
preparation. An installed assembly starts without contacting the Hub: each
source is pinned to the commit it was installed from, and --update is the only
way to follow a newer one. A target is identified by its own metadata (a GGUF
header or an MLX config), read before any weight download, and paired with the
draft trained for that architecture; repository names play no part.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

if __package__:
    from . import gguf, models
else:
    import gguf
    import models


# Splash's DFlash2 drafts share one repository, a folder per base model named
# after it: config.json (the original DFlash2 configuration plus its "splash"
# format and source), model.bin and layer-N.bin in the MDFD0004 layout.
DRAFTS = "incoai-internal/Splash-DFlash2"


@dataclass(frozen=True)
class Draft:
    # The commit of DRAFTS that holds this family's folder; None until it is
    # published.
    revision: str | None
    layers: int


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header.
    signature: tuple[tuple[str, object], ...]
    draft: Draft


FAMILIES = (
    ModelFamily(
        "Qwen3.8-27B",
        (
            ("model_type", "qwen3_5_text"),
            ("hidden_size", 5120),
            ("num_hidden_layers", 64),
            ("vocab_size", 248320),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
        ),
        Draft("f0ce2ff58f760c7e251a2a2454528273c3fa870b", 5),
    ),
    ModelFamily(
        "Qwen3.6-35B-A3B",
        (
            ("model_type", "qwen3_5_moe_text"),
            ("hidden_size", 2048),
            ("num_hidden_layers", 40),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
        ),
        Draft("b36f132a9c832599c6d08a1443cb8bbe4c2ac6cb", 6),
    ),
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


def family_for(config):
    """The one supported family whose architecture the target's config states."""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        raise models.ModelError("upstream configuration has no text_config")
    matches = [f for f in FAMILIES if all(text.get(k) == v for k, v in f.signature)]
    if len(matches) != 1:
        keys = sorted({key for family in FAMILIES for key, _ in family.signature})
        found = ", ".join(f"{key}={text.get(key)}" for key in keys if key in text)
        raise models.ModelError(
            f"no supported model has this architecture ({found}); "
            f"supported: {', '.join(f.name for f in FAMILIES)}"
        )
    return matches[0]


def select_gguf(files, variant):
    """The target GGUF: a file in the repository root named ...-<variant>.gguf.
    Of several that end so (X-Q4_K_M and X-UD-Q4_K_M for Q4_K_M), the one with
    the shortest name wins; subfolders (split BF16, MTP heads) never count."""
    candidates = sorted(
        name
        for name in files
        if "/" not in name
        and name.lower().endswith(".gguf")
        and not name.lower().startswith("mmproj")
    )
    if variant:
        suffix = "-" + variant.lower()
        matches = [n for n in candidates if Path(n).stem.lower().endswith(suffix)]
        shortest = min((len(n) for n in matches), default=0)
        matches = [n for n in matches if len(n) == shortest]
    else:
        matches = candidates
    if len(matches) != 1:
        listed = ", ".join(candidates) or "none"
        raise models.ModelError(
            (
                "no single GGUF matches :" + variant
                if variant
                else "select a GGUF with OWNER/REPO:VARIANT"
            )
            + f" (files in the repository root: {listed})"
        )
    return matches[0]


def select_vision(files):
    # The tower runs in BF16 and preparation never rounds a weight. F16 has a
    # narrower exponent than BF16, so an F16 projector has rounded small weights.
    for dtype in ("BF16", "F32"):
        name = f"mmproj-{dtype}.gguf"
        matches = [n for n in files if n.lower() == name.lower()]
        if len(matches) == 1:
            return matches[0]
    raise models.ModelError(
        "the GGUF repository has no mmproj-BF16.gguf or mmproj-F32.gguf vision "
        "projector; use --language-only to serve text only"
    )


class Repository:
    """One source at one commit: a local directory, or a Hub repository whose
    revision is resolved once, at installation, to an immutable commit."""

    def __init__(self, name, revision=None):
        from huggingface_hub import HfApi, snapshot_download

        self.name = str(name)
        local = Path(name).expanduser()
        self.local = local.is_dir()
        if self.local:
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
        except Exception as error:
            # Without the Hub, the cache can still resolve a branch it recorded
            # or a commit; it never substitutes a different revision.
            try:
                self.root = Path(
                    snapshot_download(name, revision=revision, local_files_only=True)
                )
            except Exception:
                raise models.ModelError(
                    models.hub_error(error, f"cannot resolve {name}")
                ) from None
            self.revision = self.root.name
            self.files = {
                p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
            }
        else:
            self.root = None
            self.revision = info.sha
            self.files = {item.rfilename for item in info.siblings}
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

    def open(self, name):
        """A binary stream of one file read on demand: reading a GGUF header
        costs a few range requests, not a download."""
        if name not in self.files:
            raise models.ModelError(f"missing {name} in {self.name}")
        if self.root is not None:
            return (self.root / name).open("rb")
        from huggingface_hub import HfFileSystem, try_to_load_from_cache

        cached = try_to_load_from_cache(self.name, name, revision=self.revision)
        if isinstance(cached, str):
            return open(cached, "rb")
        return HfFileSystem().open(
            f"{self.name}/{name}", "rb", revision=self.revision, block_size=8 << 20
        )

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


@dataclass
class Target:
    """What the target repository supplies, known before any weight download."""

    format: str
    config: dict
    weights: set[str]
    # Assembly path -> repository file for the vision tower: the GGUF mmproj,
    # or MLX config.json and the shards holding vision_tower.*; empty without
    # vision.
    vision: dict[str, str] = field(default_factory=dict)
    # MLX: assembly path -> repository file for configuration, tokenizer and
    # processor. A GGUF describes these itself (gguf.tokenizer_files).
    metadata: dict[str, str] = field(default_factory=dict)


def _target(repo, variant, language_only):
    if variant is not None or not any(n.endswith(".safetensors") for n in repo.files):
        name = select_gguf(repo.files, variant)
        with repo.open(name) as stream:
            header = gguf.Metadata(stream, tensors=True)
        gguf.require_loadable(header)
        vision = None if language_only else select_vision(repo.files)
        vision_header = None
        if vision:
            with repo.open(vision) as stream:
                vision_header = gguf.Metadata(stream)
        config = gguf.model_config(header, vision_header)
        print(f"Selected {name} from {repo.name}.", flush=True)
        return Target(
            "gguf", config, {name}, {"vision/mmproj.gguf": vision} if vision else {}
        )
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not language_only:
        required.add("preprocessor_config.json")
    if missing := required - repo.files:
        raise models.ModelError(
            f"target repository {repo.name} is missing: {', '.join(sorted(missing))}. "
            "Configuration, tokenizer and processor must come from the target repository."
        )
    config = models.read_json(repo.file("config.json"))
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
    metadata = {"config.json": "config.json"}
    metadata.update({"tokenizer/" + n: n for n in TOKENIZER_FILES if n in repo.files})
    vision = {}
    if not language_only:
        _validate_processor(models.read_json(repo.file("preprocessor_config.json")))
        metadata.update(
            {"processor/" + n: n for n in PROCESSOR_FILES if n in repo.files}
        )
        shards = _weight_files(repo, "vision_tower.")
        if not shards:
            raise models.ModelError(
                f"{repo.name} has no vision tower; use --language-only to serve text only"
            )
        vision = {"vision/config.json": "config.json"}
        vision.update({"vision/" + n: n for n in shards})
    return Target("mlx-affine", config, _weight_files(repo), vision, metadata)


def _weight_files(repo, prefix=""):
    """The checkpoint's shards holding a tensor whose name starts with prefix."""
    if "model.safetensors.index.json" in repo.files:
        index = models.read_json(repo.file("model.safetensors.index.json"))
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise models.ModelError("invalid safetensors shard index")
        if not all(isinstance(name, str) for name in weights.values()):
            raise models.ModelError("invalid safetensors shard filename")
        names = {file for tensor, file in weights.items() if tensor.startswith(prefix)}
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
    """The family's draft files in repo: its folder of the shared repository,
    or, for a --draft-model directory, the directory itself."""
    folder = family.name + "/" if f"{family.name}/config.json" in repo.files else ""
    layers = (f"layer-{i}.bin" for i in range(family.draft.layers))
    names = {folder + name for name in ("config.json", "model.bin", *layers)}
    if not names <= repo.files:
        raise models.ModelError(
            f"{repo.name} does not contain the Splash DFlash2 draft for {family.name}"
        )
    config = models.read_json(repo.file(folder + "config.json"))
    hidden = dict(family.signature)["hidden_size"]
    if (
        config.get("architectures") != ["DFlash2DraftModel"]
        or config.get("hidden_size") != hidden
        or config.get("num_hidden_layers") != family.draft.layers
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


def _installed(root):
    """An installation that verifies starts with no Hub access at all."""
    if not (root / "model.json").exists():
        return False
    try:
        verify(root)
    except (models.ModelError, OSError) as error:
        print(f"Reinstalling {root.name}: {error}", flush=True)
        return False
    return True


def prepare(args, repo=None):
    """Return False for a legacy package; otherwise ensure the source assembly."""
    repo_id, variant = models.split_model_id(args.model)
    revision = getattr(args, "revision", None)
    language_only = getattr(args, "language_only", False)
    draft_override = getattr(args, "draft_model", None)
    models_root = args.models.resolve()
    root = models.installed_root(
        models_root,
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
    if not getattr(args, "update", False) and _installed(root):
        print(f"Splash model {args.model} is already installed in {root}", flush=True)
        return True
    repo = repo or Repository(repo_id, revision)
    if "manifest.json" in repo.files:
        if revision or language_only or draft_override:
            raise models.ModelError(
                "source selection options require an upstream model ID"
            )
        return False
    target = _target(repo, variant, language_only)
    family = family_for(target.config)
    draft = (
        Repository(draft_override)
        if draft_override
        else Repository(DRAFTS, family.draft.revision)
    )
    draft_names = _draft_files(draft, family)
    print(
        f"Installing {args.model} as {family.name} ({target.format}); draft {draft.name}; "
        f"vision {'disabled' if language_only else 'enabled'}.",
        flush=True,
    )
    components = target.metadata | target.vision
    sources = repo.download(target.weights | set(components.values()))
    files = {name: sources[source] for name, source in components.items()}
    for name in sorted(target.weights):
        files["target/" + Path(name).name] = sources[name]
    if target.format == "gguf":
        files.update(
            _gguf_metadata(
                models_root,
                sources[next(iter(target.weights))],
                files.get("vision/mmproj.gguf"),
            )
        )
    else:
        files["target/config.json"] = files["config.json"]
    files["tokenizer/config.json"] = files["config.json"]
    drafts = draft.download(draft_names)
    for name, path in drafts.items():
        files["draft/" + Path(name).name] = path
    records = {}
    record = {
        "version": 1,
        "model": args.model,
        "family": family.name,
        "target_format": target.format,
        "vision_format": "none"
        if language_only
        else "gguf"
        if target.format == "gguf"
        else "safetensors",
        "sources": {"target": repo.identity(), "draft": draft.identity()},
        "files": {
            name: records.setdefault(str(path.absolute()), _file_record(path))
            for name, path in sorted(files.items())
        },
    }
    models_root.mkdir(parents=True, exist_ok=True)
    with models.installation_lock(models_root):
        destination = _publish(models_root, record, files)
        models.install_snapshot(destination, root)
    _pin(
        root,
        [(repo, next(iter(sources.values()))), (draft, next(iter(drafts.values())))],
    )
    return True


def _publish(models_root, record, files):
    """The verified assembly for record, rebuilding a damaged one."""
    encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
    cache = models_root / ".resolved"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / hashlib.sha256(encoded.encode()).hexdigest()
    if destination.exists():
        try:
            verify(destination)
            return destination
        except (models.ModelError, OSError):
            shutil.rmtree(destination)
    stage = Path(tempfile.mkdtemp(prefix=".loading-", dir=cache))
    try:
        for name, path in files.items():
            _link(stage, name, path)
        with (stage / "model.json").open("w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    verify(destination)
    return destination


def _snapshot_of(path):
    """The Hub snapshot folder a cached file belongs to; None for local files."""
    for parent in Path(path).absolute().parents:
        if parent.parent.name == "snapshots":
            return parent
    return None


def _pin(root, sources):
    """Pin every Hub snapshot the installation links, so pruning the Hub cache
    cannot remove them, then retire this installation's older pins."""
    refs = [
        models.retain_ref(snapshot, repo.name, root)
        for repo, sample in sources
        if not repo.local and (snapshot := _snapshot_of(sample))
    ]
    models.retire_refs(refs)


def _gguf_metadata(models_root, target, vision):
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
