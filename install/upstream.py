"""Resolve supported upstream models and their drafts into a local assembly.

The Hub owns downloads and snapshots; this module selects components and links
immutable files, and the native source adapters own tensor validation and
preparation. Every start follows the target's revision with one Hub request and
publishes a new commit's assembly atomically; the installed assembly starts
when the Hub cannot answer or the new commit cannot be installed, and a commit
revision or HF_HUB_OFFLINE needs no request. A target is identified by its own
metadata (a GGUF header or an MLX config), read before any weight download,
and paired with the draft trained for that architecture, pinned per family;
repository names play no part.
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


# Seconds the Hub may take to resolve a revision before the installed
# assembly starts without it.
HUB_TIMEOUT = 5

# Splash's DFlash2 drafts share one repository, a folder per base model named
# after it: config.json (the original DFlash2 configuration plus its "splash"
# format and source), model.bin and layer-N.bin (models.DRAFT_LAYER_MAGIC).
DRAFTS = "incoai-internal/Splash-DFlash2"


@dataclass(frozen=True)
class Draft:
    # The commit of DRAFTS that published this family's folder.
    revision: str
    layers: int


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header,
    # including every one the native source model inspection requires.
    signature: tuple[tuple[str, object], ...]
    draft: Draft


FAMILIES = (
    ModelFamily(
        "Qwen3.8-27B",
        (
            ("model_type", "qwen3_5_text"),
            ("max_position_embeddings", 262144),
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
            ("max_position_embeddings", 262144),
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
# The tokenizer files an MLX target may supply, linked when present.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
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
    """The target GGUF among the repository's root files; subfolders (split
    BF16, MTP heads) never count. :VARIANT names the file whose name is the
    model name all of them share, then -VARIANT (X-Q4_K_M for Q4_K_M, not
    X-UD-Q4_K_M). Without one, the only file whose name ends in -VARIANT is
    taken, and the choice is printed; several are an error."""
    candidates = sorted(
        name
        for name in files
        if "/" not in name
        and name.lower().endswith(".gguf")
        and not name.lower().startswith("mmproj")
    )
    matches = candidates
    if variant:
        parts = [Path(name).stem.split("-") for name in candidates]
        shared = len(os.path.commonprefix(parts))
        matches = [
            name
            for name, words in zip(candidates, parts, strict=True)
            if "-".join(words[shared:]).lower() == variant.lower()
        ]
        if len(matches) != 1:
            suffix = "-" + variant.lower()
            matches = [n for n in candidates if Path(n).stem.lower().endswith(suffix)]
            # A repository of one GGUF names no variant apart from its model.
            if len(matches) == 1 and len(candidates) > 1:
                print(
                    f"No GGUF is named for :{variant} alone; using {matches[0]}, "
                    f"the only one whose name ends in -{variant}.",
                    flush=True,
                )
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


def select_vision(repo):
    """The name and header of the GGUF repository's vision projector, chosen
    by content among its root mmproj*.gguf files, whatever the publisher
    calls them: a clip model whose weights are BF16, or F32, which
    preparation converts only where every value is exact; BF16 is preferred.
    The tower runs in BF16 and preparation never rounds a weight: F16 has a
    narrower exponent than BF16, so an F16 projector has already rounded
    small weights, as a quantized one has. Each header costs a few range
    requests."""
    usable, found = {"BF16": [], "F32": []}, []
    for name in sorted(repo.files):
        if "/" in name or not name.lower().startswith("mmproj"):
            continue
        if not name.lower().endswith(".gguf"):
            continue
        with repo.open(name) as stream:
            header = gguf.Metadata(stream, tensors=True)
        architecture = header.values.get("general.architecture")
        types = {
            gguf.TENSOR_TYPES.get(kind, f"type {kind}")
            for kind in header.tensors.values()
        }
        found.append(f"{name} ({architecture}: {', '.join(sorted(types))})")
        if architecture == "clip" and types and types <= {"BF16", "F32"}:
            usable["BF16" if "BF16" in types else "F32"].append((name, header))
    for precision, projectors in usable.items():
        if len(projectors) == 1:
            return projectors[0]
        if projectors:
            raise models.ModelError(
                f"several {precision} vision projectors, "
                + ", ".join(name for name, _ in projectors)
                + ", describe no single tower; use --language-only to serve text only"
            )
    raise models.ModelError(
        "the GGUF repository has no BF16 or F32 vision projector ("
        + ("; ".join(found) or "no mmproj*.gguf")
        + "); use --language-only to serve text only"
    )


class Repository:
    """One source at one commit: a local draft directory, or a Hub repository
    at the commit its revision resolved to, read from the Hub or from the
    commit's cached snapshot."""

    def __init__(self, name, revision, files, root=None, *, sizes=None):
        for file in files:
            path = PurePosixPath(file)
            if path.is_absolute() or ".." in path.parts:
                raise models.ModelError("invalid repository filename")
        self.name, self.revision, self.files, self.root = name, revision, files, root
        # Hub filename -> (bytes, blob ID), to report what a download fetches.
        self.sizes = sizes or {}
        # Why the Hub could not resolve this source, when its cached snapshot
        # stands in; None when the Hub resolved it.
        self.unavailable = None

    @classmethod
    def local_directory(cls, path):
        path = Path(path)
        if not path.is_dir():
            raise models.ModelError(f"local draft directory not found: {path}")
        return cls(str(path), None, _listing(path), path)

    @classmethod
    def cached(cls, name, commit):
        """name at commit from its snapshot in the Hub cache, without the Hub;
        only the files downloaded before are available."""
        snapshot = _snapshot(name, commit)
        if not snapshot.is_dir():
            raise models.ModelError(f"the Hub cache has no snapshot {commit} of {name}")
        return cls(name, commit, _listing(snapshot), snapshot)

    @classmethod
    def recorded(cls, source):
        """A source as an installed record names it, without the Hub."""
        if source["revision"] is None:
            return cls.local_directory(source["repo"])
        return cls.cached(source["repo"], source["revision"])

    @classmethod
    def resolve(cls, name, revision=None, *, installation=None):
        """name at the commit revision names now, from one Hub request. Only
        an absolute path is a local directory: parse_draft_model makes a
        --draft-model directory absolute, and a target is always a Hub ID,
        whatever the working directory holds. When the Hub cannot answer (or
        HF_HUB_OFFLINE is set), the cached snapshot of a commit this selection
        already names stands in, with the reason in unavailable
        (_cached_commits); a different revision is never substituted."""
        import httpx
        from huggingface_hub import HfApi, constants

        if Path(name).is_absolute():
            return cls.local_directory(name)
        models.validate_repo_id(name)
        if constants.HF_HUB_OFFLINE:
            reason = "HF_HUB_OFFLINE is set"
        else:
            try:
                info = HfApi().model_info(
                    name, revision=revision, files_metadata=True, timeout=HUB_TIMEOUT
                )
            except (OSError, httpx.HTTPError) as error:
                reason = models.hub_reason(error)
            else:
                if not models.is_hex_digest(info.sha, 40):
                    raise models.ModelError(
                        f"the Hub did not resolve {name} to a commit"
                    )
                return cls(
                    name,
                    info.sha,
                    {item.rfilename for item in info.siblings},
                    sizes={
                        item.rfilename: (
                            item.size,
                            item.lfs.sha256 if item.lfs else item.blob_id,
                        )
                        for item in info.siblings
                        if item.size is not None
                    },
                )
        commits = _cached_commits(name, revision, installation)
        for commit in commits:
            if _snapshot(name, commit).is_dir():
                repo = cls.cached(name, commit)
                repo.unavailable = reason
                return repo
        cache = (
            "the Hub cache has no snapshot of " + ", ".join(commits)
            if commits
            else "neither this installation nor the Hub cache records a commit "
            f"for {revision or 'the default branch'}"
        )
        raise models.ModelError(f"cannot resolve {name}: {reason}; {cache}")

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

            blobs = _hub_folder(self.name) / "blobs"
            fetch = [
                size
                for size, blob in (self.sizes[n] for n in names if n in self.sizes)
                if not (blob and (blobs / blob).exists())
            ]
            if fetch:
                print(
                    f"Fetching {len(fetch)} file(s), {sum(fetch) / 1e9:.2f} GB, "
                    f"from {self.name}@{self.revision[:12]}; cached files are reused.",
                    flush=True,
                )
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
    # MLX: assembly path -> repository file for configuration and tokenizer.
    # A GGUF describes these itself (gguf.tokenizer_files).
    metadata: dict[str, str] = field(default_factory=dict)


def _listing(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _hub_folder(name):
    """Where the Hub cache keeps repository name."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / models.hub_folder_name(name)


def _snapshot(name, commit):
    return _hub_folder(name) / "snapshots" / commit


def _cached_commits(name, revision, installation):
    """The commits of name whose cached snapshots may stand in for revision
    without the Hub, most specific first: revision itself when it is a
    commit; the commit installation recorded, then the ones it pinned
    (refs/splash); then the commit the cache recorded for the branch or tag.
    Splash downloads by commit, which never records a branch."""
    if models.is_hex_digest(revision, 40):
        return [revision.lower()]
    folder = _hub_folder(name)
    commits = []
    if installation is not None:
        try:
            sources = models.read_json(installation / "model.json").get("sources")
        except models.ModelError:
            sources = None
        if isinstance(sources, dict):
            commits += [
                source.get("revision")
                for source in sources.values()
                if isinstance(source, dict) and source.get("repo") == name
            ]
        pins = folder / "refs" / "splash" / models.pin_owner(installation)
        if pins.is_dir():
            commits += [
                pin.name
                for pin in sorted(
                    pins.iterdir(), key=lambda pin: pin.stat().st_mtime_ns, reverse=True
                )
            ]
    try:
        commits.append((folder / "refs" / (revision or "main")).read_text().strip())
    except OSError:
        pass
    return list(dict.fromkeys(c for c in commits if models.is_hex_digest(c, 40)))


def _target(repo, variant, language_only):
    if variant is not None or not any(n.endswith(".safetensors") for n in repo.files):
        name = select_gguf(repo.files, variant)
        with repo.open(name) as stream:
            header = gguf.Metadata(stream, tensors=True)
        gguf.require_loadable(header)
        vision, vision_header = (None, None) if language_only else select_vision(repo)
        if vision:
            _validate_processor(gguf.processor_config(vision_header))
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
    # MLX states its quantization under "quantization"; a transformers
    # quantization_config alone describes another method (GPTQ, AWQ, ...).
    quant = config.get("quantization")
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
        # One file: its header says whether it holds such a tensor.
        holds = not prefix or any(
            tensor.startswith(prefix)
            for tensor in _safetensors_tensors(repo, "model.safetensors")
        )
        names = {"model.safetensors"} if holds else set()
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


def _safetensors_tensors(repo, name):
    """The tensor names in a safetensors file's header: its length (8 bytes,
    little-endian), then a JSON object, read on demand, without a download."""
    with repo.open(name) as stream:
        size = int.from_bytes(stream.read(8), "little")
        # The native checkpoint reader's bound on one header.
        if not 2 <= size <= 1 << 20:
            raise models.ModelError(f"invalid safetensors header in {name}")
        try:
            header = json.loads(stream.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise models.ModelError(f"invalid safetensors header in {name}") from error
    if not isinstance(header, dict):
        raise models.ModelError(f"invalid safetensors header in {name}")
    return set(header) - {"__metadata__"}


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
        or config.get("splash", {}).get("format") != models.DRAFT_LAYER_MAGIC
    ):
        raise models.ModelError(
            "draft configuration is incompatible with " + family.name
        )
    return names


def _validate_processor(config):
    """Splash prepares images one way (server/images.py): 16-pixel patches in
    two temporal slices, merged 2x2 and normalized to [-1, 1]. A processor
    configuration asking for another is rejected before any download; it is
    not installed, since nothing reads it."""
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


def _installed(root, model):
    """The record of the installation at root if it verifies, else None;
    verifying needs no Hub access."""
    if not (root / "model.json").exists():
        return None
    try:
        return verify(root)
    except (models.ModelError, OSError) as error:
        print(f"Reinstalling {model}: {error}", flush=True)
        return None


def prepare(args):
    """Ensure the assembly of args's selection and return True, following the
    Hub; return False for a legacy Splash package.

    Every start resolves the target's revision with one Hub request. The
    installed assembly starts when that is its commit, when the Hub cannot
    answer, and when a new commit cannot be installed; a new commit is
    installed and published atomically. A commit revision never moves and
    HF_HUB_OFFLINE forbids the Hub, so both start a verified installation
    without a request."""
    from huggingface_hub import constants

    repo_id, variant = models.split_model_id(args.model)
    models_root = args.models.resolve()
    root = models.installed_root(
        models_root,
        args.model,
        revision=args.revision,
        language_only=args.language_only,
        draft_model=args.draft_model,
    )
    # An installed legacy package needs no network lookup or migration.
    if (root / "manifest.json").exists() and not (root / "model.json").exists():
        return False
    installed = _installed(root, args.model)
    kept = installed and installed["sources"]["target"]
    if installed is not None and (
        models.is_hex_digest(args.revision, 40) or constants.HF_HUB_OFFLINE
    ):
        repo = Repository.recorded(kept)
    else:
        repo = Repository.resolve(repo_id, args.revision, installation=root)
        if installed is None and "manifest.json" in repo.files:
            if args.revision or args.language_only or args.draft_model:
                raise models.ModelError(
                    "source selection options require an upstream model ID"
                )
            return False
    current = installed is not None and repo.revision == kept["revision"]
    if repo.unavailable:
        print(
            f"Could not reach the Hub ({repo.unavailable}); "
            + (
                f"using the installed {repo_id}@{repo.revision[:12]}."
                if current
                else f"installing {repo_id}@{repo.revision[:12]} from the Hub cache."
            ),
            flush=True,
        )
    elif installed is not None and not current:
        print(
            f"{repo_id} moved from {kept['revision'][:12]} to {repo.revision[:12]}.",
            flush=True,
        )
    if current:
        if not (changes := _changes(installed, args.draft_model)):
            if _start(models_root, root, args.model):
                print(
                    f"Splash model {args.model} is already installed in {root}",
                    flush=True,
                )
                return True
        else:
            print(f"Updating {args.model}: {'; '.join(changes)}.", flush=True)
        # The target is unchanged; its cached snapshot holds what it needs.
        repo = Repository.recorded(kept)
    try:
        _install(args, repo, variant, models_root, root, installed)
    except (models.ModelError, OSError) as error:
        # A revision that cannot be installed leaves the installed one in use.
        if installed is None:
            raise
        rejected = "this release's changes" if current else f"{repo_id}@{repo.revision}"
        print(
            f"Warning: keeping the installed {repo_id}@{kept['revision'][:12]}; "
            f"cannot install {rejected}: {error}",
            flush=True,
        )
        if not _start(models_root, root, args.model):
            raise
    return True


def _start(models_root, root, model):
    """Pin the installation at root and return True if it verifies under the
    lock; a concurrent installation may have replaced it since it was read."""
    with models.installation_lock(models_root):
        if (installed := _installed(root, model)) is None:
            return False
        models.retain_refs(root, _pins(installed))
    return True


def _install(args, repo, variant, models_root, root, installed):
    """Assemble repo's target and its draft, and publish the assembly at root."""
    # Every Hub request (header reads, downloads) happens here.
    with models.hub_errors(f"cannot install {args.model}"):
        target = _target(repo, variant, args.language_only)
        family = family_for(target.config)
        draft, drafts = _draft(family, args.draft_model, installed, root)
        print(
            f"Installing {args.model} as {family.name} ({target.format}); "
            f"draft {draft.name}; "
            f"vision {'disabled' if args.language_only else 'enabled'}.",
            flush=True,
        )
        components = target.metadata | target.vision
        sources = repo.download(target.weights | set(components.values()))
    files = {name: sources[source] for name, source in components.items()}
    for name in sorted(target.weights):
        files["target/" + Path(name).name] = sources[name]
    for name, path in drafts.items():
        files["draft/" + Path(name).name] = path
    record = {
        "version": 1,
        "model": args.model,
        "family": family.name,
        "target_format": target.format,
        "vision_format": "none"
        if args.language_only
        else "gguf"
        if target.format == "gguf"
        else "safetensors",
        "sources": {"target": repo.identity(), "draft": draft.identity()},
    }
    models_root.mkdir(parents=True, exist_ok=True)
    # Everything written under the models root is written under the lock.
    with models.installation_lock(models_root):
        if target.format == "gguf":
            record["metadata"], metadata = _gguf_metadata(
                models_root,
                sources[next(iter(target.weights))],
                files.get("vision/mmproj.gguf"),
            )
            files.update(metadata)
        else:
            files["target/config.json"] = files["config.json"]
        files["tokenizer/config.json"] = files["config.json"]
        # One record per linked file, however many assembly paths link it.
        records = {path: _file_record(path) for path in set(files.values())}
        record["files"] = {name: records[path] for name, path in sorted(files.items())}
        # A new installation requires its pins before it is published; its
        # older pins are retired once it is.
        refs = [
            models.retain_ref(snapshot, name, root) for snapshot, name in _pins(record)
        ]
        destination = _publish(models_root, record, files)
        models.install_snapshot(destination, root)
        models.retire_refs(refs)


def _draft_identity(family):
    return {"repo": DRAFTS, "revision": family.draft.revision}


def _changes(record, draft_override):
    """What this release changes in a verified assembly, from local files
    alone: another pinned draft for its family (unless --draft-model chose
    the draft), or another adapter for its derived GGUF metadata."""
    family = next((f for f in FAMILIES if f.name == record.get("family")), None)
    changes = []
    if family is None:
        changes.append(f"no supported family is named {record.get('family')}")
    elif not draft_override and record["sources"]["draft"] != _draft_identity(family):
        changes.append(
            f"this release pins the {family.name} draft at {family.draft.revision[:12]}"
        )
    if record.get("target_format") == "gguf" and record.get(
        "metadata"
    ) != _metadata_key(_metadata_sources(record["files"])):
        changes.append("the GGUF metadata adapter changed")
    return changes


def _draft(family, draft_override, installed, root):
    """The draft and its downloaded files: --draft-model, as resolved at
    installation, or else the family's pinned draft. An installation keeps
    its draft, from the cache, while it is the one to use, and when a newly
    pinned one cannot be fetched."""
    recorded = installed and installed["sources"]["draft"]
    if recorded and (draft_override or recorded == _draft_identity(family)):
        draft = Repository.recorded(recorded)
        return draft, draft.download(_draft_files(draft, family))
    try:
        with models.hub_errors(f"cannot fetch the {family.name} draft"):
            draft = (
                Repository.resolve(draft_override, installation=root)
                if draft_override
                else Repository.resolve(DRAFTS, family.draft.revision)
            )
            return draft, draft.download(_draft_files(draft, family))
    except models.ModelError as error:
        if not recorded:
            raise
        print(
            f"Warning: cannot fetch the {family.name} draft "
            f"{family.draft.revision[:12]}; keeping the installed one: {error}",
            flush=True,
        )
    draft = Repository.recorded(recorded)
    return draft, draft.download(_draft_files(draft, family))


def _publish(models_root, record, files):
    """The verified assembly for record, rebuilding a damaged one."""
    encoded = gguf.json_bytes(record)
    cache = models_root / ".resolved"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / hashlib.sha256(encoded).hexdigest()
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
        with (stage / "model.json").open("wb") as stream:
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


def _pins(record):
    """The Hub snapshot of each source an assembly links files from, with its
    repository ID: the snapshots its installation pins."""
    sources = {
        (models.hub_folder_name(source["repo"]), source["revision"]): source["repo"]
        for source in record["sources"].values()
        if source["revision"] is not None
    }
    pins = {}
    for entry in record["files"].values():
        snapshot = _snapshot_of(entry["path"])
        key = snapshot and (snapshot.parent.parent.name, snapshot.name)
        if key in sources:
            pins[snapshot] = sources[key]
    return sorted(pins.items())


def _metadata_sources(files):
    """The file records an assembly's derived GGUF metadata comes from: the
    target, then the vision projector."""
    names = [
        n for n in sorted(files) if n.startswith("target/") and n.endswith(".gguf")
    ]
    return [files[name] for name in (*names, "vision/mmproj.gguf") if name in files]


def _metadata_key(sources):
    """The .metadata entry this adapter derives from the source file records."""
    from importlib.metadata import version

    identity = {
        "sources": sources,
        "adapter": models.sha256(Path(gguf.__file__)),
        "tokenizers": version("tokenizers"),
    }
    return hashlib.sha256(gguf.json_bytes(identity)).hexdigest()


def _gguf_metadata(models_root, target, vision):
    """The key and files of target's derived metadata, prepared once and
    derived again when an entry is damaged. Call it under the installation
    lock, which also serializes the derivation."""
    source_paths = [target] + ([vision] if vision else [])
    sources = [_file_record(path) for path in source_paths]
    key = _metadata_key(sources)
    cache = models_root / ".metadata"
    destination = cache / key
    names = {
        "config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja",
    }
    if destination.exists():
        try:
            return key, _metadata_files(destination, names)
        except (models.ModelError, OSError) as error:
            print(f"Deriving damaged GGUF metadata again: {error}", flush=True)
            shutil.rmtree(destination)
    metadata = gguf.Metadata(target)
    vision_metadata = gguf.Metadata(vision) if vision else None
    contents = gguf.tokenizer_files(metadata)
    contents["config.json"] = gguf.json_bytes(
        gguf.model_config(metadata, vision_metadata)
    )
    if [_file_record(path) for path in source_paths] != sources:
        raise models.ModelError("GGUF source changed while reading metadata")
    cache.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".loading-", dir=cache))
    try:
        for name, data in contents.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        hashes = {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        }
        (stage / "files.json").write_bytes(gguf.json_bytes(hashes))
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return key, _metadata_files(destination, names)


def _metadata_files(destination, names):
    """The files of a metadata entry, checked against its files.json."""
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
    if not (models.is_hex_digest(digest, 40) or models.is_hex_digest(digest, 64)):
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
    sources = record.get("sources")
    if (
        record.get("version") != 1
        or not isinstance(record.get("files"), dict)
        or not isinstance(sources, dict)
        or set(sources) != {"target", "draft"}
        or not all(
            isinstance(source, dict)
            and set(source) == {"repo", "revision"}
            and isinstance(source["repo"], str)
            and (
                source["revision"] is None
                or models.is_hex_digest(source["revision"], 40)
            )
            for source in sources.values()
        )
    ):
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
