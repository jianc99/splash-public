"""Install supported upstream models: a target repository and the draft
trained for its architecture, as an assembly (assembly.py).

The Hub owns downloads and snapshots; this module inspects the target,
selects its components and decides when to follow the Hub, and the native
source adapters own tensor validation and preparation. A target is
identified by its own metadata (a GGUF header or an MLX config), read before
any weight download, and paired with its family's pinned draft
(families.py); repository names play no part. Every start follows the
target's revision with one Hub request and publishes a new commit's assembly
atomically; the installed assembly starts when the Hub cannot answer or the
new commit cannot be installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import assembly, families, gguf, hub, legacy, models
else:
    import assembly
    import families
    import gguf
    import hub
    import legacy
    import models

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


@dataclass(frozen=True)
class Target:
    """What the target repository supplies, known before any weight download."""

    # The record's target_format: "mlx-affine" or "gguf".
    format: str
    # The record's vision_format: "none", "safetensors" or "gguf".
    vision_format: str
    # The configuration that identifies the target's family.
    config: dict
    # Assembly path -> repository file, for every path a repository file
    # serves (the layout in assembly.py). A GGUF's configuration and tokenizer
    # are derived from it at installation instead.
    files: dict[str, str]


def _root_ggufs(files):
    """The GGUF files at a repository's root; subfolders (split BF16, MTP
    heads) never count."""
    return sorted(n for n in files if "/" not in n and n.lower().endswith(".gguf"))


def select_gguf(files, variant):
    """The target GGUF among a repository's root files, and whether its name
    matched :VARIANT exactly. :VARIANT names the file whose name is the model
    name all of them share, then -VARIANT (X-Q4_K_M for Q4_K_M, not
    X-UD-Q4_K_M); failing that, the only file whose name ends in -VARIANT.
    Without :VARIANT, the only target GGUF. Anything else is an error listing
    the candidates."""
    candidates = [n for n in _root_ggufs(files) if not n.lower().startswith("mmproj")]
    if variant is None:
        if len(candidates) == 1:
            return candidates[0], True
        choice = "select a GGUF with OWNER/REPO:VARIANT"
    else:
        parts = [Path(name).stem.split("-") for name in candidates]
        shared = len(os.path.commonprefix(parts))
        exact = [
            name
            for name, words in zip(candidates, parts, strict=True)
            if "-".join(words[shared:]).lower() == variant.lower()
        ]
        if len(exact) == 1:
            return exact[0], True
        suffix = "-" + variant.lower()
        ending = [n for n in candidates if Path(n).stem.lower().endswith(suffix)]
        if len(ending) == 1:
            # A repository of one GGUF names no variant apart from its model.
            return ending[0], len(candidates) == 1
        choice = "no single GGUF matches :" + variant
    listed = ", ".join(candidates) or "none"
    raise models.ModelError(f"{choice} (files in the repository root: {listed})")


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
    for name in _root_ggufs(repo.files):
        if not name.lower().startswith("mmproj"):
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


def inspect_target(repo, variant, language_only):
    """What the target repository supplies, from metadata alone: a GGUF
    header read by range requests, or an MLX config and shard index. A
    :VARIANT names a GGUF, and so does a repository without a safetensors
    checkpoint."""
    if variant is not None or not any(n.endswith(".safetensors") for n in repo.files):
        return _gguf_target(repo, variant, language_only)
    return _mlx_target(repo, language_only)


def _gguf_target(repo, variant, language_only):
    name, exact = select_gguf(repo.files, variant)
    with repo.open(name) as stream:
        header = gguf.Metadata(stream, tensors=True)
    gguf.require_loadable(header)
    files = {"target/" + name: name}
    vision_header = None
    if not language_only:
        files[assembly.GGUF_VISION], vision_header = select_vision(repo)
        _validate_processor(gguf.processor_config(vision_header))
    config = gguf.model_config(header, vision_header)
    print(
        f"Selected {name} from {repo.name}."
        if exact
        else f"No GGUF is named for :{variant} alone; selected {name} from "
        f"{repo.name}, the only one whose name ends in -{variant}.",
        flush=True,
    )
    return Target("gguf", "none" if language_only else "gguf", config, files)


def _mlx_target(repo, language_only):
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
    files = {
        path: "config.json"
        for path in ("config.json", "target/config.json", "tokenizer/config.json")
    }
    files |= {"tokenizer/" + n: n for n in TOKENIZER_FILES if n in repo.files}
    if not language_only:
        _validate_processor(models.read_json(repo.file("preprocessor_config.json")))
        shards = _weight_files(repo, "vision_tower.")
        if not shards:
            raise models.ModelError(
                f"{repo.name} has no vision tower; use --language-only to serve text only"
            )
        files["vision/config.json"] = "config.json"
        files |= {"vision/" + n: n for n in shards}
    files |= {"target/" + n: n for n in _weight_files(repo)}
    return Target(
        "mlx-affine", "none" if language_only else "safetensors", config, files
    )


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
        name in repo.files and "/" not in name and name.endswith(".safetensors")
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


def prepare(selection):
    """Start selection's model, installing what it needs.

    Every start resolves the target's revision with one Hub request
    (hub.Repository.resolve decides when none is made). The installed
    assembly of that commit starts; a new commit is installed and published
    atomically; when the Hub cannot answer, or a new commit cannot be
    installed, the verified installation starts instead. A legacy Splash
    package is installed by legacy.prepare."""
    kind = models.installation_kind(selection.link)
    if kind == models.PACKAGE:
        legacy.prepare(selection)
        return
    installed = None
    if kind == models.ASSEMBLY:
        try:
            installed = assembly.verify(selection.link)
        except (models.ModelError, OSError) as error:
            print(f"Reinstalling {selection.model}: {error}", flush=True)
    installed_commit = installed and installed["sources"]["target"]["revision"]
    target = hub.Repository.resolve(
        selection.repo_id,
        selection.revision,
        installation=selection.link,
        installed=installed_commit,
    )
    if installed is None and _is_legacy_package(target, selection.model):
        legacy.prepare(selection)
    elif installed is not None and target.revision == installed_commit:
        _start_installed(selection, target, installed)
    else:
        _install_commit(selection, target, installed)


def _is_legacy_package(repo, model):
    """Whether the target repository is a legacy Splash package, whose
    manifest.json names a package format."""
    if "manifest.json" not in repo.files:
        return False
    with hub.as_model_errors(f"cannot install {model}"):
        manifest = models.read_json(repo.file("manifest.json"))
    return legacy.is_package_manifest(manifest)


def _start_installed(selection, target, installed):
    """Start the verified installation of the target's commit, assembled again
    first when this release changes it (_changes)."""
    if target.unreachable_reason:
        print(
            f"Could not reach the Hub ({target.unreachable_reason}); "
            f"using the installed {selection.repo_id}@{target.revision[:12]}.",
            flush=True,
        )
    if changes := _changes(installed, selection):
        print(f"Updating {selection.model}: {'; '.join(changes)}.", flush=True)
        _install_or_keep(
            selection, target, installed, f"update it ({'; '.join(changes)})"
        )
        return
    if (replaced := _retain_installed(selection)) is None:
        print(
            f"Splash model {selection.model} is already installed in {selection.link}",
            flush=True,
        )
        return
    # A concurrent installation replaced or damaged the assembly after it was
    # verified: no verified installation is left to keep.
    print(f"Reinstalling {selection.model}: {replaced}", flush=True)
    _install(selection, target, installed)


def _install_commit(selection, target, installed):
    """Install a commit of the target this selection has not installed; a
    verified installation of another commit is kept if that fails."""
    if target.unreachable_reason:
        print(
            f"Could not reach the Hub ({target.unreachable_reason}); installing "
            f"{selection.repo_id}@{target.revision[:12]} from the Hub cache.",
            flush=True,
        )
    elif installed is not None:
        print(
            f"{selection.repo_id} moved from "
            f"{installed['sources']['target']['revision'][:12]} "
            f"to {target.revision[:12]}.",
            flush=True,
        )
    if installed is None:
        _install(selection, target, None)
    else:
        _install_or_keep(
            selection,
            target,
            installed,
            f"install {selection.repo_id}@{target.revision}",
        )


def _install_or_keep(selection, target, installed, attempt):
    """_install, or else start the verified installation with a warning
    naming the attempt that failed."""
    try:
        _install(selection, target, installed)
    except (models.ModelError, OSError) as error:
        commit = installed["sources"]["target"]["revision"]
        models.warn(
            f"keeping the installed {selection.repo_id}@{commit[:12]}; "
            f"cannot {attempt}: {error}"
        )
        if _retain_installed(selection) is not None:
            raise


def _retain_installed(selection):
    """Verify the installation again under the installation lock, where a
    concurrent installation may have replaced it since it was read, and
    repair its pins. None when it verifies, else why it does not."""
    with models.installation_lock(selection.models_root):
        try:
            installed = assembly.verify(selection.link)
        except (models.ModelError, OSError) as error:
            return error
        hub.repair_pins(selection.link, assembly.pins(installed))
    return None


def _install(selection, repo, installed):
    """Assemble repo's target and its draft, and publish the assembly at the
    selection link."""
    # Every Hub request (header reads, downloads) happens here.
    with hub.as_model_errors(f"cannot install {selection.model}"):
        target = inspect_target(repo, selection.variant, selection.language_only)
        family = families.family_for(target.config)
        draft, files = _draft(family, selection, installed)
        print(
            f"Installing {selection.model} as {family.name} ({target.format}); "
            f"draft {draft.name}; "
            f"vision {'disabled' if selection.language_only else 'enabled'}.",
            flush=True,
        )
        downloaded = repo.download(set(target.files.values()))
    files |= {path: downloaded[name] for path, name in target.files.items()}
    record = {
        "version": 1,
        "model": selection.model,
        "family": family.name,
        "target_format": target.format,
        "vision_format": target.vision_format,
        "sources": {"target": repo.identity(), "draft": draft.identity()},
    }
    models_root = selection.models_root
    models_root.mkdir(parents=True, exist_ok=True)
    # Everything written under the models root is written under the lock.
    with models.installation_lock(models_root):
        if target.format == "gguf":
            record["metadata"], derived = assembly.derived_metadata(models_root, files)
            files |= derived
        # One record per linked file, however many assembly paths link it.
        records = {path: assembly.file_record(path) for path in set(files.values())}
        record["files"] = {name: records[path] for name, path in sorted(files.items())}
        # A new installation requires its pins before it is published; its
        # older pins are retired once it is.
        pins = [
            hub.pin(snapshot, repo_id, selection.link)
            for snapshot, repo_id in assembly.pins(record)
        ]
        models.link_selection(
            selection.link, assembly.build(models_root, record, files)
        )
        hub.retire_other_pins(pins)
        assembly.collect_garbage(models_root)


def _pinned_draft(family):
    return {"repo": families.DRAFTS, "revision": family.draft.revision}


def _changes(installed, selection):
    """What this release changes in a verified assembly, from local files
    alone: another pinned draft for its family (unless --draft-model chose
    the draft), or another derivation of its GGUF metadata."""
    family = families.named(installed["family"])
    changes = []
    if family is None:
        changes.append(f"no supported family is named {installed['family']}")
    elif not selection.draft_model and installed["sources"]["draft"] != _pinned_draft(
        family
    ):
        changes.append(
            f"this release pins the {family.name} draft at {family.draft.revision[:12]}"
        )
    if installed["target_format"] == "gguf" and installed[
        "metadata"
    ] != assembly.metadata_key(installed["files"]):
        changes.append("the GGUF metadata adapter changed")
    return changes


def _draft(family, selection, installed):
    """The draft repository and its downloaded files, by assembly path:
    --draft-model as resolved at installation, or else the family's pinned
    draft. An installation keeps its draft, from the cache, while it is the
    one to use, and when a newly pinned one cannot be resolved."""
    recorded = installed and installed["sources"]["draft"]
    keep = recorded and (selection.draft_model or recorded == _pinned_draft(family))
    repo = None if keep else _new_draft(family, selection, recorded)
    if repo is None:
        repo = hub.Repository.recorded(recorded)
    return repo, _draft_files(repo, family)


def _new_draft(family, selection, recorded):
    """The draft chosen anew: --draft-model, or the family's pinned draft.
    None, with a warning, when it cannot be resolved and the installation
    recorded a draft to keep."""
    try:
        with hub.as_model_errors(f"cannot fetch the {family.name} draft"):
            if selection.draft_model:
                return hub.Repository.resolve(
                    selection.draft_model, installation=selection.link
                )
            return hub.Repository.resolve(families.DRAFTS, family.draft.revision)
    except models.ModelError as error:
        if not recorded:
            raise
        models.warn(
            f"cannot fetch the {family.name} draft {family.draft.revision[:12]}; "
            f"keeping the installed one: {error}"
        )
        return None


def _draft_files(repo, family):
    """The family's draft files in repo, downloaded, by assembly path: its
    folder of the shared repository, or, for a --draft-model directory, the
    directory itself."""
    folder = family.name + "/" if f"{family.name}/config.json" in repo.files else ""
    layers = (f"layer-{i}.bin" for i in range(family.draft.layers))
    names = {folder + name for name in ("config.json", "model.bin", *layers)}
    if not names <= repo.files:
        raise models.ModelError(
            f"{repo.name} does not contain the Splash DFlash2 draft for {family.name}"
        )
    config = models.read_json(repo.file(folder + "config.json"))
    splash = config.get("splash")
    if (
        config.get("architectures") != ["DFlash2DraftModel"]
        or config.get("hidden_size") != dict(family.signature)["hidden_size"]
        or config.get("num_hidden_layers") != family.draft.layers
        or not isinstance(splash, dict)
        or splash.get("format") != models.DRAFT_LAYER_MAGIC
    ):
        raise models.ModelError(
            "draft configuration is incompatible with " + family.name
        )
    return {
        "draft/" + Path(name).name: path for name, path in repo.download(names).items()
    }
