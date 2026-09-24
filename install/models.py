#!/usr/bin/env python3

"""Verify and atomically install Splash packages from Hugging Face snapshots.

The native model descriptor validates architecture, tensors, headers, and
execution geometry before mapping weights.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

if __package__:
    from . import paths
else:
    import paths

MODELS = paths.MODELS
ALIGNMENT = 16384
HUB_ENDPOINT = "https://huggingface.co"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
TOKENIZER_FILES = {
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
# Format name -> (schema versions, target layer magic). The schema names the
# target: 3 a Qwen3.8 (64 layers, 5 draft layers), 4 a Qwen3.6 MoE (40, 6).
PACKAGE_FORMATS = {
    "splash-packed-q4": ((3,), "MDFL0006"),
    "splash-packed-q4-moe": ((4,), "MDFM0001"),
    "gguf": ((3, 4), "MDGG0001"),
}
# GGUF packages ship no target weights: manifest.target.gguf names a source
# repository and its files; owner/repo:VARIANT selects one, downloaded into the
# Hub cache and prepared into a reusable local cache by the engine.
VARIANT_FORMATS = {"gguf"}
VARIANT_SEPARATOR = ":"
VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class ModelError(RuntimeError):
    pass


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def validate_repo_id(value: str) -> str:
    # Keep argument validation available before the Hub dependency is installed.
    if (
        not isinstance(value, str)
        or not REPO_ID.fullmatch(value)
        or "--" in value
        or ".." in value
        or value.endswith(".git")
    ):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    return value


def parse_repo_id(value: str) -> str:
    try:
        return validate_repo_id(value)
    except ModelError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def split_model_id(value: str) -> tuple[str, str | None]:
    """owner/repo[:variant] -> (repository ID, variant or None)."""
    if not isinstance(value, str):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    repo_id, separator, variant = value.partition(VARIANT_SEPARATOR)
    validate_repo_id(repo_id)
    if not separator:
        return repo_id, None
    if not VARIANT.fullmatch(variant) or ".." in variant:
        raise ModelError(
            "model variant must be a short name such as UD-Q4_K_M "
            f"(owner/repo{VARIANT_SEPARATOR}VARIANT)"
        )
    return repo_id, variant


def validate_model_id(value: str) -> str:
    repo_id, variant = split_model_id(value)
    return repo_id if variant is None else f"{repo_id}{VARIANT_SEPARATOR}{variant}"


def parse_model_id(value: str) -> str:
    try:
        return validate_model_id(value)
    except ModelError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_draft_model(value: str) -> str:
    # A local draft directory is recorded as an absolute path, so the
    # installation it selects does not depend on the working directory.
    if value and (local := Path(value).expanduser()).is_dir():
        return str(local.resolve())
    try:
        return validate_repo_id(value)
    except ModelError:
        raise argparse.ArgumentTypeError(
            "must be a local DFlash2 draft directory or a Hugging Face "
            "repository ID (owner/repo)"
        ) from None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def _validate_records(records, artifact_paths):
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not isinstance(record["path"], str)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        pure = PurePosixPath(record["path"])
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or pure.as_posix() != record["path"]
            or any(character in record["path"] for character in "\\*?[]")
            or any(ord(character) < 32 for character in record["path"])
            or record["path"] == "manifest.json"
            or type(record["size"]) is not int
            or record["size"] <= 0
            or not is_hex_digest(record["sha256"], 64)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        if record["path"] in artifact_paths:
            raise ModelError("runtime package artifact paths are not unique")
        if pure.suffix == ".bin" and record["size"] % ALIGNMENT:
            raise ModelError(
                f"runtime package packed file is unaligned: {record['path']}"
            )
        artifact_paths.add(record["path"])


def validate_package_manifest(path: Path):
    manifest = read_json(path)
    format_ = manifest.get("format")
    format_name = format_.get("name") if isinstance(format_, dict) else None
    layout = PACKAGE_FORMATS.get(format_name) if isinstance(format_name, str) else None
    if (
        layout is None
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in layout[0]
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"].strip()
        or not isinstance(manifest.get("execution_geometry"), dict)
    ):
        raise ModelError("repository is not a supported Splash runtime package")
    expected_format = {
        "section_alignment_bytes": ALIGNMENT,
        "target_layer_magic": layout[1],
        "draft_layer_magic": "MDFD0004",
        "vision_magic": "MDFV0001",
    }
    if any(
        type(format_.get(key)) is not type(value) or format_[key] != value
        for key, value in expected_format.items()
    ):
        raise ModelError("runtime package has an unsupported packed weight format")
    schema = manifest["schema_version"]
    if schema == 4:
        for key, architecture in (
            ("target", "qwen3_5_moe"),
            ("draft", "DFlash2DraftModel"),
        ):
            declaration = manifest.get(key)
            if (
                not isinstance(declaration, dict)
                or declaration.get("architecture") != architecture
            ):
                raise ModelError(
                    f"runtime package has an unsupported {key} architecture"
                )

    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ModelError("runtime package manifest has no artifact list")
    artifact_paths = set()
    _validate_records(records, artifact_paths)
    gguf = (
        manifest.get("target", {}).get("gguf")
        if isinstance(manifest.get("target"), dict)
        else None
    )
    if format_name in VARIANT_FORMATS:
        # The target is a llama.cpp GGUF in another repository; the model ID's
        # :VARIANT names one of its files, prepared locally before inference.
        if (
            not isinstance(gguf, dict)
            or not isinstance(gguf.get("variants"), dict)
            or not gguf["variants"]
        ):
            raise ModelError("runtime package manifest has no GGUF variant table")
        try:
            validate_repo_id(gguf.get("repo_id"))
        except ModelError:
            raise ModelError("runtime package GGUF repository ID is invalid") from None
        revision = gguf.get("revision")
        if revision is not None and not is_hex_digest(revision, 40):
            raise ModelError("runtime package GGUF revision must be a commit hash")
        for name, variant in gguf["variants"].items():
            if (
                not isinstance(name, str)
                or not VARIANT.fullmatch(name)
                or ".." in name
                or not isinstance(variant, dict)
                or set(variant) != {"file", "size", "sha256"}
                or not isinstance(variant["file"], str)
                or not variant["file"].endswith(".gguf")
                or "/" in variant["file"]
                or variant["file"] in ("", ".gguf")
                or type(variant["size"]) is not int
                or variant["size"] <= 0
                or not is_hex_digest(variant["sha256"], 64)
            ):
                raise ModelError("runtime package manifest has an invalid GGUF variant")
        default = gguf.get("default")
        if default is not None and default not in gguf["variants"]:
            raise ModelError("runtime package default variant is not in the table")
        if any(r["path"].startswith("target/") for r in records):
            raise ModelError(
                "runtime package with a GGUF target must not ship target files"
            )
    elif gguf is not None:
        raise ModelError("runtime package format does not support GGUF targets")
    if any(
        parent.as_posix() in artifact_paths
        for name in artifact_paths
        for parent in PurePosixPath(name).parents
    ):
        raise ModelError("runtime package artifact paths overlap")
    target_layers, draft_layers = (64, 5) if schema == 3 else (40, 6)
    required_files = {
        "draft/model.bin",
        "vision/model.bin",
        *(f"draft/layer-{index}.bin" for index in range(draft_layers)),
        *(f"tokenizer/{name}" for name in TOKENIZER_FILES),
    }
    if format_name not in VARIANT_FORMATS:
        required_files.update(
            f"target/{f}"
            for f in (
                "embedding.bin",
                "head.bin",
                *(f"layer-{index}.bin" for index in range(target_layers)),
            )
        )
    missing = required_files - artifact_paths
    if missing:
        raise ModelError(
            "runtime package artifact list is missing: " + ", ".join(sorted(missing))
        )
    return manifest


def gguf_table(manifest):
    """The GGUF variant table of a manifest, or None for packed targets."""
    target = manifest.get("target")
    return target.get("gguf") if isinstance(target, dict) else None


def select_variant(manifest, variant: str | None) -> str | None:
    """The variant a model ID selects in this manifest (None for packed targets)."""
    table = gguf_table(manifest)
    if table is None:
        if variant is not None:
            raise ModelError(
                "this runtime package has no variants; drop the :VARIANT suffix"
            )
        return None
    if variant is None:
        variant = table.get("default")
    if variant is None or variant not in table["variants"]:
        raise ModelError(
            "select a variant with owner/repo:VARIANT; available: "
            + ", ".join(sorted(table["variants"]))
        )
    return variant


def gguf_record(manifest, variant: str):
    """The installed-root record of a variant's GGUF: target/<file>."""
    entry = gguf_table(manifest)["variants"][variant]
    return {
        "path": "target/" + entry["file"],
        "size": entry["size"],
        "sha256": entry["sha256"],
    }


def artifact_records(manifest, variant: str | None, *, installed: bool):
    """Records to verify: the shared files, plus the GGUF in an installed root."""
    records = list(manifest["artifacts"])
    if variant is not None and installed:
        records.append(gguf_record(manifest, variant))
    return records


def verify_artifacts(
    root: Path, manifest, *, full: bool, variant: str | None = None, installed=False
):
    for record in artifact_records(manifest, variant, installed=installed):
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ModelError(f"installed artifact has the wrong size: {record['path']}")
        if path.suffix == ".bin" and record["size"] % ALIGNMENT:
            raise ModelError(f"installed packed file is unaligned: {record['path']}")
        if full and sha256(path) != record["sha256"].lower():
            raise ModelError(f"installed artifact checksum changed: {record['path']}")


def installed_root(
    models: Path, model_id: str, *, revision=None, language_only=False, draft_model=None
) -> Path:
    repo_id, variant = split_model_id(model_id)
    if revision or language_only or draft_model:
        selection = json.dumps(
            [model_id, revision, language_only, draft_model], separators=(",", ":")
        )
        return models / ".selections" / hashlib.sha256(selection.encode()).hexdigest()
    if variant is None:
        return models / repo_id
    return models / f"{repo_id}{VARIANT_SEPARATOR}{variant}"


def installed_snapshot(root: Path) -> Path:
    """The Hub snapshot an installed root points into (symlinked root or variant root)."""
    if root.is_symlink():
        return root.resolve()
    manifest = root / "manifest.json"
    if not manifest.is_symlink():
        raise ModelError(f"installed model root is not a Splash installation: {root}")
    # A Hub snapshot's manifest is itself a link into blobs/. Preserve the
    # snapshot identity by following only our installation link.
    return (manifest.parent / manifest.readlink()).parent.resolve()


def verify_installed(
    models: Path,
    *,
    model_id: str,
    full: bool,
):
    repo_id, variant = split_model_id(model_id)
    root = installed_root(models, model_id)
    manifest = validate_package_manifest(root / "manifest.json")
    variant = select_variant(manifest, variant)
    _snapshot_revision(installed_snapshot(root), repo_id)
    _installed_gguf_snapshot(root, manifest, variant)
    verify_artifacts(root, manifest, full=full, variant=variant, installed=True)
    return model_id


def hub_folder_name(repo_id: str) -> str:
    """The folder of a model repository in the Hub cache."""
    return "models--" + repo_id.replace("/", "--")


def _snapshot_revision(snapshot: Path, model_id: str) -> str:
    if (
        snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != hub_folder_name(model_id)
        or not is_hex_digest(snapshot.name, 40)
    ):
        raise ModelError(
            "installed package is not a snapshot of the requested Hub repository"
        )
    return snapshot.name


def retain_ref(snapshot: Path, model_id: str, installation: Path) -> Path:
    """Pin snapshot for installation (refs/splash/<installation>/<commit>), so
    pruning the Hub cache cannot remove files the installation links."""
    revision = _snapshot_revision(snapshot, model_id)
    # Each installation owns its references; Hub branch updates and other
    # installations must not unpin this installation's current weights.
    owner_path = installation.parent.resolve() / installation.name
    owner = hashlib.sha256(os.fsencode(owner_path)).hexdigest()
    ref = snapshot.parent.parent / "refs" / "splash" / owner / revision
    try:
        existing = ref.read_text()
    except FileNotFoundError:
        pass
    else:
        if existing != revision:
            raise ModelError(f"invalid installed snapshot reference: {ref}")
        return ref
    ref.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ref.parent) as temporary:
        temporary.write(revision.encode())
        temporary.flush()
        try:
            os.link(temporary.name, ref)
        except FileExistsError:
            if ref.read_text() != revision:
                raise ModelError(f"invalid installed snapshot reference: {ref}")
    return ref


def retire_refs(refs):
    """Remove the installation's other pins beside refs. Call it only after
    publishing and verifying the installation refs pin; other installations
    own other folders."""
    try:
        for ref in refs:
            for previous in ref.parent.iterdir():
                if previous not in refs and is_hex_digest(previous.name, 40):
                    previous.unlink()
    except OSError as error:
        # Keeping an old pin uses cache space but cannot invalidate the
        # verified installation or its successfully retained current pin.
        print(
            f"Warning: could not retire old Hub cache references: {error}",
            file=sys.stderr,
        )


def retain_refs(installation: Path, snapshots):
    """Pin a verified installation's (snapshot, repository ID) pairs again,
    repairing a pin that was lost or never written, and retire its older pins.
    Only local files are written; a read-only cache leaves the verified model
    usable, with a warning."""
    try:
        refs = [
            retain_ref(snapshot, repo, installation) for snapshot, repo in snapshots
        ]
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        print(
            "Warning: the verified model can be used, but its Hub cache "
            "reference could not be retained; protect this snapshot from "
            f"external cache pruning: {error}",
            file=sys.stderr,
        )
        return
    retire_refs(refs)


def _download_snapshot(model_id: str, token):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    model_id, variant = split_model_id(model_id)
    options = {
        "repo_id": model_id,
        "repo_type": "model",
        "token": token or False,
        "endpoint": HUB_ENDPOINT,
    }
    for _ in range(3):
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            model_id, revision="main", files_metadata=True
        )
        if not is_hex_digest(info.sha, 40):
            raise ModelError("Hub did not resolve the model to a snapshot commit")
        manifest_file = next(
            (item for item in info.siblings if item.rfilename == "manifest.json"), None
        )
        if manifest_file is None:
            raise ModelError("repository has no Splash runtime package manifest.json")
        if (
            type(manifest_file.size) is not int
            or not 0 < manifest_file.size <= MAX_MANIFEST_BYTES
        ):
            raise ModelError("runtime package manifest.json has an invalid size")
        # Resolve the named revision through the Hub cache before pinning every
        # artifact. A branch update between metadata and download retries before
        # any weights are downloaded.
        manifest_path = Path(
            hf_hub_download(filename="manifest.json", revision="main", **options)
        )
        revision = _snapshot_revision(manifest_path.parent, model_id)
        if revision == info.sha:
            break
    else:
        raise ModelError("Hub main changed repeatedly during installation; retry")

    options["revision"] = revision
    try:
        manifest = validate_package_manifest(manifest_path)
    except ModelError:
        manifest_path = Path(
            hf_hub_download(
                filename="manifest.json",
                force_download=True,
                **options,
            )
        )
        manifest = validate_package_manifest(manifest_path)
    manifest_sha = sha256(manifest_path)
    records = artifact_records(
        manifest, select_variant(manifest, variant), installed=False
    )
    published = {item.rfilename: item for item in info.siblings}
    for record in records:
        item = published.get(record["path"])
        if item is None or item.size != record["size"]:
            raise ModelError(f"Hub artifact does not match manifest: {record['path']}")
        lfs = getattr(item, "lfs", None)
        if lfs is not None and lfs.sha256.lower() != record["sha256"].lower():
            raise ModelError(
                f"Hub artifact hash does not match manifest: {record['path']}"
            )
    snapshot = Path(
        snapshot_download(
            allow_patterns=["manifest.json", *(r["path"] for r in records)],
            **options,
        )
    )
    if sha256(snapshot / "manifest.json") != manifest_sha:
        raise ModelError("runtime package manifest changed during download")
    if _snapshot_revision(snapshot, model_id) != revision:
        raise ModelError("Hub returned a different runtime package revision")
    # Repair only corrupt cached artifacts. A manifest error is deterministic
    # and must not trigger a second download of all model weights.
    for record in records:
        path = snapshot / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or sha256(path) != record["sha256"].lower()
        ):
            hf_hub_download(filename=record["path"], force_download=True, **options)
            verify_artifacts(snapshot, {"artifacts": [record]}, full=True)
    return snapshot.resolve()


def _gguf_matches(path: Path, entry) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and sha256(path) == entry["sha256"].lower()
    )


def _cached_gguf(manifest, variant: str):
    from huggingface_hub import try_to_load_from_cache

    table = gguf_table(manifest)
    entry = table["variants"][variant]
    path = try_to_load_from_cache(
        table["repo_id"], entry["file"], revision=table.get("revision") or "main"
    )
    if not isinstance(path, str):
        return None
    cached = Path(path).absolute()
    if not _gguf_matches(cached, entry):
        return None
    _snapshot_revision(cached.parent, table["repo_id"])
    return cached


def _cached_snapshot(model_id):
    from huggingface_hub import try_to_load_from_cache

    model_id, variant = split_model_id(model_id)
    path = try_to_load_from_cache(model_id, "manifest.json", revision="main")
    if not isinstance(path, str):
        return None
    snapshot = Path(path).parent
    try:
        _snapshot_revision(snapshot, model_id)
        manifest = validate_package_manifest(snapshot / "manifest.json")
        select_variant(manifest, variant)
        verify_artifacts(snapshot, manifest, full=True)
    except (ModelError, OSError):
        return None
    return snapshot.resolve()


def resolve_snapshot(model_id: str):
    validate_model_id(model_id)
    try:
        from huggingface_hub import get_token
    except ImportError as error:
        raise ModelError(
            "missing dependency huggingface_hub; reinstall Splash"
        ) from error
    token = os.environ.get("HF_TOKEN") or get_token()
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        return _download_snapshot(model_id, token)
    except Exception as error:
        if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError)):
            if cached := _cached_snapshot(model_id):
                print("Using a verified cached model while offline.", flush=True)
                return cached
        if isinstance(error, ModelError):
            raise
        raise ModelError(
            hub_error(
                error,
                f"could not download Splash runtime package {model_id}@main",
                token,
            )
        ) from error


def hub_error(error, context: str, token=None) -> str:
    """context: the Hub's reason, with the token redacted (the caller's, or the
    configured one) and, for denied access, how to authenticate."""
    from huggingface_hub import get_token
    from huggingface_hub.errors import HfHubHTTPError

    message = str(error)
    if token := token or os.environ.get("HF_TOKEN") or get_token():
        message = message.replace(token, "[redacted]")
    if (
        isinstance(error, HfHubHTTPError)
        and error.response is not None
        and error.response.status_code in (401, 403)
    ):
        message += (
            "; set HF_TOKEN or run 'hf auth login' with access to this repository"
        )
    return f"{context}: {message}"


@contextmanager
def hub_errors(context: str):
    """Report a Hub, network or cache failure in the block as a ModelError
    (see hub_error), which callers print without a traceback."""
    import httpx

    try:
        yield
    except (OSError, httpx.HTTPError) as error:
        raise ModelError(hub_error(error, context)) from error


def resolve_target_gguf(manifest, variant: str) -> Path:
    """Download the variant's GGUF from its source repository into the Hub
    cache, checked against the manifest; offline, a verified cached copy."""
    import httpx
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.errors import OfflineModeIsEnabled

    token = os.environ.get("HF_TOKEN") or get_token()
    table = gguf_table(manifest)
    entry = table["variants"][variant]
    print(
        f"Fetching {entry['file']} ({entry['size'] / 1e9:.1f} GB) from "
        f"{table['repo_id']}; cached files are reused.",
        flush=True,
    )
    try:
        revision = table.get("revision") or "main"
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            table["repo_id"], revision=revision, files_metadata=True
        )
        item = next((i for i in info.siblings if i.rfilename == entry["file"]), None)
        lfs = getattr(item, "lfs", None)
        if item is None or item.size != entry["size"]:
            raise ModelError(f"Hub GGUF does not match manifest: {entry['file']}")
        if lfs is not None and lfs.sha256.lower() != entry["sha256"].lower():
            raise ModelError(f"Hub GGUF hash does not match manifest: {entry['file']}")
        options = dict(
            repo_id=table["repo_id"],
            filename=entry["file"],
            revision=info.sha if is_hex_digest(info.sha, 40) else revision,
            repo_type="model",
            token=token or False,
            endpoint=HUB_ENDPOINT,
        )
        # Keep the snapshot path: resolving its file symlink to a blob loses the
        # revision we must retain while an installed model uses these weights.
        path = Path(hf_hub_download(**options)).absolute()
        # Hub metadata authenticates the expected file, not the local cache
        # contents. Repair a corrupt cached file once before failing.
        if not _gguf_matches(path, entry):
            path = Path(hf_hub_download(force_download=True, **options)).absolute()
            if not _gguf_matches(path, entry):
                raise ModelError(
                    f"downloaded GGUF checksum or size changed: {entry['file']}"
                )
        _snapshot_revision(path.parent, table["repo_id"])
        return path
    except Exception as error:
        if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError)):
            if cached := _cached_gguf(manifest, variant):
                print("Using the cached GGUF while offline.", flush=True)
                return cached
        if isinstance(error, ModelError):
            raise
        raise ModelError(
            hub_error(error, f"could not download {entry['file']}", token)
        ) from error


@contextmanager
def installation_lock(models: Path):
    lock_path = models / ".install.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another Splash model installation is running; waiting...",
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def install_snapshot(snapshot: Path, destination: Path):
    if destination.exists() and not destination.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".prepare-{destination.name}-", dir=destination.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(snapshot, temporary, target_is_directory=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def install_variant(
    snapshot: Path, destination: Path, manifest, variant: str, gguf: Path
):
    """Publish a variant root: real draft/, vision/, tokenizer/ directories of
    per-file symlinks into the snapshot, target/<file>.gguf linking the cached
    GGUF, and a manifest.json symlink. The engine requires target/ and draft/
    to be real subdirectories of one root. The staged root is verified before
    it replaces the previous one, which is restored if publishing fails."""
    if destination.exists() and not (
        destination.is_symlink() or (destination / "manifest.json").is_symlink()
    ):
        raise ModelError(f"refusing to replace non-symlink model path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".prepare-{destination.name}-", dir=destination.parent)
    )
    retired = None
    try:
        os.symlink(snapshot / "manifest.json", stage / "manifest.json")
        links = {r["path"]: snapshot / r["path"] for r in manifest["artifacts"]}
        links[gguf_record(manifest, variant)["path"]] = gguf
        for name, source in links.items():
            link = stage / name
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source, link)
        verify_artifacts(stage, manifest, full=False, variant=variant, installed=True)

        if destination.exists() or destination.is_symlink():
            retired = Path(
                tempfile.mkdtemp(
                    prefix=f".retired-{destination.name}-", dir=destination.parent
                )
            )
            os.rename(destination, retired / "root")
        os.rename(stage, destination)
    except BaseException:
        if (
            retired is not None
            and not destination.exists()
            and not destination.is_symlink()
        ):
            os.rename(retired / "root", destination)
            retired.rmdir()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)


def _installed_gguf_snapshot(root, manifest, variant):
    """The Hub snapshot an installed variant's GGUF links, checked against the
    manifest's source repository and revision; None for packed targets."""
    if variant is None:
        return None
    table = gguf_table(manifest)
    snapshot = (root / gguf_record(manifest, variant)["path"]).readlink().parent
    revision = _snapshot_revision(snapshot, table["repo_id"])
    if table.get("revision") and revision != table["revision"]:
        raise ModelError("installed GGUF source revision does not match manifest")
    return snapshot


def prepare_legacy(args):
    repo_id, variant = split_model_id(args.model)
    models = args.models.resolve()
    models.mkdir(parents=True, exist_ok=True)
    root = installed_root(models, args.model)
    with installation_lock(models):
        try:
            _snapshot_revision(installed_snapshot(root), repo_id)
            manifest = validate_package_manifest(root / "manifest.json")
            selected = select_variant(manifest, variant)
            gguf_snapshot = _installed_gguf_snapshot(root, manifest, selected)
            verify_artifacts(
                root, manifest, full=False, variant=selected, installed=True
            )
        except (ModelError, OSError):
            if root.exists() and not (
                root.is_symlink() or (root / "manifest.json").is_symlink()
            ):
                raise ModelError(
                    f"cannot identify the local package at {root}; move it aside before installing"
                ) from None
            print(
                f"Installing {args.model}; missing artifacts will be downloaded.",
                flush=True,
            )
            snapshot = resolve_snapshot(args.model)
            manifest = validate_package_manifest(snapshot / "manifest.json")
            selected = select_variant(manifest, variant)
            # A new installation requires its pins before it is published.
            refs = [retain_ref(snapshot, repo_id, root)]
            if selected is None:
                install_snapshot(snapshot, root)
            else:
                gguf = resolve_target_gguf(manifest, selected)
                refs.append(
                    retain_ref(gguf.parent, gguf_table(manifest)["repo_id"], root)
                )
                install_variant(snapshot, root, manifest, selected, gguf)
            manifest = validate_package_manifest(root / "manifest.json")
            verify_artifacts(
                root, manifest, full=False, variant=selected, installed=True
            )
            print(f"Installed verified Splash model {args.model} in {root}")
            retire_refs(refs)
        else:
            print(f"Splash model {args.model} is already installed in {root}")
            snapshots = [(installed_snapshot(root), repo_id)]
            if gguf_snapshot is not None:
                snapshots.append((gguf_snapshot, gguf_table(manifest)["repo_id"]))
            retain_refs(root, snapshots)


def prepare(args):
    if __package__:
        from . import upstream
    else:
        import upstream
    if not upstream.prepare(args):
        prepare_legacy(args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install Splash runtime weights")
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument(
        "--model",
        required=True,
        type=parse_model_id,
        help="Hugging Face repository ID (owner/repo[:variant])",
    )
    parser.add_argument("--revision", help="optional upstream branch, tag or commit")
    parser.add_argument(
        "--draft-model",
        type=parse_draft_model,
        help="override the automatically selected DFlash2 repository or local directory",
    )
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="resolve the upstream model and draft again instead of using the installation",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify").add_argument("--full", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            root = installed_root(
                args.models.resolve(),
                args.model,
                revision=args.revision,
                language_only=args.language_only,
                draft_model=args.draft_model,
            )
            if (root / "model.json").exists():
                if __package__:
                    from . import upstream
                else:
                    import upstream
                upstream.verify(root, full=args.full)
                selected = args.model
            else:
                selected = verify_installed(
                    args.models.resolve(), model_id=args.model, full=args.full
                )
            print(
                f"Splash model {selected} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
