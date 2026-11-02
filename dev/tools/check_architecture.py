#!/usr/bin/env python3
"""Enforce production dependency boundaries, and keep Hugging Face tokens out
of CI."""

from __future__ import annotations

import ast
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".metal"}
INCLUDE = re.compile(r'^\s*#\s*(?:include|import)\s*(["<])([^">]+)[">]', re.MULTILINE)
# Tile, split and kernel-configuration names runtime/ops owns: model/ and
# engine/ size workspaces through the operators, never through these.
OPERATOR_WORKSPACE_POLICY_NAMES: tuple[str, ...] = (
    "prefillAttentionTiles",
    "kVerifySplits",
    "kPrefillAttentionTileRows",
    "moeMaximumTiles",
    "LinearTile",
    "LinearConfig",
    "MoeExpertTile",
    "MoeConfig",
)
OPERATOR_WORKSPACE_POLICY = re.compile(
    r"\b(?:" + "|".join(OPERATOR_WORKSPACE_POLICY_NAMES) + r")\b"
)
# The standard library's steady clocks, which count sleep on macOS, and the
# timed waits that measure on them; production measures durations on
# AwakeClock and wall-clock instants on system_clock.
SLEEP_COUNTING_CLOCK = re.compile(
    r"\b(?:steady_clock|high_resolution_clock|wait_for|try_lock_for"
    r"|try_acquire_for)\b"
)
# The variables huggingface_hub reads a token from, and the repository
# secrets a workflow would pass one in.
HUGGING_FACE_TOKEN = re.compile(
    r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b|\bsecrets\s*(?:\.\s*|\[\s*['\"])HF_"
)
# Engine files that assemble concrete models or parse their options, and the
# status writer, which renders model telemetry types: they may name models and
# mechanisms. Every other engine file is policy.
ENGINE_ASSEMBLY = frozenset(
    {
        "runtime/engine/Bootstrap.hpp",
        "runtime/engine/Bootstrap.mm",
        "runtime/engine/RuntimeResources.hpp",
        "runtime/engine/RuntimeResources.mm",
        "runtime/engine/StreamedMemoryPlan.hpp",
        "runtime/engine/StreamedMemoryPlan.cpp",
        "runtime/engine/NativeArguments.hpp",
        "runtime/engine/NativeArguments.cpp",
    }
)
ENGINE_REPORTING = frozenset({"runtime/engine/Status.hpp", "runtime/engine/Status.cpp"})
# The concrete models' headers, which engine assembly alone includes: neither
# the status writer nor startup does, and engine policy includes model headers
# from the boundary only.
CONCRETE_MODEL_HEADERS = (
    "model/DFlashDraft.hpp",
    "model/Flint.hpp",
    "model/FlintTarget.hpp",
    "model/ModelFactory.hpp",
    "model/Qwen3_6Moe.hpp",
    "model/Qwen3_8.hpp",
    "model/Qwen4Exp.hpp",
    "model/Qwen4ExpTarget.hpp",
    "model/Qwen4Steps.hpp",
    "model/QwenHybridLayout.hpp",
    "model/QwenState.hpp",
    "model/QwenTarget.hpp",
    "model/QwenTargetFiles.hpp",
    "model/QwenTargetLoader.hpp",
    "model/Runtime.hpp",
    "model/WeightStore.hpp",
)
# Identifier word parts (identifier_parts): engine policy decides from what
# the model reports (costs, pools, shape), never from which model
# (MODEL_FAMILY_PARTS, below) or mechanism serves it. An identifier names a
# mechanism when it holds every word of one of these, in any order
# (names_mechanism).
MODEL_MECHANISM_WORDS = (
    ("expert",),
    ("indexer",),
    ("ple",),
    ("history", "stream"),
    ("history", "staging"),
    ("block", "cache"),
    ("sparse", "row"),
    ("hyper", "connection"),
    ("values", "token", "major"),
)
# The model headers engine policy may include: the boundary and the storage
# the KV tier uses.
ENGINE_MODEL_BOUNDARY = frozenset(
    {
        "model/Model.hpp",
        "model/CommandCost.hpp",
        "model/ElasticPool.hpp",
        "model/LatencyHistogram.hpp",
        "model/SlotFile.hpp",
        "model/SsdReaders.hpp",
        "model/KvCopyRead.hpp",
        "model/WeightMemory.hpp",
    }
)
# Comments and string and character literals, which the identifier rules do
# not read; a quote after a word character separates digits.
NOT_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|(?<!\w)'(?:\\.|[^'\\\n])*'", re.S
)
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*")
# A word of an identifier: an uppercase run before a capitalized word, a
# capitalized or lowercase word, an uppercase run, or a number.
IDENTIFIER_PART = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]*[a-z]+|[A-Z]+|\d+")


def production_sources() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "runtime").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def include_paths(path: Path, text: str) -> list[str]:
    """Each include of a runtime source as the path below runtime/ the
    compiler reads: a quoted include beside the including file if it is
    there, else on -Iruntime."""
    result = []
    for delimiter, include in INCLUDE.findall(text):
        beside = posixpath.normpath((path.parent / include).as_posix())
        if delimiter == '"' and Path(beside).is_file():
            include = posixpath.relpath(beside, (ROOT / "runtime").as_posix())
        result.append(posixpath.normpath(include))
    return result


def identifier_parts(identifier: str) -> list[str]:
    """The lower-cased words of an identifier, split at underscores, numbers
    and case changes. An uppercase run before a capitalized word is a word of
    its own (PLERows: ple, rows); a single capital is the start of the word
    after it (DFlashDraft: dflash, draft)."""
    return [part.lower() for part in IDENTIFIER_PART.findall(identifier)]


def family_names(path: Path) -> list[tuple[str, str]]:
    """The name and the draft's repository of each family the installer
    serves (FAMILIES in install/families.py)."""
    families = next(
        node.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "FAMILIES" for target in node.targets)
    )
    return [(family.args[0].value, family.args[1].value) for family in families.elts]


def family_words(
    families: list[tuple[str, str]], headers: tuple[str, ...]
) -> frozenset[str]:
    """The words a family goes by: the first word of a family's name and of
    its draft's repository after the owner, the model's brand (Qwen, Ornith,
    Flint), and those of either that a concrete model header's name holds
    too (DFlash). Numbers are not words, and a name's other words (Flash,
    Next, a size, the repository's owner) are no family's."""
    header_words = {
        part for header in headers for part in identifier_parts(Path(header).stem)
    }

    def words(name: str) -> list[str]:
        return [part for part in identifier_parts(name) if part.isalpha()]

    return frozenset(
        part
        for family, repository in families
        for name in (words(family), words(repository.rpartition("/")[2]))
        for part in name
        if part == name[0] or part in header_words
    )


MODEL_FAMILY_PARTS = family_words(
    family_names(ROOT / "install/families.py"), CONCRETE_MODEL_HEADERS
)


def singular(word: str) -> str:
    """A word without a trailing s, so that a plural names what its singular
    does."""
    return word.removesuffix("s")


def names_mechanism(parts: list[str]) -> bool:
    """Whether an identifier of these words holds every word of a model
    mechanism, in any order, singular or plural."""
    words = {singular(part) for part in parts}
    return any(
        all(singular(word) in words for word in mechanism)
        for mechanism in MODEL_MECHANISM_WORDS
    )


def code_identifiers(text: str) -> list[str]:
    """The identifiers of a source, without its comments, literals and
    include paths."""
    return IDENTIFIER.findall(NOT_CODE.sub(" ", INCLUDE.sub(" ", text)))


def check_engine_policy(
    name: str, text: str, includes: list[str], mechanisms: bool
) -> list[str]:
    """An engine policy file's model dependencies: model headers outside the
    boundary, and identifiers that name a model family or, with `mechanisms`,
    a model mechanism."""
    errors = [
        f"{name}: engine policy depends on model internals {include}"
        for include in includes
        if include.startswith("model/") and include not in ENGINE_MODEL_BOUNDARY
    ]
    for identifier in dict.fromkeys(code_identifiers(text)):
        parts = identifier_parts(identifier)
        if MODEL_FAMILY_PARTS.intersection(parts):
            errors.append(f"{name}: engine policy names a model family ({identifier})")
        elif mechanisms and names_mechanism(parts):
            errors.append(
                f"{name}: engine policy names a model mechanism ({identifier})"
            )
    return errors


def imported_module(call: ast.Call) -> str | None:
    """The module an importlib.import_module or __import__ call names."""
    name = getattr(call.func, "attr", getattr(call.func, "id", None))
    if name in ("import_module", "__import__") and call.args:
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
    return None


def check_server_dependencies() -> list[str]:
    # Request bridges and model-output handling must not import HTTP or prompt
    # preparation. The wire transport stays independent of those adapters.
    forbidden = {
        "backend": {"frontend", "api_shapes"},
        "constraints": {"frontend", "backend", "output", "api_shapes"},
        "output": {"frontend", "backend", "constraints", "api_shapes"},
        "runtime": {"frontend", "backend", "constraints", "output", "api_shapes"},
    }
    errors = []
    for path in sorted((ROOT / "server").glob("*.py")):
        if path.stem == "server":
            continue
        blocked = {"server", *forbidden.get(path.stem, ())}
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.module is None or (node.module == "server" and not node.level):
                    dependencies.update(alias.name for alias in node.names)
                else:
                    dependencies.add(node.module.removeprefix("server.").split(".")[0])
            elif isinstance(node, ast.Import):
                dependencies.update(
                    alias.name.removeprefix("server.").split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.Call) and (module := imported_module(node)):
                dependencies.add(
                    module.lstrip(".").removeprefix("server.").split(".")[0]
                )
        for dependency in sorted(dependencies & blocked):
            errors.append(f"{relative(path)}: imports upper serving layer {dependency}")
    return errors


def check_package_imports() -> list[str]:
    # The server and the installer each import their own modules one way,
    # relatively, never by their package's name. The installer's script
    # entry points name their package in a PEP 366 header, their one use of
    # __package__.
    header = ast.dump(
        ast.parse('__name__ == "__main__" and not __package__', mode="eval").body
    )
    entry_points = {
        "install/launcher.py",
        "install/models.py",
        "install/local.py",
        "install/cached.py",
    }
    errors = []
    for package in ("server", "install"):
        paths = sorted((ROOT / package).glob("*.py"))
        own_names = {package, *(path.stem for path in paths)}
        for path in paths:
            name = relative(path)
            tree = ast.parse(path.read_text())
            allowed = set()
            if name in entry_points:
                allowed = {
                    id(node)
                    for statement in tree.body
                    if isinstance(statement, ast.If)
                    and ast.dump(statement.test) == header
                    for node in ast.walk(statement)
                }
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    modules = []
                for module in modules:
                    if module.split(".")[0] in own_names:
                        errors.append(
                            f"{name}: imports {module} without a relative import"
                        )
                if (
                    isinstance(node, ast.Name)
                    and node.id == "__package__"
                    and id(node) not in allowed
                ):
                    errors.append(f"{name}: reads __package__")
    return errors


def check_workflows() -> list[str]:
    # CI installs public models alone, which need no token, and a token a
    # workflow passes reaches every program its job runs.
    return [
        f"{relative(path)}: passes a Hugging Face token ({match.group()})"
        for path in sorted((ROOT / ".github/workflows").glob("*.y*ml"))
        if (match := HUGGING_FACE_TOKEN.search(path.read_text()))
    ]


def check() -> list[str]:
    errors = check_server_dependencies() + check_package_imports() + check_workflows()
    forbidden_metal_dependencies = ("engine/", "model/", "ops/")
    forbidden_model_dependencies = ("engine/",)
    forbidden_operator_dependencies = ("engine/", "model/")
    concrete_model_symbols = re.compile(r"\bmodel::(?:Qwen\w*|DFlash\w*|Runtime)\b")
    client_names = re.compile(r"\b(?:Claude Code|OpenCode|Codex|Hermes)\b", re.I)

    for path in production_sources():
        name = relative(path)
        text = path.read_text(errors="replace")
        includes = include_paths(path, text)
        for include in includes:
            if include.startswith("tests/"):
                errors.append(f"{name}: production depends on test code {include}")
        if name.startswith("runtime/metal/"):
            for include in includes:
                if include.startswith(forbidden_metal_dependencies):
                    errors.append(
                        f"{name}: Metal depends on production layer {include}"
                    )
        if name.startswith("runtime/model/"):
            for include in includes:
                if include.startswith(forbidden_model_dependencies):
                    errors.append(f"{name}: model depends on engine layer {include}")
            if re.search(r"\b(?:graph|commandGraph)\.add\s*\(", text):
                errors.append(f"{name}: model dispatches a Metal pipeline directly")
        if name.startswith(("runtime/model/", "runtime/engine/")):
            if OPERATOR_WORKSPACE_POLICY.search(text):
                layer = name.split("/")[1]
                errors.append(f"{name}: {layer} owns an operator workspace policy")
        if name.startswith("runtime/ops/"):
            for include in includes:
                if include.startswith(forbidden_operator_dependencies):
                    errors.append(
                        f"{name}: operator depends on upper production layer {include}"
                    )
        if name in ENGINE_REPORTING:
            for include in includes:
                if include in CONCRETE_MODEL_HEADERS:
                    errors.append(
                        f"{name}: engine reporting depends on concrete model {include}"
                    )
        elif name.startswith("runtime/engine/") and name not in ENGINE_ASSEMBLY:
            # Mechanism names are not checked: engine policy still names some.
            errors.extend(check_engine_policy(name, text, includes, mechanisms=False))
        if name == "runtime/main.mm":
            for include in includes:
                if include in CONCRETE_MODEL_HEADERS:
                    errors.append(
                        f"{name}: startup depends on concrete model {include}"
                    )
            if concrete_model_symbols.search(text):
                errors.append(f"{name}: startup names a concrete model type")
        if name.startswith("runtime/engine/") and re.search(
            r"^namespace splash\s*\{", text, re.MULTILINE
        ):
            errors.append(f"{name}: engine declarations leak into root namespace")
        if client_names.search(text):
            errors.append(
                f"{name}: production backend contains client-specific behavior"
            )
        # Every timeout and duration counts time the Mac is awake, as the
        # server's time.monotonic() does.
        if SLEEP_COUNTING_CLOCK.search(text):
            errors.append(f"{name}: measures time on a clock that counts sleep")
    return errors


def stale_policy_names() -> list[str]:
    """The guarded operator policy names no source under runtime/ops uses,
    which the rule would keep guarding for nothing."""
    operators = [
        path.read_text(errors="replace")
        for path in production_sources()
        if relative(path).startswith("runtime/ops/")
    ]
    return [
        name
        for name in OPERATOR_WORKSPACE_POLICY_NAMES
        if not any(re.search(rf"\b{name}\b", text) for text in operators)
    ]


def main() -> int:
    errors = check() + [
        f"rule guards vanished operator policy symbol {name}"
        for name in stale_policy_names()
    ]
    if errors:
        for error in errors:
            print(f"architecture error: {error}", file=sys.stderr)
        return 1
    print("backend architecture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
