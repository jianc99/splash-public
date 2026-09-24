"""The model families Splash serves: each architecture's signature and the
DFlash2 draft trained for it.

A target is identified by its own configuration, never by its repository's
name: an MLX config.json states it, and gguf.model_config derives the same
fields from a GGUF header. Legacy Splash packages pack these same layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from . import models
else:
    import models

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


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def family_for(config):
    """The one family whose architecture the target's config states."""
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
