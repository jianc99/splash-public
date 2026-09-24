"""Narrow compatibility fixes for otherwise unmodified upstream templates."""

import hashlib
from functools import lru_cache

# Verified upstream templates; trailing newlines do not affect these templates.
# Qwen3.6-35B-A3B: mlx-community revision 38740b847e4cb78f352aba30aa41c76e08e6eb46
# Qwen3.8-27B: mlx-community revision 3e6447f082e89cc7f0bc6e5441afd38dfce760ff
SYSTEM_POSITION_TEMPLATES = frozenset(
    {
        # Unsloth Qwen3.8-27B GGUF embedded template.
        "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    }
)
# Unsloth Qwen3.6-35B-A3B's embedded template skips later system messages.
SYSTEM_SKIP_TEMPLATE = (
    "55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c"
)
_SYSTEM_SKIP = '{%- if loop.index0 >= num_sys and message.role != "system" and message.role != "developer" %}'
_SYSTEM_RENDER = """{%- if loop.index0 >= num_sys and message.role == "system" %}
    {{- '<|im_start|>system\\n' + (render_content(message.content, true, true)|trim) + '<|im_end|>\\n' }}
    {%- elif loop.index0 >= num_sys and message.role != "system" and message.role != "developer" %}"""
_SYSTEM_REJECTION = "{{- raise_exception('System message must be at the beginning.') }}"
_SYSTEM_MESSAGE = "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}"


@lru_cache(maxsize=16)
def _system_position_template(source):
    digest = hashlib.sha256(source.rstrip("\n").encode()).hexdigest()
    if digest == SYSTEM_SKIP_TEMPLATE:
        if source.count(_SYSTEM_SKIP) != 1:
            raise RuntimeError("verified chat template has an unexpected system branch")
        return source.replace(_SYSTEM_SKIP, _SYSTEM_RENDER, 1)
    if digest not in SYSTEM_POSITION_TEMPLATES:
        return None
    if source.count(_SYSTEM_REJECTION) != 1:
        raise RuntimeError("verified chat template has an unexpected system branch")
    return source.replace(_SYSTEM_REJECTION, _SYSTEM_MESSAGE, 1)


def compatible_chat_template(tokenizer, messages, *, tools=None):
    """Return an override only when a known template rejects or skips a later system.

    Agent clients may introduce instructions during a conversation. Preserve
    their position and the existing Splash rendering, without rewriting the
    tokenizer, its cached files, or unrelated/unknown model templates.
    """
    if not any(message["role"] == "system" for message in messages[1:]):
        return None
    source = tokenizer.get_chat_template(tools=tools)
    return _system_position_template(source) if isinstance(source, str) else None
