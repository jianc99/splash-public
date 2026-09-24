"""Choose the chat templates once at startup, with later system messages.

Agent clients add instructions during a conversation: a system message after
the first message (request preparation merges the leading system and developer
messages into one, so every later one follows another role). Splash renders
it where it occurs, as a system turn in the template's own markup. Upstream
templates differ: the official Qwen templates raise for it and Unsloth's
Qwen3.6 GGUF template skips it silently.

Every template the tokenizer defines is probed once, when the server starts,
by rendering a canary conversation whose later system message carries a
marker. The outcome is one of:

- ``native``: the marker renders in place; the template is used unchanged.
- ``patched``: the template rejected or dropped the message. A structural
  patch at the construct responsible -- the ``raise_exception`` in the message
  loop's system branch, or the loop condition that skips system messages --
  renders it with the template's own system block. The patch is kept only if
  ordinary conversations still render byte-identically and the canary renders
  in place.
- ``unsupported``: no such construct, or the patched template failed a probe.
  Requests with a later system message are rejected instead of dropped.

Tokenizer files and the tokenizer object are never modified.
"""

import re
from dataclasses import dataclass

from jinja2 import Environment, TemplateSyntaxError, nodes

NATIVE = "native"
PATCHED = "patched"
UNSUPPORTED = "unsupported"

LATER_SYSTEM_UNSUPPORTED = (
    "this model's chat template does not accept system messages after the first message"
)


class ChatTemplateError(ValueError):
    """The tokenizer has no chat template Splash can serve."""


def template_options(
    *, reasoning_effort, preserve_thinking, tools, add_generation_prompt
):
    """Template variables exactly as request preparation passes them."""
    options = {"add_generation_prompt": add_generation_prompt}
    if reasoning_effort is not None:
        options["enable_thinking"] = reasoning_effort != "none"
        if reasoning_effort != "none":
            options["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        options["preserve_thinking"] = preserve_thinking
    if tools:
        options["tools"] = tools
    return options


def has_later_system(messages):
    """Whether normalized messages carry a system message after the first."""
    return any(message["role"] == "system" for message in messages[1:])


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    source: str
    # NATIVE, PATCHED or UNSUPPORTED.
    later_system: str
    # What the unmodified template does with a later system message:
    # IN_PLACE, REJECTS, DROPS or MISPLACES.
    original: str


class ChatTemplates:
    """The tokenizer's templates, probed and chosen once at startup."""

    def __init__(self, tokenizer):
        defined = tokenizer.chat_template
        named = dict(defined) if isinstance(defined, dict) else {None: defined}
        if isinstance(defined, dict) and "default" not in named:
            raise ChatTemplateError(
                "the tokenizer's named chat templates have no default"
            )
        if not all(isinstance(source, str) and source for source in named.values()):
            raise ChatTemplateError("the tokenizer defines no chat template")
        render = _renderer(tokenizer)
        prepared = {}
        for source in named.values():
            if source not in prepared:
                prepared[source] = _prepare(render, source)
        self.templates = {name: prepared[source] for name, source in named.items()}

    def select(self, tools):
        """The template the tokenizer itself would choose for this request."""
        if None in self.templates:
            return self.templates[None]
        if tools and "tool_use" in self.templates:
            return self.templates["tool_use"]
        return self.templates["default"]

    def status(self):
        if None in self.templates:
            return {"later_system": self.templates[None].later_system}
        return {
            "later_system": {
                name: template.later_system for name, template in self.templates.items()
            }
        }

    def describe(self):
        """One line for the startup log."""
        parts = []
        for name, template in self.templates.items():
            if template.later_system == NATIVE:
                text = "renders later system messages in place"
            elif template.later_system == PATCHED:
                text = (
                    "patched to render later system messages in place "
                    f"(it {template.original} them)"
                )
            else:
                text = (
                    "requests with later system messages are rejected "
                    f"(it {template.original} them)"
                )
            parts.append(text if name is None else f"{name} {text}")
        return " · ".join(parts)


# How the unmodified template renders the canary's later system message.
IN_PLACE = "renders"
REJECTS = "rejects"
DROPS = "drops"
MISPLACES = "misplaces"


def _renderer(tokenizer):
    def render(source, messages, options):
        return tokenizer.apply_chat_template(
            messages, chat_template=source, tokenize=False, **options
        )

    return render


def _prepare(render, source):
    original = _later_system(render, source)
    if original == IN_PLACE:
        return ChatTemplate(source, NATIVE, original)
    patched = _patch(render, source, original)
    if patched is not None and _verified(render, source, patched):
        return ChatTemplate(patched, PATCHED, original)
    return ChatTemplate(source, UNSUPPORTED, original)


# Probe conversations. Leading system messages are already merged, as request
# preparation merges them; images use the canonical request part.
_MARKER = "Splash later-system canary 5d0c8e"
_SYSTEM = {"role": "system", "content": "Leading instructions"}
_ASK = {"role": "user", "content": "First question"}
_ANSWER = {"role": "assistant", "content": "First answer"}
_LATER = {"role": "system", "content": _MARKER}
_NEXT = {"role": "user", "content": "Next question"}
_THOUGHT = {"role": "assistant", "content": "First answer", "reasoning_content": "Hm"}
_CALL = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Look it up",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"key": "alpha"}},
        }
    ],
}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "beta"}
_IMAGE = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {"type": "text", "text": "Describe it"},
    ],
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]
# Pairs of a canary and the same conversation without its later system message.
_CANARIES = (
    ([_SYSTEM, _ASK, _ANSWER, _LATER, _NEXT], [_SYSTEM, _ASK, _ANSWER, _NEXT]),
    ([_ASK, _ANSWER, _LATER, _NEXT], [_ASK, _ANSWER, _NEXT]),
)
_ORDINARY = (
    [_ASK],
    [_SYSTEM, _ASK],
    [_SYSTEM, _ASK, _THOUGHT, _NEXT],
    [_SYSTEM, _ASK, _CALL, _RESULT],
    [_SYSTEM, _ASK, _CALL, _RESULT, _THOUGHT, _NEXT],
    [_SYSTEM, _IMAGE],
)
_OPTIONS = (
    *(
        template_options(
            reasoning_effort=effort,
            preserve_thinking=preserve,
            tools=tools,
            add_generation_prompt=True,
        )
        for tools in (None, _TOOLS)
        for effort in (None, "none", "minimal", "low", "medium", "high", "xhigh")
        for preserve in (None, True, False)
    ),
    {"add_generation_prompt": False},
)


def _outcome(render, source, messages, options):
    try:
        return render(source, messages, options)
    except Exception as error:  # templates reject some options themselves
        return type(error), str(error)


def _in_place(rendered):
    marker = rendered.find(_MARKER)
    before = rendered.find(_ANSWER["content"])
    after = rendered.find(_NEXT["content"])
    return (
        rendered.count(_MARKER) == 1
        and 0 <= before
        and before + len(_ANSWER["content"]) <= marker
        and marker + len(_MARKER) <= after
    )


def _later_system(render, source):
    for canary, _ in _CANARIES:
        try:
            rendered = render(source, canary, {"add_generation_prompt": True})
        except Exception:
            return REJECTS
        if _MARKER not in rendered:
            return DROPS
        if not _in_place(rendered):
            return MISPLACES
    return IN_PLACE


def _verified(render, original, patched):
    """Ordinary conversations render byte-identically, and wherever the
    original renders a canary's conversation, the patch renders the canary
    in place."""
    for options in _OPTIONS:
        for messages in _ORDINARY:
            if _outcome(render, original, messages, options) != _outcome(
                render, patched, messages, options
            ):
                return False
        for canary, without in _CANARIES:
            if not isinstance(_outcome(render, original, without, options), str):
                continue
            rendered = _outcome(render, patched, canary, options)
            if not isinstance(rendered, str) or not _in_place(rendered):
                return False
    return True


def _system_block(render, source):
    """The template's own block around a system message's content.

    It is what a leading system message adds to a one-question conversation
    without tools or thinking instructions.
    """
    ask = [_ASK]
    options = {"add_generation_prompt": False, "enable_thinking": False}
    try:
        with_system = render(
            source, [{"role": "system", "content": _MARKER}, *ask], options
        )
        without = render(source, ask, options)
    except Exception:
        return None
    if not with_system.endswith(without):
        return None
    block = with_system[: len(with_system) - len(without)]
    if block.count(_MARKER) != 1:
        return None
    return block.split(_MARKER)


def _literal(text):
    """A Jinja string literal for text."""
    return "'" + text.encode("unicode_escape").decode("ascii").replace("'", "\\'") + "'"


def _patch(render, source, original):
    """The template with its rejecting or dropping construct replaced, or None."""
    if original not in (REJECTS, DROPS):
        return None
    try:
        tags = _tags(source)
        blocks, parents = _blocks(tags)
    except ValueError:
        return None
    # The one construct responsible, by position; none or several is no patch.
    constructs = {
        tag.start: (tag, variable)
        for loop, variable in _message_loops(tags, blocks)
        for tag in (
            _rejections(tags, blocks, loop, variable)
            if original == REJECTS
            else _skips(tags, blocks, parents, loop, variable)
        )
    }
    block = _system_block(render, source) if len(constructs) == 1 else None
    if block is None:
        return None
    ((tag, variable),) = constructs.values()
    prefix, suffix = block
    # The later system message, rendered as the template renders a system
    # message: its own block around the trimmed content.
    later = (
        f"{{{{- {_literal(prefix)} + ({variable}.content|trim) + "
        f"{_literal(suffix)} -}}}}"
    )
    if original == REJECTS:
        # Render where the template raised; its own guard (not the first
        # message, or past the merged leading messages) still applies.
        replacement = later
    else:
        # Render ahead of the condition that skipped the message. Leading
        # system messages are merged into the first message, so every system
        # message after the first is a later one.
        text = source[tag.start : tag.end]
        head = _STATEMENT_HEAD.match(text).group()
        replacement = (
            f"{head}if not loop.first and {variable}.role == 'system' %}}"
            f"{later}{head}el{text[len(head) :]}"
        )
    return source[: tag.start] + replacement + source[tag.end :]


# Structural matching. Tags are located in the source; each construct's
# expression is parsed by Jinja itself, so formatting and quoting do not
# matter.

_ENVIRONMENT = Environment()
_TAG_START = re.compile(r"\{[{%#]")
_TAG_END = {"{": "}}", "%": "%}", "#": "#}"}
_RAW_END = re.compile(r"\{%[-+]?\s*endraw\s*[-+]?%\}")
_STATEMENT_HEAD = re.compile(r"\{%[-+]?\s*")
_KEYWORD = re.compile(r"[A-Za-z_]\w*")
# Statements closed by "end" + keyword; ``set`` is one only without "=".
_BLOCK_STATEMENTS = frozenset(
    {
        "autoescape",
        "block",
        "call",
        "filter",
        "for",
        "generation",
        "if",
        "macro",
        "trans",
        "with",
    }
)
_BRANCH_STATEMENTS = frozenset({"elif", "else", "pluralize"})


@dataclass(frozen=True, slots=True)
class _Tag:
    start: int
    end: int
    # "{" expression, "%" statement or "#" comment.
    kind: str
    # Between the delimiters, without whitespace control.
    text: str

    @property
    def keyword(self):
        word = _KEYWORD.match(self.text) if self.kind == "%" else None
        return word.group() if word else ""


def _tags(source):
    """The template's tags in source order, without raw sections."""
    tags, position = [], 0
    while match := _TAG_START.search(source, position):
        kind = match.group()[1]
        close, cursor, quote = _TAG_END[kind], match.end(), None
        while quote is not None or not source.startswith(close, cursor):
            if cursor >= len(source):
                raise ValueError("unterminated template tag")
            char = source[cursor]
            if quote is not None:
                if char == "\\":
                    cursor += 1
                elif char == quote:
                    quote = None
            elif kind != "#" and char in "'\"":
                quote = char
            cursor += 1
        text = source[match.end() : cursor]
        text = text[1:] if text[:1] in ("-", "+") else text
        text = text[:-1] if text[-1:] in ("-", "+") else text
        tag = _Tag(match.start(), cursor + len(close), kind, text.strip())
        position = tag.end
        if tag.keyword == "raw":
            end = _RAW_END.search(source, position)
            if end is None:
                raise ValueError("unterminated raw block")
            position = end.end()
        else:
            tags.append(tag)
    return tags


def _blocks(tags):
    """Each block's chain of tags (opening, branches, closing) by opening
    index, and the innermost open block at every tag."""
    blocks, parents, stack = {}, [], []
    for index, tag in enumerate(tags):
        keyword = tag.keyword
        parents.append(stack[-1] if stack else None)
        if keyword in _BLOCK_STATEMENTS or (keyword == "set" and "=" not in tag.text):
            stack.append(index)
            blocks[index] = [index]
        elif keyword in _BRANCH_STATEMENTS:
            if not stack:
                raise ValueError("branch outside a block")
            blocks[stack[-1]].append(index)
        elif keyword.startswith("end"):
            if not stack or tags[stack[-1]].keyword != keyword[3:]:
                raise ValueError("unbalanced template block")
            blocks[stack.pop()].append(index)
    if stack:
        raise ValueError("unclosed template block")
    return blocks, parents


def _parse(text):
    try:
        return _ENVIRONMENT.parse(text).body[0]
    except TemplateSyntaxError:
        return None


def _condition(tag):
    """The test of an ``if`` or ``elif`` tag."""
    node = _parse(f"{{% if {tag.text[len(tag.keyword) :]} %}}{{% endif %}}")
    return node.test if node is not None else None


def _message_loops(tags, blocks):
    """(opening index, loop variable) of each ``for`` over ``messages``."""
    for index in blocks:
        if tags[index].keyword != "for":
            continue
        node = _parse(f"{{% {tags[index].text} %}}{{% endfor %}}")
        if (
            node is not None
            and isinstance(node.target, nodes.Name)
            and isinstance(node.iter, nodes.Name)
            and node.iter.name == "messages"
            and node.test is None
        ):
            yield index, node.target.name


def _terms(node, operator):
    """The operands of a chain of ``and`` or ``or``."""
    if isinstance(node, operator):
        return [*_terms(node.left, operator), *_terms(node.right, operator)]
    return [node]


def _compares_role(node, variable, op, role):
    """Whether node is ``variable.role <op> role`` (attribute or item)."""
    if not (
        isinstance(node, nodes.Compare)
        and len(node.ops) == 1
        and node.ops[0].op == op
        and isinstance(node.ops[0].expr, nodes.Const)
        and node.ops[0].expr.value == role
    ):
        return False
    subject = node.expr
    if isinstance(subject, nodes.Getattr):
        name, target = subject.attr, subject.node
    elif isinstance(subject, nodes.Getitem) and isinstance(subject.arg, nodes.Const):
        name, target = subject.arg.value, subject.node
    else:
        return False
    return name == "role" and isinstance(target, nodes.Name) and target.name == variable


def _body(blocks, loop):
    """Tag indices inside a loop's body, before its ``else`` or end."""
    return range(loop + 1, blocks[loop][1])


def _rejections(tags, blocks, loop, variable):
    """The raise inside the loop's branch for system messages."""
    for opening in _body(blocks, loop):
        if tags[opening].keyword != "if":
            continue
        chain = blocks[opening]
        for branch, following in zip(chain, chain[1:]):
            test = (
                _condition(tags[branch])
                if tags[branch].keyword in ("if", "elif")
                else None
            )
            if test is None or not any(
                _compares_role(term, variable, "eq", "system")
                for term in _terms(test, nodes.Or)
            ):
                continue
            for index in range(branch + 1, following):
                node = (
                    _parse(f"{{{{ {tags[index].text} }}}}")
                    if tags[index].kind == "{"
                    else None
                )
                call = node.nodes[0] if node is not None else None
                if (
                    isinstance(call, nodes.Call)
                    and isinstance(call.node, nodes.Name)
                    and call.node.name == "raise_exception"
                ):
                    yield tags[index]


def _skips(tags, blocks, parents, loop, variable):
    """The condition directly in the loop that excludes system messages."""
    for opening in _body(blocks, loop):
        if (
            tags[opening].keyword != "if"
            or parents[opening] != loop
            or len(blocks[opening]) != 2
        ):
            continue
        test = _condition(tags[opening])
        if test is not None and any(
            _compares_role(term, variable, "ne", "system")
            for term in _terms(test, nodes.And)
        ):
            yield tags[opening]
