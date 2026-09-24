import copy
import unittest
from pathlib import Path

from jinja2 import TemplateError
from transformers import PreTrainedTokenizerFast

from dev.tests.test_server import FakeRuntime, Harness, TemplateTokenizer, _byte_backend
from server import api_shapes
from server.chat_templates import compatible_chat_template

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/chat_templates"


class ChatTemplateTests(unittest.TestCase):
    def tokenizer(self, name):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_byte_backend({0: "hello"})
        )
        tokenizer.chat_template = (FIXTURES / f"{name}.jinja").read_text()
        return tokenizer

    def test_default_and_named_templates_are_unchanged_for_ordinary_requests(self):
        for name in ("qwen36", "qwen38", "qwen36_gguf", "qwen38_gguf"):
            tokenizer = self.tokenizer(name)
            original = tokenizer.chat_template
            for source in (original, {"default": original, "tool_use": original}):
                tokenizer.chat_template = source
                for messages in (
                    [{"role": "user", "content": "Hello"}],
                    [
                        {"role": "system", "content": "Be brief"},
                        {"role": "user", "content": "Hello"},
                    ],
                ):
                    self.assertIsNone(compatible_chat_template(tokenizer, messages))
                self.assertEqual(tokenizer.chat_template, source)

    def test_later_system_is_rendered_in_place_without_mutating_the_tokenizer(self):
        for name in ("qwen36", "qwen38", "qwen36_gguf", "qwen38_gguf"):
            tokenizer = self.tokenizer(name)
            original = tokenizer.chat_template
            messages = [
                {"role": "system", "content": "Original instructions"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "system", "content": "New instructions"},
                {"role": "system", "content": "More instructions"},
                {"role": "user", "content": "Next question"},
            ]
            saved = copy.deepcopy(messages)
            if name == "qwen36_gguf":
                self.assertNotIn(
                    "New instructions",
                    tokenizer.apply_chat_template(messages, tokenize=False),
                )
            else:
                with self.assertRaises(TemplateError):
                    tokenizer.apply_chat_template(messages, tokenize=False)
            override = compatible_chat_template(tokenizer, messages)
            rendered = tokenizer.apply_chat_template(
                messages, chat_template=override, tokenize=False
            )
            self.assertIn(
                "First answer<|im_end|>\n"
                "<|im_start|>system\nNew instructions<|im_end|>\n"
                "<|im_start|>system\nMore instructions<|im_end|>\n",
                rendered,
            )
            self.assertEqual(tokenizer.chat_template, original)
            self.assertEqual(messages, saved)
            # Legacy Splash templates already support these messages.
            tokenizer.chat_template = override
            self.assertIsNone(compatible_chat_template(tokenizer, messages))

    def test_unknown_templates_keep_their_own_validation(self):
        tokenizer = self.tokenizer("qwen36")
        tokenizer.chat_template += "{# an unverified upstream change #}"
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Later"},
        ]
        self.assertIsNone(compatible_chat_template(tokenizer, messages))
        with self.assertRaises(TemplateError):
            tokenizer.apply_chat_template(messages, tokenize=False)

    def test_named_tool_template_is_selected_before_compatibility(self):
        tokenizer = self.tokenizer("qwen36")
        source = tokenizer.chat_template
        tokenizer.chat_template = {"default": "unknown template", "tool_use": source}
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Later"},
        ]
        self.assertIsNone(compatible_chat_template(tokenizer, messages))
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        override = compatible_chat_template(tokenizer, messages, tools=tools)
        self.assertIsNotNone(override)
        rendered = tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=override, tokenize=False
        )
        self.assertIn("lookup", rendered)
        self.assertIn("<|im_start|>system\nLater<|im_end|>", rendered)

    def test_frontend_uses_compatibility_for_generation_and_image_rendering(self):
        # Exercise the shared entry point, including image placeholder tracking.
        source = (FIXTURES / "qwen36.jinja").read_text()

        class OffsetTokenizer(TemplateTokenizer):
            def __call__(self, text, **kwargs):
                return self.renderer(text, **kwargs)

        tokenizer = OffsetTokenizer(source)
        harness = Harness(FakeRuntime(), tokenizer=tokenizer)
        self.addCleanup(harness.close)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Later"},
        ]
        prompt = harness.app._prepare_prompt({"messages": messages})
        harness.app._render_prompt(prompt, float("inf"))
        override = tokenizer.templates[-1][1]["chat_template"]
        self.assertIsNotNone(override)
        self.assertEqual(tokenizer.renderer.chat_template, source)
        status, _, payload = harness.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": "test-model", "messages": messages},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(tokenizer.templates[-1][1]["chat_template"], override)
        image_messages = [
            *messages,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "unused"},
                    {"type": "text", "text": "Describe"},
                ],
            },
        ]
        # The marker substitution must happen after choosing the compatibility
        # template. Using the original here would raise on the later system.
        _, _, rendered = harness.app._render_image_tokens(
            image_messages, {"chat_template": override}
        )
        self.assertIn("<|im_start|>system\nLater<|im_end|>", rendered)
        self.assertIn("<|image_pad|>", rendered)


class LeadingSystemMergeTests(unittest.TestCase):
    def test_leading_system_and_developer_messages_merge_into_one(self):
        merged = api_shapes.normalize_messages(
            [
                {"role": "system", "content": "One"},
                {"role": "developer", "content": [{"type": "text", "text": "Two"}]},
                {"role": "system", "content": ""},
                {"role": "developer", "content": "Three"},
                {"role": "user", "content": "Ask"},
                {"role": "developer", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ]
        )
        self.assertEqual(
            merged,
            [
                {"role": "system", "content": "One\n\nTwo\n\nThree"},
                {"role": "user", "content": "Ask"},
                {"role": "system", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ],
        )
        self.assertEqual(
            api_shapes.normalize_messages(
                [
                    {"role": "system", "content": "  Only  "},
                    {"role": "user", "content": "x"},
                ]
            )[0],
            {"role": "system", "content": "  Only  "},
        )

    def test_every_api_shape_leads_with_one_system_message(self):
        responses = api_shapes.responses_to_chat_body(
            {
                "instructions": "Base",
                "input": [
                    {"role": "developer", "content": "Developer"},
                    {"role": "user", "content": "Ask"},
                ],
            }
        )["messages"]
        anthropic = api_shapes.anthropic_to_chat_prompt(
            {
                "model": "m",
                "system": "Base",
                "messages": [
                    {"role": "system", "content": "Developer"},
                    {"role": "user", "content": "Ask"},
                ],
            }
        )["messages"]
        for messages in (responses, anthropic):
            with self.subTest(messages=messages):
                self.assertEqual(
                    api_shapes.normalize_messages(messages),
                    [
                        {"role": "system", "content": "Base\n\nDeveloper"},
                        {"role": "user", "content": "Ask"},
                    ],
                )
