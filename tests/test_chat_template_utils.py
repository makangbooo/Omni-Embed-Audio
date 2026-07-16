from __future__ import annotations

import unittest

from AudioRetrieval.models.chat_template_utils import (
    normalize_single_chat_template_output,
)


class ChatTemplateUtilsTest(unittest.TestCase):
    def test_accepts_plain_string(self) -> None:
        self.assertEqual(normalize_single_chat_template_output("rendered"), "rendered")

    def test_unwraps_qwen_single_item_batch(self) -> None:
        self.assertEqual(
            normalize_single_chat_template_output(["rendered"]), "rendered"
        )

    def test_rejects_nested_or_multi_item_outputs(self) -> None:
        for rendered in ([[]], ["first", "second"], [], None):
            with self.subTest(rendered=rendered):
                with self.assertRaisesRegex(TypeError, "one-item list"):
                    normalize_single_chat_template_output(rendered)


if __name__ == "__main__":
    unittest.main()
