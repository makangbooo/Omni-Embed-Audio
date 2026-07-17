from __future__ import annotations

import unittest

from scripts.audit_7z_listing import audit_members, parse_slt_members


def listing(*paths: tuple[str, str]) -> str:
    members = []
    for path, attributes in paths:
        members.append(
            "\n".join(
                [
                    f"Path = {path}",
                    "Size = 1",
                    f"Attributes = {attributes}",
                ]
            )
        )
    return (
        "7-Zip technical listing\n\n"
        "Path = fixture.7z\n"
        "Type = 7z\n\n"
        "----------\n"
        + "\n\n".join(members)
        + "\n"
    )


class Audit7zListingTest(unittest.TestCase):
    def test_safe_wav_members_are_counted(self) -> None:
        records = parse_slt_members(
            listing(("development", "D"), ("development/one.wav", "A"))
        )
        report = audit_members(records, expected_wav_files=1)
        self.assertEqual(report["wav_members"], 1)
        self.assertEqual(report["directory_members"], 1)

    def test_path_traversal_and_absolute_paths_are_rejected(self) -> None:
        for path in ("../escape.wav", "/absolute.wav", "C:/absolute.wav"):
            with self.subTest(path=path):
                records = parse_slt_members(listing((path, "A")))
                with self.assertRaisesRegex(ValueError, "archive member path"):
                    audit_members(records, expected_wav_files=1)

    def test_non_wav_and_case_collisions_are_rejected(self) -> None:
        records = parse_slt_members(listing(("README.txt", "A")))
        with self.assertRaisesRegex(ValueError, "unexpected non-WAV"):
            audit_members(records, expected_wav_files=1)

        records = parse_slt_members(
            listing(("Audio.wav", "A"), ("audio.wav", "A"))
        )
        with self.assertRaisesRegex(ValueError, "case-insensitive"):
            audit_members(records, expected_wav_files=2)

    def test_missing_separator_or_count_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no member separator"):
            parse_slt_members("Path = fixture.7z\n")
        records = parse_slt_members(listing(("one.wav", "A")))
        with self.assertRaisesRegex(ValueError, "WAV count mismatch"):
            audit_members(records, expected_wav_files=2)


if __name__ == "__main__":
    unittest.main()
