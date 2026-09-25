"""Regression tests for freezing an M4 profile into a later project SSOT."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m4_verification as m4  # noqa: E402


class M4ConfigVersionPreservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = yaml.safe_load(m4.CONFIG_PATH.read_text(encoding="utf-8"))

    def freeze(self, project_version: str) -> dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            old_ssot_path = m4.SSOT_PATH
            old_config_path = m4.CONFIG_PATH
            try:
                m4.SSOT_PATH = temp_root / "ssot.yaml"
                m4.CONFIG_PATH = temp_root / "perception_config.yaml"
                m4._freeze_ssot_profile({"config_version": project_version}, self.profile)
                return yaml.safe_load(m4.SSOT_PATH.read_text(encoding="utf-8"))
            finally:
                m4.SSOT_PATH = old_ssot_path
                m4.CONFIG_PATH = old_config_path

    def test_later_project_version_is_not_downgraded(self) -> None:
        result = self.freeze("M5-v1.1")
        self.assertEqual(result["config_version"], "M5-v1.1")
        self.assertEqual(result["perception"]["profile_version"], m4.M4_VERSION)
        self.assertEqual(result["perception"]["runtime_config"]["config_version"], m4.M4_VERSION)

    def test_older_project_version_advances_to_frozen_m4(self) -> None:
        for version in ("M3-v1.0", "M4-v1.2"):
            with self.subTest(version=version):
                self.assertEqual(self.freeze(version)["config_version"], m4.M4_VERSION)

    def test_unknown_project_version_fails_without_silent_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            old_ssot_path = m4.SSOT_PATH
            old_config_path = m4.CONFIG_PATH
            try:
                m4.SSOT_PATH = temp_root / "ssot.yaml"
                m4.CONFIG_PATH = temp_root / "perception_config.yaml"
                with self.assertRaisesRegex(ValueError, "Unsupported SSOT config version"):
                    m4._freeze_ssot_profile({"config_version": "future"}, self.profile)
                self.assertFalse(m4.SSOT_PATH.exists())
            finally:
                m4.SSOT_PATH = old_ssot_path
                m4.CONFIG_PATH = old_config_path


if __name__ == "__main__":
    unittest.main()
