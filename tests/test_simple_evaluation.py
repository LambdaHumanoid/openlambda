"""Result integrity checks without a simulator or model installation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "openlambda_eval_simple", Path(__file__).parents[1] / "scripts/eval_simple.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.selection = {
            "task": "bend_pick", "level": "level-0", "env": "simple/Test-v0",
            "backend": "amo", "policy": "openlambda_amo", "episode_start": 3,
            "num_episodes": 2, "save_video": False,
        }

    def result(self, lines, returncode=0):
        (self.output / "eval_stats.txt").write_text(lines)
        return evaluation.summarize(self.output, self.selection, returncode)

    def test_failed_episode_is_a_completed_evaluation(self):
        result = self.result("episode_3: True\nepisode_4: False\n")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["exit_code"], 0)

    def test_wrong_or_incomplete_episode_sets_cannot_report_a_score(self):
        for lines in (
            "episode_3: True\n",
            "episode_3: True\nepisode_3: False\nepisode_4: True\n",
            "episode_0: True\nepisode_1: True\n",
            "episode_3: True\nepisode_4: True\nepisode_5: True\n",
            "episode_3: True\nepisode_4: unknown\n",
        ):
            with self.subTest(lines=lines):
                result = self.result(lines)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["success_rate"])
                self.assertNotEqual(result["exit_code"], 0)

    def test_native_failure_overrides_complete_episode_stats(self):
        result = self.result("episode_3: True\nepisode_4: True\n", -15)
        self.assertEqual(result["exit_code"], 143)
        self.assertIsNone(result["success_rate"])

    def test_requested_video_must_exist_for_every_episode(self):
        self.selection["save_video"] = True
        result = self.result("episode_3: True\nepisode_4: False\n")
        self.assertEqual(result["missing_video_episode_ids"], ["episode_3", "episode_4"])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["success_rate"])


if __name__ == "__main__":
    unittest.main()
