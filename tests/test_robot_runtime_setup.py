"""Source preparation must preserve user edits and support layered re-runs."""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "setup_robot_runtime", Path(__file__).parents[1] / "scripts/setup_robot_runtime.py"
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def patch(old, new):
    return (
        "diff --git a/controller.txt b/controller.txt\n"
        "--- a/controller.txt\n+++ b/controller.txt\n"
        "@@ -1,3 +1,3 @@\n before\n-" + old + "\n+" + new + "\n after\n"
    )


class SourcePreparationTests(unittest.TestCase):
    def test_layered_patches_are_idempotent_without_touching_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sonic"
            source.mkdir()
            controller = source / "controller.txt"
            controller.write_text("before\nbase\nafter\n")
            unrelated = source / "user-settings.txt"
            unrelated.write_text("preserve this\n")
            first, second = root / "first.patch", root / "second.patch"
            first.write_text(patch("base", "collection"))
            second.write_text(patch("collection", "deployment"))
            self.assertEqual(setup.plan_patches(source, [first, second]), [first, second])
            self.assertEqual(controller.read_text(), "before\nbase\nafter\n")
            subprocess.run(["git", "apply", str(first)], cwd=source, check=True)
            self.assertEqual(setup.plan_patches(source, [first, second]), [second])
            subprocess.run(["git", "apply", str(second)], cwd=source, check=True)
            self.assertEqual(setup.plan_patches(source, [first, second]), [])
            self.assertEqual(controller.read_text(), "before\ndeployment\nafter\n")
            self.assertEqual(unrelated.read_text(), "preserve this\n")

    def test_later_patch_conflict_leaves_original_checkout_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sonic"
            source.mkdir()
            controller = source / "controller.txt"
            original = "before\nbase\nafter\n"
            controller.write_text(original)
            first, second = root / "first.patch", root / "second.patch"
            first.write_text(patch("base", "collection"))
            second.write_text(patch("different", "deployment"))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                setup.plan_patches(source, [first, second])
            self.assertEqual(controller.read_text(), original)

    def test_symlink_target_cannot_modify_another_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sonic"
            source.mkdir()
            outside = root / "outside.txt"
            outside.write_text("before\nbase\nafter\n")
            (source / "controller.txt").symlink_to(outside)
            change = root / "change.patch"
            change.write_text(patch("base", "deployment"))
            with self.assertRaisesRegex(ValueError, "escapes"):
                setup.plan_patches(source, [change])
            self.assertEqual(outside.read_text(), "before\nbase\nafter\n")


if __name__ == "__main__":
    unittest.main()
