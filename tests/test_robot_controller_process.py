"""Reject incompatible processes before any actuator session is bound."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from infer_robot import ControllerProcess


class ControllerProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.binary = self.root / "controller"
        self.binary.write_text("test executable identity")
        self.proc = self.root / "proc"
        self.proc.mkdir()
        (self.proc / "exe").symlink_to(self.binary)
        (self.proc / "environ").write_bytes(b"OPENLAMBDA_RUNTIME=1\0")
        self.args = [str(self.binary), "lo", "decoder.onnx", "reference/", "--zmq-robot-ip", "192.0.2.1", "--hand-type", "none"]
        self.write_args(self.args)
        (self.proc / "stat").write_text("123 (controller with spaces) " + " ".join(["S"] + ["0"] * 18 + ["100"]))
        self.controller = ControllerProcess.__new__(ControllerProcess)
        self.controller.root = self.proc
        self.controller.binary = self.binary
        self.controller.robot_ip = "192.0.2.1"

    def write_args(self, args):
        (self.proc / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")

    def test_adapter_requires_explicit_opt_in_and_exclusive_hand_output(self):
        self.assertEqual(self.controller.snapshot(), "100")
        (self.proc / "environ").write_bytes(b"OPENLAMBDA_RUNTIME=0\0")
        with self.assertRaisesRegex(RuntimeError, "explicitly enabled"):
            self.controller.snapshot()
        (self.proc / "environ").write_bytes(b"OPENLAMBDA_RUNTIME=1\0")
        self.write_args(self.args[:-1] + ["brainco"])
        with self.assertRaisesRegex(RuntimeError, "hand output"):
            self.controller.snapshot()

    def test_rejects_wrong_robot_and_competing_encoder_or_planner(self):
        for extra in (["--encoder-file", "encoder.onnx"], ["--planner-file=planner.onnx"]):
            self.write_args(self.args + extra)
            with self.assertRaisesRegex(RuntimeError, "without encoder/planner"):
                self.controller.snapshot()
        self.write_args([arg.replace("192.0.2.1", "192.0.2.2") for arg in self.args])
        with self.assertRaisesRegex(RuntimeError, "address mismatch"):
            self.controller.snapshot()

    def test_restarted_or_replaced_controller_is_rejected(self):
        self.controller.start_identity = self.controller.snapshot()
        (self.proc / "stat").write_text("123 (controller) " + " ".join(["S"] + ["0"] * 18 + ["101"]))
        with self.assertRaisesRegex(RuntimeError, "restarted"):
            self.controller.check()
        (self.proc / "exe").unlink()
        (self.proc / "exe").symlink_to(self.root / "another-controller")
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.controller.snapshot()


if __name__ == "__main__":
    unittest.main()
