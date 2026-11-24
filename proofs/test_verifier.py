"""Verifier behavior tests with a disposable fixture and mocked Lean process.

These do not replace the real lake build / binding checks.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "resident_checker", Path(__file__).resolve().parents[1] / "tools" / "check_resident_proofs.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        # Keep all fixtures inside the owned proof project, not external paths.
        fixture_dir = Path(__file__).resolve().parent / ".lake"
        fixture_dir.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=fixture_dir)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src" / "mega").mkdir(parents=True)
        (self.root / "proofs").mkdir()
        self.modes = ["--zig", "--metal", "--replay-c", "--inkling-zig", "--inkling-metal"]
        self.artifacts = [self.root / "src" / name
                          for name in ("resident_plan.zig", "resident_plan.metal", "replay_limits.h",
                                       "mega/inkling_plan.zig", "mega/inkling_plan.metal")]
        self.expected = [b"generated zig\n", b"generated metal\n", b"generated replay\n",
                         b"generated inkling zig\n", b"generated inkling metal\n"]
        for path in self.artifacts:
            path.write_bytes(b"stale\n")
        self.calls = []
        self.fail_metal = False

    def run_process(self, command, **kwargs):
        self.calls.append(command)
        if command[:2] == ["lake", "build"] or command[:2] == ["lake", "env"]:
            return subprocess.CompletedProcess(command, 0)
        mode = command[-1]
        if mode == "--metal" and self.fail_metal:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0,
                                           stdout=self.expected[self.modes.index(mode)])

    def invoke(self, *args):
        with patch.object(CHECKER, "__file__", str(self.root / "tools" / "check_resident_proofs.py")), \
                patch.object(CHECKER.subprocess, "run", side_effect=self.run_process), \
                patch.object(sys, "argv", ["check_resident_proofs.py", *args]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return CHECKER.main()

    def test_drift_fails_without_rewriting(self):
        self.assertEqual(self.invoke(), 1)
        self.assertEqual([p.read_bytes() for p in self.artifacts], [b"stale\n"] * len(self.artifacts))
        self.assertEqual(self.calls[0], ["lake", "build"])

    def test_matching_bindings_pass(self):
        for path, content in zip(self.artifacts, self.expected):
            path.write_bytes(content)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual([p.read_bytes() for p in self.artifacts], self.expected)

    def test_generate_writes_exact_emitter_bytes(self):
        self.assertEqual(self.invoke("--generate"), 0)
        self.assertEqual([p.read_bytes() for p in self.artifacts], self.expected)

    def test_tile_parameter_drift_is_rejected_in_either_binding(self):
        self.expected = [b"pub const row_tiles = [_]u32{1, 2, 4, 8};\n",
                         b"#define RESIDENT_ROW_TILE_8 8\n",
                         b"replay\n",
                         b"pub const expert_block_m: u32 = 32;\n",
                         b"#define INKLING_EXPERT_BLOCK_M 32u\n"]
        for drifted in (0, 1, 3, 4):
            with self.subTest(binding=drifted):
                for index, (path, content) in enumerate(zip(self.artifacts, self.expected)):
                    bad = content.replace(b"8", b"16") if drifted < 2 else content.replace(b"32", b"64")
                    path.write_bytes(bad if index == drifted else content)
                before = [p.read_bytes() for p in self.artifacts]
                self.assertEqual(self.invoke(), 1)
                self.assertEqual([p.read_bytes() for p in self.artifacts], before)

    def test_worker_limit_drift_is_rejected_in_either_binding(self):
        self.expected = [b"pub const max_workers: u32 = 32;\n",
                         b"#define RESIDENT_MAX_WORKERS 32\n",
                         b"replay\n",
                         b"inkling_z\n",
                         b"inkling_m\n"]
        for drifted in range(2):
            for limit in (b"60", b"33"):
                with self.subTest(binding=drifted, limit=limit):
                    for index, (path, content) in enumerate(zip(self.artifacts, self.expected)):
                        path.write_bytes(content.replace(b"32", limit)
                                         if index == drifted else content)
                    before = [p.read_bytes() for p in self.artifacts]
                    self.assertEqual(self.invoke(), 1)
                    self.assertEqual([p.read_bytes() for p in self.artifacts], before)

    def test_emitter_failure_does_not_partially_generate(self):
        self.fail_metal = True
        self.assertEqual(self.invoke("--generate"), 1)
        self.assertEqual([p.read_bytes() for p in self.artifacts], [b"stale\n"] * len(self.artifacts))


if __name__ == "__main__":
    unittest.main()
