import tempfile
import unittest
from pathlib import Path

from dev.tools.weight_preparation_identity import INPUTS, header


class PreparationIdentityTest(unittest.TestCase):
    def test_tracks_preparation_but_not_unrelated_inference_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for paths in INPUTS.values():
                for name in paths:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(name)
            original = header(root)
            (root / "runtime/unrelated.cpp").write_text("new inference policy")
            self.assertEqual(header(root), original)
            path = root / INPUTS["GGUF"][-1]
            path.write_text(path.read_text() + "changed repack")
            changed = header(root)
            self.assertNotEqual(changed, original)
            self.assertEqual(changed.splitlines()[2], original.splitlines()[2])
            self.assertNotEqual(changed.splitlines()[3], original.splitlines()[3])
