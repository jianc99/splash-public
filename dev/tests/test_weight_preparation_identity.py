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
            for name in (
                "runtime/metal/kernels/shared/gguf_linear.metal",
                "runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal",
                "runtime/metal/kernels/common/quant_formats.h",
            ):
                with self.subTest(inference=name):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("changed inference implementation")
                    self.assertEqual(header(root), original)
            for name in INPUTS["GGUF"]:
                with self.subTest(preparation=name):
                    path = root / name
                    previous = path.read_text()
                    path.write_text(previous + "changed preparation")
                    changed = header(root)
                    self.assertEqual(changed.splitlines()[2], original.splitlines()[2])
                    self.assertNotEqual(
                        changed.splitlines()[3], original.splitlines()[3]
                    )
                    path.write_text(previous)
