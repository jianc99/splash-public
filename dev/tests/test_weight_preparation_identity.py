import tempfile
import unittest
from pathlib import Path

from dev.tools.weight_preparation_identity import INPUTS, header


def identities(root):
    """Each preparation's generated identity, by name."""
    return {
        define.removeprefix("SPLASH_").removesuffix("_PREPARATION_ID"): value
        for _, define, value in (
            line.split() for line in header(root).decode().splitlines()[2:]
        )
    }


class PreparationIdentityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for paths in INPUTS.values():
            for name in paths:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)

    def tearDown(self):
        self.directory.cleanup()

    def changed_by(self, name):
        """The preparations whose identity changes with the file name."""
        original = identities(self.root)
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text() if path.exists() else ""
        path.write_text(previous + "changed")
        changed = identities(self.root)
        path.write_text(previous)
        return {kind for kind in original if changed[kind] != original[kind]}

    def test_unrelated_inference_code_keeps_every_identity(self):
        for name in (
            "runtime/metal/kernels/shared/gguf_linear.metal",
            "runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal",
            "runtime/metal/kernels/common/quant_formats.h",
            "runtime/metal/kernels/shared/vision.metal",
            "runtime/model/QwenVision.cpp",
        ):
            with self.subTest(inference=name):
                self.assertEqual(self.changed_by(name), set())

    def test_preparation_code_changes_the_identities_it_defines(self):
        for name, kinds in (
            ("runtime/model/AffineTarget.cpp", {"AFFINE"}),
            ("runtime/model/GgufImage.cpp", {"GGUF"}),
            ("runtime/metal/kernels/shared/gguf_repack.metal", {"GGUF"}),
            ("runtime/model/VisionPreparation.cpp", {"VISION"}),
            ("runtime/model/SafetensorsCheckpoint.mm", {"AFFINE", "VISION"}),
            # The GGUF reader decides the offsets and sizes vision reads.
            ("runtime/model/GgufFile.cpp", {"VISION"}),
            ("runtime/model/GgufFile.hpp", {"VISION"}),
        ):
            with self.subTest(preparation=name):
                self.assertEqual(self.changed_by(name), kinds)
