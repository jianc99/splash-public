`goldens.json` holds every SHA-256 the weight tests compare against:

- `gguf_dequantization`: upstream GGML's fp32 dequantization of the reference
  fixture of each GGUF format (`gguf-reference`). They pin the CPU reference
  (`dev/tests/engine/GgufFormatReference.hpp`) to llama.cpp 7ab4ee7, so no
  Splash change touches them.
- `gguf_images`: every image `gguf-preparation` prepares from its dense and
  MoE GGUF fixtures.
- `affine_images`: every image `run_affine_preparation.py` prepares from its
  dense and MoE checkpoints.
- `vision_image`: the file every `run_vision_preparation.py` tower prepares.

A prepared-image hash fails on any change of prepared bytes. When a change
means to change them:

1. Give the new bytes a new preparation identity, or caches of the old bytes
   are served. The identity fingerprints the files listed in `INPUTS` of
   `dev/tools/weight_preparation_identity.py`, so an edit to one of them
   changes it. If the bytes changed through another file, add that file to
   `INPUTS`.
2. Run the tests: `gguf-preparation` prints each mismatching image with its
   new hash, and the Python drivers' assertions print the hashes they got.
3. Check that only the images the change should touch moved and that the
   independent oracles still pass: the CPU reference planes and bytes in
   `gguf-preparation`, the serialized `expected` images in
   `run_affine_preparation.py` and `run_vision_preparation.py`. Then write
   the new hashes here in the same commit, which says which images changed
   and why.

The dequantization hashes change only with the fixture itself. Regenerate
them with a libggml-base built from llama.cpp 7ab4ee7:
`SPLASH_GGML_ORACLE=<libggml-base.dylib> build/engine-tests/gguf-reference dev/tests/fixtures/weight-goldens/goldens.json`
compares the reference with GGML and prints GGML's hash of each format.
