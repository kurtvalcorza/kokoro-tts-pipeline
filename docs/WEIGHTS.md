# Weight provenance and DIMER hosting

- Upstream: `hexgrad/Kokoro-82M`
- Immutable revision: `f3ff3571791e39611d31c381e3a41a3af07b4987`
- Weight format: PyTorch pickle — `kokoro-v1_0.pth` (327212226 bytes; SHA-256 `496dba11…` matches the hash printed in the upstream README) plus 54 `voices/*.pt` style packs. Not SafeTensors.
- Upstream weight license: Apache-2.0; upstream records CC BY 3.0 (Koniwa) and CC BY 4.0 (SIWIS) attributions for part of the training audio.
- Local snapshot: `weights/kokoro-82m/` with `dimer-base-manifest.json` (58 entries, per-file bytes + SHA-256, `totalBytes` 355493259); the Git repository does not vendor the checkpoint or the voice packs.
- Load-time check: `verify_snapshot()` in `src/kokoro_tts_pipeline/pipeline.py` re-hashes every manifest entry — checkpoint and all voice packs — and refuses on any mismatch; `stage_missing_files()` fetches only manifest-listed files at the pinned revision. There is no Hub-loading path for the pickle checkpoint.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint and voice packs in its model store under the upstream license, preserving the upstream CC BY attributions.
- Loader trust boundary: `kokoro==0.9.4` `KModel(config=<local path>, model=<local path>)` and `KPipeline.load_single_voice(<local .pt path>)`, both of which call `torch.load(..., weights_only=True)` (library `model.py:68`, `pipeline.py:147`), so unpickling is restricted to tensors and primitive containers, and only digest-verified bytes are ever passed to it. No remote code; `trust_remote_code` does not apply (no `transformers` model loading). The G2P dependency `misaki` may pip-install spaCy `en_core_web_sm` on first English use.
