# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/kokoro-tts-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kokoro-tts-pipeline/blob/main/tutorials/kokoro_tts_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-hexgrad%2FKokoro--82M-ffcc4d?style=flat)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Upstream](https://img.shields.io/badge/Upstream-hexgrad%2Fkokoro-181717?style=flat&logo=github&logoColor=white)](https://github.com/hexgrad/kokoro)
[![arXiv](https://img.shields.io/badge/arXiv-2306.07691-b31b1b.svg)](https://arxiv.org/abs/2306.07691)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `kokoro_tts_colab.ipynb` | `TASK-INFERENCE` | Kokoro-82M v1.0 text-to-speech on one synthetic English sentence with the `af_heart` voice pack; seeded synthesis, a 24 kHz 16-bit PCM WAV under `outputs/`, duration/peak/phonemes as run-level facts; no quality metric exists or is reported | CPU float32 (CUDA used automatically when available, also float32) | one UTF-8 text file, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `KokoroTTSPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision, stages the 55 git-ignored snapshot files (checkpoint plus 54 voice packs) through the package's `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`, all 58 manifest entries), and reads the voice list from that manifest (`list_voices`). The notebook never imports `kokoro`, `misaki`, `transformers` or `huggingface_hub` directly.
- Trust boundary (MOD12/SEC11): the checkpoint and voice packs are PyTorch pickles, not SafeTensors (`WEIGHT_FORMAT`); the notebook states before loading that digest verification runs first and that the library deserialises them with the weights-only loader (`LOADER_WEIGHTS_ONLY`), and that neither check makes an unverified pickle safe.
- No intrinsic metric exists (EVAL9): speech has no ground truth to score against; the repository ships no metric helper; the notebook says so, names the external judges a real evaluation needs (MOS from listeners, ASR round-trip WER with the judge named), and presents its structural sanity checks (dtype, rate, finite samples, peak within full scale, one segment per line) as plumbing checks only. The model card's smoke observation is quoted as one measurement, not an expected value. Recorded `SHOULD` deviation: EVAL11 (no baseline — none is meaningful for synthesis without a judge).
- Determinism (ENV11/ENV12): the vocoder is stochastic by default; the notebook seeds `torch.manual_seed(SEED)` immediately before `synthesize`, states that the seed controls one host and build only, and exports the seed.
- Ceilings `MAX_TEXT_CHARS`, `MIN_SPEED`/`MAX_SPEED`, `LANG_CODES`/`DEFAULT_LANG_CODE` (voice must match the loaded language), `SAMPLE_RATE` are surfaced before the model runs; the library's 510-phoneme segment cap and the espeak-ng fallback (`espeak_fallback`, dropped out-of-dictionary words when absent) are explained, and the phoneme string is printed as the only in-repository dropped-word check (DAT22/DAT23).
- Documented dependency-chain network call (ENV2 note): `misaki` may download the spaCy `en_core_web_sm` model on first English use; the notebook names it under External access and Troubleshooting.
- The default sample is synthetic text authored in code; `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
