# Kokoro TTS Pipeline

DIMER inference wrapper for **`hexgrad/Kokoro-82M`** v1.0 — text-to-speech, 24 kHz mono float32, 54 selectable synthetic voice packs across 9 language codes — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `hexgrad/Kokoro-82M`
- Revision: `f3ff3571791e39611d31c381e3a41a3af07b4987`
- Upstream weight license: Apache-2.0
- Upstream task: text-to-speech (StyleTTS 2 decoder + ISTFTNet vocoder, 82 M parameters), phonemised by `misaki`
- Repository adaptation: **none**; inference only, one language code per pipeline instance, no cloning or voice mixing

## Quick start

```python
import soundfile as sf
from kokoro_tts_pipeline import KokoroTTSPipeline

pipe = KokoroTTSPipeline.from_pretrained(device="cpu", lang_code="a")   # American English; cuda:0 if available and device=None
result = pipe.synthesize("The quick brown fox jumps over the lazy dog.", voice="af_heart", speed=1.0)
sf.write("out.wav", result["audio"], result["sample_rate"])
print(result["duration_s"], result["peak_amplitude"], result["segments"][0]["phonemes"])
```

`audio` is a 1-D float32 array at 24000 Hz. Output is not bit-deterministic (the vocoder adds phase noise); call `torch.manual_seed(n)` before `synthesize` for reproducible samples. `espeak_fallback` in the result says whether out-of-dictionary words get an espeak-ng pronunciation on this host or are dropped.

## Weights layout

```
weights/kokoro-82m/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 for all 58 files (verified on every load)
  config.json                # StyleTTS 2 / ISTFTNet hyper-parameters and the 178-symbol phoneme vocab
  kokoro-v1_0.pth            # 327212226 bytes, PyTorch pickle checkpoint, git-ignored
  voices/<name>.pt           # 54 style-vector packs (af_heart, am_adam, bf_emma, ... zm_yunyang), PyTorch pickles, git-ignored
  README.md, VOICES.md
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any file — checkpoint or voice pack — is missing or its SHA-256 differs from the manifest. The checkpoint and voice packs are pickles, not SafeTensors; the `kokoro` library loads them with `torch.load(weights_only=True)`, and only digest-verified bytes reach the unpickler. There is no Hub-loading path: without a manifest `from_pretrained` raises. To stage the snapshot: `hf download hexgrad/Kokoro-82M --revision f3ff3571791e39611d31c381e3a41a3af07b4987 --local-dir weights/kokoro-82m`, then write the manifest.

`espeak-ng` is not required to be installed system-wide: the venv's `espeakng-loader` package bundles the library, and `misaki` auto-installs spaCy `en_core_web_sm` on first English use (a network call outside this package's control — pre-install it in air-gapped environments).

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed; 14 tests
python -c "from kokoro_tts_pipeline import KokoroTTSPipeline; r = KokoroTTSPipeline.from_pretrained(device='cpu').synthesize('The quick brown fox jumps over the lazy dog.'); print(r['duration_s'], r['peak_amplitude'], r['espeak_fallback'])"
```

Measured on CPU (float32, Windows venv, 2026-09-12): load 6.22 s, 3.25 s of audio in 0.72 s, peak amplitude 0.342.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance, pickle trust boundary and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; upstream records CC BY attributions for two training sources (Koniwa, SIWIS); see `docs/WEIGHTS.md`.
