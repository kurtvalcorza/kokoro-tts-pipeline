---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
---

# Kokoro-82M v1.0 (DIMER package v0.1.0) — Text-to-Speech Model (Speech Synthesis)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-hexgrad%2FKokoro--82M-ffcc4d?style=flat)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-hexgrad%2Fkokoro-181717?style=flat&logo=github&logoColor=white)](https://github.com/hexgrad/kokoro)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2306.07691-b31b1b.svg)](https://arxiv.org/abs/2306.07691)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-kokoro--tts--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/kokoro-tts-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`hexgrad/Kokoro-82M` v1.0 is an 82 M-parameter open-weight text-to-speech model published 2025-01-27 (upstream README "Releases"), pinned here to revision `f3ff3571791e39611d31c381e3a41a3af07b4987`. Architecturally it is a StyleTTS 2 decoder (Li et al., arXiv:2306.07691) with an ISTFTNet vocoder (Kaneko et al., arXiv:2203.02395), "decoder only: no diffusion, no encoder release" (upstream "Model Facts"): a PL-BERT text encoder (12 layers, hidden 768, `config.json` `plbert`) and a 3-layer prosody predictor read a phoneme string plus a 128-d style vector, predict per-phoneme durations (`max_dur` 50), F0 and energy, and the ISTFTNet generator (upsample rates 10×6, 20-point iSTFT) renders a 24 kHz mono waveform in one pass. The style vector is not learned at inference: it is read from one of 54 pre-computed voice packs (`voices/*.pt`, 178-token vocabulary) and selected by phoneme count, so adaptation is by voice-pack selection only — no training, cloning or in-context conditioning happens in this repository. Grapheme-to-phoneme conversion is done by the `misaki` library (a dictionary G2P for English with an espeak-ng fallback). What this repository adds is packaging: `KokoroTTSPipeline` in `src/kokoro_tts_pipeline/pipeline.py`, digest verification of the checkpoint and all 54 voice packs (`verify_snapshot`, `stage_missing_files`), input validation, a fixed output contract and a CPU smoke run; it exposes no quality metric because none can be computed without external judges.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is text-to-speech: input a Unicode string of up to `MAX_TEXT_CHARS = 2000` characters, a voice name from the manifest's 54 packs (default `af_heart`) and a speed multiplier in 0.5–2.0; output a 1-D float32 waveform at `SAMPLE_RATE = 24000` Hz with its duration, peak amplitude and the grapheme/phoneme segments. Envisioned applications are spoken prompts and read-aloud for accessibility tooling, narration of generated or internal text, audio previews in research prototypes, and low-cost batch synthesis where a small CPU-runnable model is wanted — the upstream card notes deployments in commercial APIs under Apache-2.0. In a larger system the pipeline is a rendering component at the end of a text pipeline; it does not decide what to say, and the language of a run is fixed at `from_pretrained(lang_code=...)` (default `a`, American English).

###### Primary Intended Users

The intended users are machine-learning engineers, application developers and accessibility or content teams integrating speech synthesis into research prototypes, internal enterprise tooling, or the DIMER model workbench. The pipeline assumes its users understand that the voices are synthetic named speakers (upstream `VOICES.md` grades them by training data quality and duration), not recordings of or licences to any real person; that pronunciation of names, acronyms and non-English words depends on a dictionary plus an espeak-ng fallback that may be absent on a host; that quality varies with utterance length (upstream: best at 100–200 tokens, weak under 10–20, rushed over 400); and that the checkpoint is a PyTorch pickle whose deserialisation is a supply-chain surface (Mitigations). It is not designed for hobbyist "type and publish" use.

###### Out-of-scope use cases

1. **Capability boundary:** not voice cloning or speaker adaptation — Kokoro ships no encoder and accepts no reference audio, so it cannot imitate a given person; not speech-to-text (`whisper-asr-pipeline` is the sibling); not singing, sound effects, emotion control or SSML; no word-level timestamps are exposed in v0.1.0. Languages other than the nine `LANG_CODES` are unsupported, and non-English quality is described upstream as "absent or thin".
2. **Input boundary:** only `str` input (`TypeError` otherwise); empty or whitespace-only text and text above 2000 characters are rejected; the voice must be one of the 54 manifest packs and must start with the pipeline's `lang_code` letter (`ValueError` otherwise — no cross-language voice mixing); `speed` outside `MIN_SPEED = 0.5`–`MAX_SPEED = 2.0` is rejected; the library splits text on newlines and truncates any segment above 510 phonemes with a warning, so very long paragraphs should be pre-chunked by the caller.
3. **Decision boundary:** not for safety-critical announcements, emergency alerting, medical or legal read-outs, or any channel where a mispronounced number or dropped word has consequences, without a human listening to the output.

#### Factors

###### Groups

The pipeline is human-centric in one sense — it produces human-sounding speech in voices labelled by language, gender letter (`f`/`m`) and a first name — but it takes no data about people as input and classifies nobody. Upstream discloses the training audio only as "a few hundred hours" of permissive, public-domain and synthetic speech with per-voice quality grades in `VOICES.md`; it reports no breakdown of intelligibility or naturalness by listener group, accent or age, and the corpus is not group-audited. Two consequences transfer to the operator: (a) intelligibility for listeners with hearing impairment, non-native listeners or children must be measured on the operator's own audience before deployment, and (b) the gender and accent presentation of a chosen voice is a design choice the operator makes and owns. This repository measures nothing of the kind.

###### Instrumentation

The training data is audio plus IPA phoneme labels: public-domain and permissively licensed recordings (upstream names Koniwa `tnc` under CC BY 3.0 and SIWIS under CC BY 4.0 as attributed sources) and synthetic audio "generated by closed TTS models from large providers" — so for a large share of the data the instrument is another TTS system, not a microphone, and its artefacts and prosody are inherited. Upstream does not disclose microphones, rooms, sample rates or compression of the recorded portion. At inference the instrument on the input side is the G2P chain: `misaki` dictionary lookup, an espeak-ng fallback for out-of-dictionary words (present on the build host via the bundled `espeakng-loader` DLL, recorded per run as `espeak_fallback`), and a spaCy English model for tokenisation; a missing fallback silently drops unknown words, which the pipeline reports only through the returned `phonemes` string. Output is 24 kHz float32 with no loudness normalisation.

###### Environment

Operating environment: Python 3.12 with `kokoro==0.9.4`, `misaki==0.9.4`, `torch==2.14.0`, `numpy==2.5.3`, `soundfile==0.14.0` (exact pins in `pyproject.toml`). CUDA is optional; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both; the DIMER build environment is CPU-only (`CUDA_VISIBLE_DEVICES=-1`). On this repository's smoke run (Windows venv, CPU, float32, `HF_HUB_OFFLINE=1`) loading and verifying all 58 manifest files took 6.22 s and synthesising "The quick brown fox jumps over the lazy dog." with `af_heart` took 0.72 s for 3.25 s of audio (78000 samples, peak 0.342); `espeak-ng` is not installed on the host, and the fallback came from the venv's `espeakng-loader` package. The CUDA path is not executed in this repository. Data environment: inputs are assumed to be well-formed English prose with ordinary punctuation, of 10–400 tokens per line; heavy abbreviation, code, tables, mixed scripts or other languages under an English `lang_code` degrade pronunciation in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure: speech quality has no intrinsic ground truth, and the standard measures — Mean Opinion Score (naturalness, by human raters), word error rate of an ASR system re-transcribing the output (intelligibility), or speaker-similarity scores — all require an external judge that this repository does not ship. The code exposes only run-level facts: `duration_s`, `num_samples`, `peak_amplitude` (for clipping checks; the smoke run peaked at 0.342) and the `phonemes` string per segment, which lets a caller confirm that no word was dropped by the G2P. Upstream publishes an `EVAL.md` with Elo-style arena results; it is not part of the pinned snapshot and this pipeline quotes none of it. A caller who needs a number should run a reference sentence set through `synthesize`, re-transcribe with `whisper-asr-pipeline` and compute WER, or collect MOS ratings, stating the judge.

###### Decision thresholds

The pipeline applies no decision rule in the classification sense: the model regresses durations, pitch and energy and renders a waveform, and every input that passes validation produces audio. The implicit thresholds are structural — `speed` defaults to 1.0 and rescales predicted durations, the library caps a segment at 510 phonemes and truncates beyond it, and the style vector is indexed by phoneme count. No acceptance threshold on output quality was set during development and none is shipped, because the pipeline cannot score its own output. A deployment that needs to reject bad renders must add its own check — a peak-amplitude ceiling against clipping, a duration-per-character sanity band, or an ASR round-trip WER cut-off — trading the cost of publishing a garbled clip (false positive) against the cost of discarding a good one (false negative).

###### Approaches to uncertainty and variability

This pipeline reports no quality metric, so there is no estimation procedure or dispersion to state; upstream figures are not reproduced. Output is not bit-deterministic by default: the ISTFTNet source module adds Gaussian noise (`noise_std` 0.003) and random initial phase to the harmonic excitation (`kokoro/istftnet.py`), so two calls on the same text and voice differ at the sample level — measured max absolute difference 0.093 between consecutive smoke calls with identical length (78000 samples). The variability is seed-controlled: with `torch.manual_seed(0)` before each call the two waveforms were bit-identical (executed 2026-09-12); the pipeline does not set a seed itself, so a caller who needs reproducible audio must. Durations and phonemes are deterministic given the same text, voice, speed and library versions. No confidence or probability is emitted for any output.

#### Ethical considerations and biases

###### Data

Upstream states the model was trained "exclusively on permissive/non-copyrighted audio data and IPA phoneme labels" — public-domain audio, Apache/MIT-licensed audio, and synthetic audio from closed commercial TTS systems, explicitly excluding "custom voice clones" — totalling a few hundred hours, with CC BY attributions for Koniwa and SIWIS (upstream README "Training Details"); the disclosure stops there: individual speakers of the recorded portion are not enumerated, so whether any recording is of an identifiable person who did not consent to synthesis cannot be ruled out, and the synthetic portion inherits whatever data the closed providers used. This repository distributes code, tests and documentation; the 327 MB `kokoro-v1_0.pth` checkpoint and the 54 voice packs (355 MB in total) are git-ignored and staged locally under `weights/kokoro-82m/` with a manifest, and no sample audio is shipped. The operator must audit the text they submit — it will be spoken verbatim — for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life — it renders speech and decides nothing — and it has not been validated or certified for any such use by anyone. Its only validation is the offline unit suite (14 tests) and one CPU smoke run in this repository. Where a sensitive use is foreseeable — reading medication instructions aloud, public-safety announcements, assistive communication for a non-speaking person — it is admissible only with a human reviewing the rendered audio for every consequential message, an intelligibility evaluation on the target audience, and whatever accessibility or regulatory standard the domain requires.

###### Mitigations

Implemented and inspectable in `src/kokoro_tts_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every one of the 58 files in `weights/kokoro-82m/dimer-base-manifest.json`, including all 54 `voices/*.pt` packs, and raises on the first size or SHA-256 mismatch before any file is opened; `stage_missing_files` fetches only manifest-listed files at the pinned revision and refuses a manifest naming another model; no Hub-loading path exists for the pickle checkpoint — `from_pretrained` without a manifest raises `FileNotFoundError` and `KModel`/`KPipeline` are given explicit local file paths, verified offline with `HF_HUB_OFFLINE=1`. (2) Deserialisation trust boundary — `WEIGHT_FORMAT = "pytorch-pickle"`: the checkpoint and voice packs are `torch.save` pickles, not SafeTensors; the `kokoro==0.9.4` library loads both with `torch.load(..., weights_only=True)` (`model.py:68`, `pipeline.py:147`, INSPECTED), which restricts unpickling to tensors and primitive containers, recorded as `LOADER_WEIGHTS_ONLY = True`; digest verification runs first so only the pinned bytes ever reach the unpickler. (3) Input integrity — `_validate` rejects non-string, empty or over-long text, unknown or language-mismatched voices and out-of-range or non-numeric `speed` before the model runs; an empty synthesis raises `RuntimeError`. (4) Reproducibility — exact `==` pins, `model.eval()`, `model_id`/`model_revision`/`voice`/`speed`/`espeak_fallback` in every result; the seed is left to the caller and the card says so. (5) Refusals — no reference-audio, cloning, training or voice-mixing API is exposed (the library's comma-separated voice averaging is not reachable through `synthesize`). No statistical mitigation is applied because the pipeline does not train.

###### Risks and harms

Mispronunciation and dropped words: out-of-dictionary names, acronyms and numbers are handed to the espeak-ng fallback, or silently dropped when that fallback is unavailable — the operator and the listener bear the harm of a wrong dosage or address read aloud; likelihood is moderate on real-world text. Deceptive audio: although Kokoro cannot clone a specific person, a fluent synthetic voice can still be presented as human or as an official announcement; third parties bear the harm of being misled. Unattributed synthetic media: listeners are not told the audio is synthetic unless the operator says so. Supply-chain: the checkpoint is a pickle; a tampered file would fail digest verification, but an operator who bypasses `verify_snapshot` and loads an unverified `.pth` takes on arbitrary-code-execution risk (mitigated but not eliminated by `weights_only=True`). Dependency side effects: the `misaki` English G2P auto-downloads and pip-installs the spaCy `en_core_web_sm` model on first use if it is absent (`misaki/en.py:501` `spacy.cli.download`) — observed on the build host on 2026-09-12 — so a fresh environment makes a network call the pipeline did not request. Quality degradation outside the 100–200-token band, and rushing above 400 tokens, are documented upstream.

###### Use cases

The pipeline must not be used to impersonate a real person or organisation, to fabricate statements attributed to anyone, or to produce audio presented as a human recording without disclosure — and it must not be marketed as voice cloning, which it cannot do: the model has no speaker encoder and the 54 voices are synthetic named speakers, not real people. It must not be used for harassment, fraud, robocalls, unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, or for surveillance or profiling of any kind (it has no such capability, and adapting it would be a misuse). Any use that violates the Apache-2.0 terms of the upstream weights, the CC BY attribution obligations upstream records for its training sources, or the DIMER deployment terms is prohibited. The developers identify these because convincing speech is the primitive such misuses need; no further prohibited use is identified beyond them.

## Immutable provenance

- Model: `hexgrad/Kokoro-82M`
- Revision: `f3ff3571791e39611d31c381e3a41a3af07b4987`
- Snapshot manifest: `weights/kokoro-82m/dimer-base-manifest.json`, 58 files (4 top-level + 54 voice packs), `totalBytes` 355493259
- `kokoro-v1_0.pth` SHA-256: `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4` (327212226 bytes) — matches the "Model SHA256 Hash" printed in the upstream README
- `config.json` SHA-256: `5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f` (2351 bytes)
- Weight format: PyTorch pickle (`.pth` checkpoint, `.pt` voice packs); loader `kokoro.KModel(repo_id, config=<path>, model=<path>)` and `kokoro.KPipeline(lang_code, repo_id, model=<KModel>)`, both reading local files with `torch.load(weights_only=True)`

## Input/output contract

- `KokoroTTSPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, lang_code="a")` — `lang_code` in `LANG_CODES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")`.
- `synthesize(text, voice="af_heart", *, speed=1.0)` — `text`: non-empty `str` ≤ 2000 characters; `voice`: one of the 54 manifest packs whose first letter equals `lang_code`; `speed`: 0.5–2.0. Returns `{"audio": np.float32[N], "sample_rate": 24000, "num_samples", "duration_s", "peak_amplitude", "segments": [{"graphemes", "phonemes"}, ...], "voice", "lang_code", "speed", "espeak_fallback", "device", "source", "model_id", "model_revision"}`.
- `list_voices(manifest)` — voice names from a manifest dict; `verify_snapshot(path=None)`; `stage_missing_files(path=None, *, allow_download=False, downloader=None)`.
- No metric helper: MOS and ASR-WER need external judges.

## Runtime

- Pins: `kokoro==0.9.4`, `misaki==0.9.4`, `torch==2.14.0`, `numpy==2.5.3`, `huggingface-hub==0.36.2`, `soundfile==0.14.0`; Python 3.12. The build venv carries `torch 2.14.0+cu130`, `espeakng-loader 0.2.4`, `spacy 3.8.16` and `en_core_web_sm 3.8.0` (the last auto-installed by `misaki` on first use).
- Precision: float32 on CPU and CUDA; output 24 kHz mono float32, no loudness normalisation.
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, 2026-09-12): device `cpu`, source `local-snapshot`, 54 voices, `espeak_fallback: true` (via the bundled `espeak-ng.dll`; no system `espeak-ng`); text "The quick brown fox jumps over the lazy dog." (all in-dictionary words; phonemes `ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.`), voice `af_heart`, speed 1.0 → 78000 samples, 3.25 s, peak 0.342, RMS 0.047; load 6.22 s, synthesis 0.72 s, total 6.94 s, exit 0; the WAV written to `outputs/` was deleted after the run. Repeat call: same length, max |Δ| 0.093 unseeded; bit-identical with `torch.manual_seed(0)`.
- Tests: `pytest -q -o addopts= tests` — 14 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Li, Han, Raghavan, Mischler, Mesgarani. StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models. arXiv:2306.07691 (2023). https://arxiv.org/abs/2306.07691
- Kaneko, Tanaka, Kameoka, Seki. iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform. arXiv:2203.02395 (2022). https://arxiv.org/abs/2203.02395
- Upstream card and voice list: https://huggingface.co/hexgrad/Kokoro-82M (README.md, VOICES.md in the pinned snapshot); library: https://github.com/hexgrad/kokoro; G2P: https://github.com/hexgrad/misaki
- StyleTTS 2 reference implementation: https://github.com/yl4579/StyleTTS2
