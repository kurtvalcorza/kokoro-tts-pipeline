"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "kokoro_tts_pipeline",
    "repo_name": "kokoro-tts-pipeline",
    "stem": "kokoro_tts",
    "notebook_name": "kokoro_tts_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "KokoroTTSPipeline",
    "weights_key": "kokoro-82m",
    "runtime_imports": ["torch", "kokoro", "misaki", "soundfile"],
    "title": "Kokoro-82M — DIMER text-to-speech tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/kokoro-tts-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/kokoro-tts-pipeline/blob/main/tutorials/kokoro_tts_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-hexgrad%2FKokoro--82M-ffcc4d?style=flat",
            "https://huggingface.co/hexgrad/Kokoro-82M",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-hexgrad%2Fkokoro-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/hexgrad/kokoro",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2306.07691-b31b1b.svg", "https://arxiv.org/abs/2306.07691"),
    ],
    "capability": "text-to-speech (24 kHz mono float32 waveform from a named synthetic voice pack) using the pinned Kokoro-82M v1.0 weights",
    "intro": (
        "At inference the `misaki` grapheme-to-phoneme library turns the text into a phoneme string, a 128-d style "
        "vector is read from the selected pre-computed voice pack (indexed by phoneme count), the StyleTTS 2 decoder "
        "predicts per-phoneme durations, pitch and energy, and the ISTFTNet vocoder renders one 24 kHz mono waveform per "
        "text segment; the segments are concatenated. **No adaptation occurs:** no training, fine-tuning, voice cloning, "
        "in-context conditioning, or preprocessing fitting — the only choice is which of the 54 shipped voice packs to "
        "use. What the upstream snapshot supplies is the checkpoint, the configuration and the voice packs; what the "
        "carried pipeline module adds is manifest staging and SHA-256 verification of all 58 snapshot files, input "
        "validation and ceilings, one-language-per-instance voice checking, a fixed output contract and the "
        "`validate_inputs` and `evaluation_report` stage helpers. **Speech quality has no intrinsic metric:** the "
        "pipeline ships no metric helper because naturalness (MOS) needs human listeners and intelligibility (ASR word "
        "error rate) needs an external recogniser; the notebook reports run-level facts (duration, peak amplitude, "
        "phonemes) and no quality score.\n\n"
        "**Trust boundary:** the checkpoint and the voice packs are PyTorch pickle files, not SafeTensors "
        "(`WEIGHT_FORMAT`). In Section 3 digest verification runs before any file is opened, and the `kokoro` library "
        "then deserialises them with the weights-only loader (`LOADER_WEIGHTS_ONLY`), which restricts unpickling to "
        "tensors and primitive containers. Path-safety and digest checks do not make an unverified pickle safe; only the "
        "pinned, verified bytes ever reach the unpickler, and there is no fallback to a different download. "
        "`from_pretrained` then loads from that verified directory with `lang_code='a'` and reports `espeak_fallback`: "
        "whether the espeak-ng fallback for out-of-dictionary words bound on this host (the pinned `espeakng-loader` "
        "wheel bundles the library)."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic English "
        "sentence (or upload your own text), stage and digest-verify the immutable upstream snapshot including every "
        "voice pack, surface the pipeline's ceilings and voice/language rules and validate the request into an input "
        "manifest, synthesise speech through the public API with an explicit seed, read the output contract correctly, "
        "write a playable WAV under `outputs/`, read from the machine-readable evaluation report why no metric is "
        "reported and what external judges a real evaluation needs, and export machine-readable results plus provenance."
    ),
    "exclusions": (
        "voice cloning or speaker adaptation (Kokoro has no speaker encoder and accepts no reference audio), "
        "speech-to-text (the `whisper-asr-pipeline` sibling covers that), voice mixing, SSML or emotion control, "
        "word-level timestamps, languages other than the pipeline's `lang_code` at load time, or any quality score. "
        "The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline does not change precision by device). The model card's CPU smoke loaded and verified the 58-file snapshot in 6.22 s and rendered 3.25 s of audio in 0.72 s, so the one-sentence default runs in seconds on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 355 MB snapshot (327 MB checkpoint plus 54 voice packs) are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a phoneme string is; why a synthesised waveform has no ground truth to score against.",
        "- **Data:** the default sample is one synthetic English sentence authored in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file of at most 2,000 characters; the pipeline splits it on newlines and synthesises each line as a segment. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API. The text you submit will be spoken verbatim.",
        "- **Dependency network call:** outside this package's control, the `misaki` G2P library downloads the spaCy `en_core_web_sm` model once on first English use.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the sample text or optional BYOD\n\n"
                "The default sample is **synthetic**: one English pangram written in this cell — the same sentence the "
                "model card's CPU smoke used, chosen because every word is in the `misaki` dictionary, so the espeak-ng "
                "fallback is not needed to pronounce it. It exists to prove the code path, not to measure anything: it "
                "ships **no reference recording**, so the audio it produces is smoke/sanity evidence that the pipeline "
                "works, never a quality measurement and never benchmark evidence. The voice, the speed and the seed are "
                "Colab form parameters so they can be changed without editing code; their allowed ranges are checked "
                "against the carried module in Section 5.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file of at most "
                "`MAX_TEXT_CHARS` characters; the pipeline splits it on newlines and synthesises each line as a segment, "
                "and the library truncates any single segment above 510 phonemes with a warning, so keep lines to a "
                "sentence or two. The upload stays inside this runtime. Numbers, acronyms and names outside the "
                "dictionary are pronounced by the espeak-ng fallback when it is available on the host and are otherwise "
                "dropped — Section 6 shows how to check."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VOICE = 'af_heart'  # @param {{type:\"string\"}}\n"
                "SPEED = 1.0  # @param {{type:\"number\"}}\n"
                "SEED = 0  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    text = io.TextIOWrapper(io.BytesIO(uploaded[sample_name]), encoding='utf-8').read().strip()\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    text = 'The quick brown fox jumps over the lazy dog.'\n"
                "    sample_name = 'synthetic_pangram'\n"
                "    sample_kind = 'synthetic (authored in this cell; the model card smoke sentence)'\n"
                "text_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'chars': len(text), 'lines': len(text.splitlines()), 'text_sha256': text_sha256, 'voice': VOICE, 'speed': SPEED, 'seed': SEED}})\n"
                "print(text[:200])"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks "
                "`synthesize` applies — both route through the same private `_check_inputs` — so the text type, "
                "non-emptiness, the character ceiling `MAX_TEXT_CHARS`, voice membership in the digest-verified voice "
                "inventory, the one-language-per-instance rule and the `MIN_SPEED`–`MAX_SPEED` range are enforced "
                "identically. The voice inventory is passed in as `voices=pipe.voices`, which `list_voices` read from the "
                "verified manifest in Section 3 — the only authoritative voice list. The helper returns an **input "
                "manifest** naming the schema and ceilings, the request's character count and segment count, the voice, "
                "speed and language code in force, and the verdict; it is written to "
                "`outputs/{stem}_input_manifest.json`. `LANG_CODES` are the nine language codes the library supports, of "
                "which `DEFAULT_LANG_CODE` (`a`, American English) is what this notebook loads, so a voice must start "
                "with that letter (`af_…`/`am_…`) and a British `bf_…` voice is rejected by design rather than silently "
                "mixed — the cell demonstrates exactly that rejection and records the pipeline's own error message as a "
                "finding. `SAMPLE_RATE` is the fixed 24 kHz output rate. The notebook never trims or alters the text; the "
                "library's 510-phoneme segment cap is applied inside the pipeline and is only visible afterwards through "
                "the returned `phonemes`."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MIN_SPEED': MIN_SPEED, 'MAX_SPEED': MAX_SPEED, 'SAMPLE_RATE': SAMPLE_RATE, 'LANG_CODES': LANG_CODES, 'DEFAULT_LANG_CODE': DEFAULT_LANG_CODE, 'DEFAULT_VOICE': DEFAULT_VOICE}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(text, VOICE, speed=SPEED, voices=pipe.voices, lang_code=pipe.lang_code, names=[sample_name])\n"
                "# Demonstrate the cross-language rejection; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(text, 'bf_emma', speed=SPEED, voices=pipe.voices, lang_code=pipe.lang_code)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'wrong-language-voice-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2, ensure_ascii=False))"
            ),
        },
        {
            "md": (
                "## 6. Synthesise, write the WAV, and read the output correctly\n\n"
                "`synthesize(text, voice=…, speed=…)` returns `audio` (a 1-D float32 NumPy array at `sample_rate` 24000 "
                "Hz), `num_samples`, `duration_s`, `peak_amplitude`, `segments` (one `graphemes`/`phonemes` pair per "
                "newline-split segment), the `voice`, `lang_code` and `speed` used, `espeak_fallback`, the device, and "
                "the model identity. **The output is not bit-deterministic by default:** the vocoder adds Gaussian noise "
                "and a random initial phase to its harmonic excitation, so two unseeded calls differ at the sample level "
                "(the model card measured a maximum absolute difference of 0.093 between consecutive smoke calls of "
                "identical length) while durations and phonemes stay the same; the pipeline sets no seed, so this cell "
                "calls `torch.manual_seed(SEED)` immediately before synthesis — the card recorded bit-identical waveforms "
                "across two seeded calls on the same host. Seeding does not make the audio identical across devices, "
                "PyTorch builds or CPU kernels. The sanity checks below (right dtype, rate and shape, finite samples, "
                "peak within full scale, one segment per line) are falsifiable plumbing checks, not a quality result, and "
                "the model card's smoke observation (78,000 samples, 3.25 s, peak 0.342 on the same sentence, unseeded "
                "CPU) is quoted as one measurement on that host, not an expected value. The `phonemes` string is the only "
                "in-repository way to see whether the G2P dropped a word — with `espeak_fallback` `False`, "
                "out-of-dictionary words vanish silently. The WAV written to `outputs/` is 16-bit PCM at 24 kHz via the "
                "pinned `soundfile`; the inline player below appears only in an IPython front end."
            ),
            "code": (
                "import time\n\n"
                "torch.manual_seed(SEED)\n"
                "started = time.perf_counter()\n"
                "result = pipe.synthesize(text, voice=VOICE, speed=SPEED)\n"
                "elapsed = time.perf_counter() - started\n"
                "audio = result['audio']\n"
                "checks = {{\n"
                "    'audio_is_float32_1d': isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1,\n"
                "    'sample_rate_matches_contract': result['sample_rate'] == SAMPLE_RATE,\n"
                "    'all_samples_finite': bool(np.isfinite(audio).all()),\n"
                "    'peak_within_full_scale': 0.0 < result['peak_amplitude'] <= 1.0,\n"
                "    'one_segment_per_line': len(result['segments']) == len([line for line in text.splitlines() if line.strip()]),\n"
                "    'duration_consistent': abs(result['duration_s'] - result['num_samples'] / SAMPLE_RATE) < 1e-6,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'synthesize output failed a sanity check: {{checks}}')\n"
                "wav_path = 'outputs/{stem}_sample.wav'\n"
                "soundfile.write(wav_path, audio, result['sample_rate'], subtype='PCM_16')\n"
                "wav_sha256 = hashlib.sha256(Path(wav_path).read_bytes()).hexdigest()\n"
                "print({{key: value for key, value in result.items() if key not in ('audio', 'segments')}})\n"
                "print({{'seconds': round(elapsed, 3), 'audio_seconds_per_wall_second': round(result['duration_s'] / elapsed, 2), 'rms': round(float(np.sqrt(np.mean(np.square(audio)))), 4), 'checks': checks}})\n"
                "for index, segment in enumerate(result['segments']):\n"
                "    print(f\"segment {{index}}: graphemes={{segment['graphemes'][:80]!r}}\")\n"
                "    print(f\"segment {{index}}: phonemes ={{segment['phonemes'][:80]!r}}\")\n"
                "print({{'wav': wav_path, 'wav_sha256': wav_sha256, 'subtype': 'PCM_16'}})\n"
                "try:\n"
                "    from IPython.display import Audio, display\n"
                "    display(Audio(audio, rate=result['sample_rate']))\n"
                "except ImportError:\n"
                "    print('inline player unavailable outside an IPython front end; open the WAV file instead')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure**: speech quality has no ground truth to compare against, so the verdict is always "
                "`not-measurable` and the report states what would make the task measurable — Mean Opinion Score ratings "
                "from human listeners for naturalness, or an independent speech recogniser to re-transcribe the waveform "
                "and compute word error rate against the input text for intelligibility (for example the "
                "`whisper-asr-pipeline` sibling), over a reference sentence set and with the judge named. Supplying a "
                "reference does not change the verdict, because no metric helper exists to score it; the helper records "
                "that in `reason` rather than inventing a number. The run-level facts it carries — segment count and "
                "duration — are observations, not scores. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2, ensure_ascii=False))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: speech has no intrinsic ground truth, the sample has no reference recording, and the repository ships no metric helper; MOS needs human listeners and intelligibility needs an external ASR judge.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Three files are written under `outputs/` beside the input manifest: the WAV from Section 6, the "
                "evaluation report from Section 7, and one JSON record with the WAV path and its SHA-256 (so the audio "
                "can be tied to this record), the run-level facts (`num_samples`, `duration_s`, `peak_amplitude`, RMS, "
                "wall time), the per-segment graphemes and phonemes, the request (`voice`, `lang_code`, `speed`, `seed`), "
                "`espeak_fallback`, the sanity checks, the ceilings in force, the input manifest, the evaluation report, "
                "the sample identity and text digest, the notebook's source (repository, revision, embedded module "
                "digest, generator), the model identifier, the immutable model revision, the model licence, the weight "
                "format and loader trust facts, the verified snapshot summary, and the runtime identity (Python, `torch`, "
                "`kokoro`, `misaki`, `soundfile`, device, dtype). No credentials are involved in any step, so none can "
                "reach the export."
            ),
            "code": (
                "payload = {{\n"
                "    'wav': {{'path': wav_path, 'sha256': wav_sha256, 'subtype': 'PCM_16', 'sample_rate': result['sample_rate']}},\n"
                "    'audio': {{'num_samples': result['num_samples'], 'duration_s': result['duration_s'], 'peak_amplitude': result['peak_amplitude'], 'rms': float(np.sqrt(np.mean(np.square(audio))))}},\n"
                "    'segments': result['segments'],\n"
                "    'request': {{'voice': result['voice'], 'lang_code': result['lang_code'], 'speed': result['speed'], 'seed': SEED}},\n"
                "    'espeak_fallback': result['espeak_fallback'],\n"
                "    'sanity_checks': checks,\n"
                "    'ceilings': ceilings,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'text': text, 'text_sha256': text_sha256}},\n"
                "    'seconds': round(elapsed, 3),\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'weight_format': WEIGHT_FORMAT,\n"
                "    'loader_weights_only': LOADER_WEIGHTS_ONLY,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'voices': len(pipe.voices)}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'kokoro': kokoro.__version__,\n"
                "        'misaki': misaki.__version__,\n"
                "        'soundfile': soundfile.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The WAV is a synthetic rendering of the input text in one of 54 named synthetic voices: it is not a recording "
        "of any person, carries no quality score, and its only in-repository checks are structural (rate, dtype, finite "
        "samples, peak within full scale) plus the phoneme string that shows what the G2P actually voiced. On the "
        "synthetic sample the audio is plumbing evidence only; the evaluation report is `not-measurable` because no "
        "metric can be computed without an external judge, and a real evaluation needs MOS ratings from listeners or an "
        "ASR round-trip WER over a reference sentence set, with the judge named. The seed controls the vocoder noise on "
        "one host and build; it does not promise identical audio across devices. Out-of-dictionary words depend on the "
        "espeak-ng fallback and are dropped when it is absent; segments above 510 phonemes are truncated by the library; "
        "non-English `lang_code`s and non-English quality are not exercised here; the pipeline exposes no cloning, "
        "mixing, timestamps or emotion control. The checkpoint and voice packs are pickles loaded through the "
        "weights-only loader after digest verification — an operator who bypasses `verify_snapshot` and loads an "
        "unverified file takes on arbitrary-code-execution risk.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot including all voice packs, validate the demonstrated "
        "request against the enforced ceilings, execute the public pipeline path, write a playable WAV, and emit the "
        "shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, naturalness or intelligibility on any audience, safety for high-consequence "
        "read-outs, or production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` (or the affected `voices/*.pt`) and rerun "
        "Section 3. A `ValueError` naming `MAX_TEXT_CHARS`, the speed range or the voice in Section 5: fix the form "
        "parameter or shorten the BYOD file and rerun from Section 4. `espeak_fallback: False` in Section 3 with words "
        "missing from the `phonemes` string in Section 6: the bundled espeak-ng library did not bind on this host — "
        "restrict the text to dictionary words or install `espeak-ng` system-wide and reload. A long pause in Section 3 "
        "on first use: `misaki` is downloading the spaCy `en_core_web_sm` model.\n\n"
        "**Next experiments.** Change `VOICE` to another `af_`/`am_` pack and compare the renderings of the same "
        "sentence; set `SPEED` to 0.8 and 1.5 and compare `duration_s`; run the same seed twice and diff the two WAV "
        "digests to see the seeded determinism on your host; upload a paragraph with names and numbers via `USE_BYOD` "
        "and inspect the `phonemes` string for dropped words; re-transcribe the WAV with the `whisper-asr-pipeline` "
        "sibling and compute WER against the input as the first step towards the intelligibility number the evaluation "
        "report asks for. None of these turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kokoro-tts-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kokoro-tts-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/kokoro-tts-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/hexgrad/kokoro (G2P: https://github.com/hexgrad/misaki)\n"
        "- StyleTTS 2 (Li et al., 2023): https://arxiv.org/abs/2306.07691\n"
        "- iSTFTNet (Kaneko et al., 2022): https://arxiv.org/abs/2203.02395"
    ),
}
