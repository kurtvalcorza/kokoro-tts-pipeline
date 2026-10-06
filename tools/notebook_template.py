"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, generator /2.1).

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
    "mode": "GUIDED",
    "isolated_runtime": True,
    # KTT-B1 / KTT-M1: the fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # generator /2.1 = gliner-ner-pipeline fe3d5ba tools/build_notebook.py byte for byte). kokoro 0.9.4 and misaki 0.9.4 declare
    # Requires-Python <3.13, so they cannot be installed into Colab's Python 3.13 kernel; the managed CPython 3.12.12 environment
    # does not depend on the kernel's Python. The lock is compiled with
    # `uv pip compile pyproject.toml tutorials/requirements-colab-extra.in -c <versions of ast-audio-classification-pipeline's
    # T4-passed tutorials/requirements-colab.lock.txt @ 16eee39> --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`: ast's 47 packages at the same versions plus
    # kokoro/misaki and their spaCy/phonemizer stack (65 more). Two consequences of the lock, stated to the learner:
    # (1) `--only-binary :all:` resolves num2words to 0.5.6, the newest release whose dependencies are all wheels (0.5.8-0.5.14
    # require docopt, which is published only as an sdist); (2) the spaCy model en_core_web_sm 3.8.0, which misaki would otherwise
    # pip-install at run time (the uv environment has no pip), is locked by URL and SHA-256 from the extra input file.
    # No system package is needed: the espeakng-loader wheel bundles libespeak-ng and its data.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated Python 3.12.12 environment from the "
        "hash-locked pins (the kernel's own packages and its Python version are left alone, so no restart is needed and a "
        "Python 3.13 kernel such as Colab's works), stages and digest-verifies the pinned 58-file snapshot, authors the "
        "one-sentence synthetic sample in code, validates it into an input manifest before the model runs, synthesises one "
        "24 kHz WAV with a fixed seed, writes the evaluation report, and exports machine-readable outputs with provenance. "
        "The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.2 §5). Building the isolated environment takes a few minutes on top of the "
        "seconds the synthesis itself needs."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose **Runtime → "
        "Run after** (it re-runs Section 4 and every later cell). Supply one UTF-8 `.txt` file of at most `MAX_TEXT_CHARS` "
        "(2,000) characters through the upload dialog on Colab, or by path in `BYOD_PATH` on any runtime (a path, when set, "
        "is used instead of the dialog). It passes through the same notebook-local validation, synthesis, evaluation-report "
        "and export cells as the sample. Long lines are split by the library into several chunks, and a line that produces "
        "no audio is named in a warning in Section 6. The expected input and the privacy guidance are stated in the "
        "Prerequisites and in Section 4, and the file stays inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
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
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. The pinned `kokoro` 0.9.4 and `misaki` 0.9.4 support only Python 3.10–3.12, and Colab's kernel runs Python 3.13, so Section 1 does not install them into the kernel: it builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, and the kernel's own Python version does not matter. A Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline does not change precision by device). The model card's CPU smoke loaded and verified the 58-file snapshot in 6.22 s and rendered 3.25 s of audio in 0.72 s, so the one-sentence default runs in seconds once the environment exists. The locked install (PyTorch 2.14.0 with its CUDA libraries, a few GB) and the 355 MB snapshot (327 MB checkpoint plus 54 voice packs) are the largest downloads of the run.",
        "- **Expected warnings:** the first import of `num2words` 0.5.6 (the newest release the wheel-only lock can use; `misaki` uses it to spell out numbers) may print `SyntaxWarning: invalid escape sequence` lines from its Portuguese module; they do not affect English. Loading the model may print PyTorch warnings about LSTM `dropout` with one layer and the deprecated `weight_norm`; they come from the `kokoro` library and do not stop the run.",
        "- **Knowledge:** basic Python; what a phoneme string is; why a synthesised waveform has no ground truth to score against.",
        "- **Data:** the default sample is one synthetic English sentence authored in code, so nothing is downloaded and no private data is needed. Optional BYOD is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 `.txt` file of at most 2,000 characters, supplied with the Colab upload dialog or by path in `BYOD_PATH`; the library splits it on newlines, splits a line above 510 phonemes into several chunks, and skips a line that phonemises to nothing (Section 6 names any skipped line). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API. The text you submit will be spoken verbatim.",
        "- **spaCy model:** the `misaki` G2P library needs the spaCy English model `en_core_web_sm` 3.8.0. It is part of the hash-locked environment, fetched once in Section 1 from its GitHub release (`github.com/explosion/spacy-models`, SHA-256 checked), so `misaki` does not download it while the notebook runs.",
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
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 `.txt` file of at most "
                "`MAX_TEXT_CHARS` characters. On Colab, leave `BYOD_PATH` empty and an upload dialog opens; on any "
                "runtime (Kaggle, Linux Jupyter, or Colab with a file already in `/content`), put the file's path in "
                "`BYOD_PATH` instead. The library splits the text on newlines and synthesises each line: a line whose "
                "phonemes exceed 510 is split into several chunks (at punctuation or word boundaries) that are "
                "concatenated, and a line that phonemises to nothing produces no audio and is named in a warning in "
                "Section 6. The file stays inside this runtime. Numbers are spelled out in words and all-capital "
                "acronyms letter by letter; other words outside the dictionary are pronounced by the espeak-ng fallback "
                "when it is available on the host and are otherwise dropped, and a line of symbols only (such as `***`) "
                "always phonemises to nothing — Section 6 shows how to check. If the cell stops, its message says what to change: "
                "no file uploaded, a file that is not UTF-8, or a `BYOD_PATH` that is not a file."
            ),
            "code": (
                "import hashlib\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "VOICE = 'af_heart'  # @param {{type:\"string\"}}\n"
                "SPEED = 1.0  # @param {{type:\"number\"}}\n"
                "SEED = 0  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    print({{'expected_byod_input': 'one UTF-8 .txt file', 'max_chars': MAX_TEXT_CHARS, 'source': 'BYOD_PATH' if BYOD_PATH else 'Colab upload dialog'}})\n"
                "    if BYOD_PATH:\n"
                "        byod_file = Path(BYOD_PATH)\n"
                "        if not byod_file.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file in this runtime: give the path of one UTF-8 .txt file and run this cell again')\n"
                "        sample_name, payload = byod_file.name, byod_file.read_bytes()\n"
                "        sample_kind = 'BYOD file (BYOD_PATH)'\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD = True needs a file: this runtime has no Colab upload dialog, so set BYOD_PATH to the path of one UTF-8 .txt file and run this cell again') from None\n"
                "        uploaded = files.upload()\n"
                "        if not uploaded:\n"
                "            raise ValueError('no file uploaded; rerun this cell and choose one UTF-8 .txt file')\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'{{len(uploaded)}} files uploaded; rerun this cell and choose exactly one UTF-8 .txt file')\n"
                "        sample_name, payload = next(iter(uploaded.items()))\n"
                "        sample_kind = 'BYOD upload'\n"
                "    text = decode_byod_text(payload, sample_name)\n"
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
                "finding. `SAMPLE_RATE` is the fixed 24 kHz output rate. The manifest's `segments` count is the number of "
                "non-blank lines. The notebook never trims or alters the text; inside the library a line above 510 "
                "phonemes is split into several chunks, which is only visible afterwards through the returned `segments` "
                "and their `line` indices."
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
                "Hz), `num_samples`, `duration_s`, `peak_amplitude`, `segments` (one `graphemes`/`phonemes` chunk per "
                "piece of audio, each with the `line` it came from: a short line gives one chunk, a line above 510 "
                "phonemes several), `lines` (the number of non-blank lines) and `skipped_lines` (lines that phonemised to "
                "nothing and produced no audio), the `voice`, `lang_code` and `speed` used, `espeak_fallback`, the device, "
                "and the model identity. **The output is not bit-deterministic by default:** the vocoder adds Gaussian noise "
                "and a random initial phase to its harmonic excitation, so two unseeded calls differ at the sample level "
                "(the model card measured a maximum absolute difference of 0.093 between consecutive smoke calls of "
                "identical length) while durations and phonemes stay the same; the pipeline sets no seed, so this cell "
                "calls `torch.manual_seed(SEED)` immediately before synthesis — the card recorded bit-identical waveforms "
                "across two seeded calls on the same host. Seeding does not make the audio identical across devices, "
                "PyTorch builds or CPU kernels. The sanity checks below (right dtype, rate and shape, finite samples, "
                "peak within full scale, every non-blank line either voiced or reported as skipped) are falsifiable "
                "plumbing checks, not a quality result; a skipped line is printed as a warning naming the line, not "
                "raised as an error, because it is a property of your text rather than of the pipeline, and "
                "the model card's smoke observation (78,000 samples, 3.25 s, peak 0.342 on the same sentence, unseeded "
                "CPU) is quoted as one measurement on that host, not an expected value. The `phonemes` string is the only "
                "in-repository way to see whether the G2P dropped a word inside a line — with `espeak_fallback` `False`, "
                "out-of-dictionary words vanish silently; a whole line that vanishes is reported in `skipped_lines`. The "
                "WAV written to `outputs/` is 16-bit PCM at 24 kHz via the pinned `soundfile`, and an inline HTML audio "
                "player for it appears below the printed facts in a notebook front end."
            ),
            "code": (
                "import base64\n"
                "import builtins\n"
                "import time\n\n"
                "torch.manual_seed(SEED)\n"
                "started = time.perf_counter()\n"
                "result = pipe.synthesize(text, voice=VOICE, speed=SPEED)\n"
                "elapsed = time.perf_counter() - started\n"
                "audio = result['audio']\n"
                "lines_with_text = [i for i, part in enumerate(text_lines(text)) if part.strip()]\n"
                "voiced_lines = sorted({{segment['line'] for segment in result['segments']}})\n"
                "skipped_lines = result['skipped_lines'] or []\n"
                "checks = {{\n"
                "    'audio_is_float32_1d': isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1,\n"
                "    'sample_rate_matches_contract': result['sample_rate'] == SAMPLE_RATE,\n"
                "    'all_samples_finite': bool(np.isfinite(audio).all()),\n"
                "    'peak_within_full_scale': 0.0 < result['peak_amplitude'] <= 1.0,\n"
                "    'every_line_voiced_or_reported': result['skipped_lines'] is not None and sorted(voiced_lines + [s['line'] for s in skipped_lines]) == lines_with_text,\n"
                "    'duration_consistent': abs(result['duration_s'] - result['num_samples'] / SAMPLE_RATE) < 1e-6,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'synthesize output failed a sanity check: {{checks}}')\n"
                "wav_path = 'outputs/{stem}_sample.wav'\n"
                "soundfile.write(wav_path, audio, result['sample_rate'], subtype='PCM_16')\n"
                "wav_sha256 = hashlib.sha256(Path(wav_path).read_bytes()).hexdigest()\n"
                "print({{key: value for key, value in result.items() if key not in ('audio', 'segments')}})\n"
                "print({{'seconds': round(elapsed, 3), 'audio_seconds_per_wall_second': round(result['duration_s'] / elapsed, 2), 'rms': round(float(np.sqrt(np.mean(np.square(audio)))), 4), 'checks': checks}})\n"
                "print({{'lines': result['lines'], 'chunks': len(result['segments']), 'chunks_per_line': {{line: sum(1 for s in result['segments'] if s['line'] == line) for line in voiced_lines}}}})\n"
                "for index, segment in enumerate(result['segments']):\n"
                "    print(f\"chunk {{index}} (line {{segment['line']}}): graphemes={{segment['graphemes'][:80]!r}}\")\n"
                "    print(f\"chunk {{index}} (line {{segment['line']}}): phonemes ={{segment['phonemes'][:80]!r}}\")\n"
                "if skipped_lines:\n"
                "    print(f'WARNING: {{len(skipped_lines)}} line(s) produced no audio because they phonemised to nothing (symbols only, or only words outside the dictionary with espeak_fallback={{result[\"espeak_fallback\"]}}); reword them with dictionary words: {{skipped_lines}}')\n"
                "print({{'wav': wav_path, 'wav_sha256': wav_sha256, 'subtype': 'PCM_16'}})\n\n\n"
                "class WavPlayer:\n"
                "    \"\"\"An inline <audio> player for the WAV; it renders as HTML, so it also works from the isolated environment.\"\"\"\n\n"
                "    def __init__(self, path):\n"
                "        self.src = 'data:audio/wav;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')\n\n"
                "    def _repr_html_(self):\n"
                "        return f'<audio controls src=\"{{self.src}}\"></audio>'\n\n\n"
                "show = globals().get('display') or getattr(builtins, 'display', None)\n"
                "if callable(show):\n"
                "    show(WavPlayer(wav_path))\n"
                "else:\n"
                "    print('no notebook front end to show an audio player; open the WAV file instead')"
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
                "wall time), the per-chunk graphemes, phonemes and line indices, the line coverage (non-blank lines, "
                "chunks, skipped lines), the request (`voice`, `lang_code`, `speed`, `seed`), "
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
                "    'line_coverage': {{'lines': result['lines'], 'chunks': len(result['segments']), 'skipped_lines': result['skipped_lines']}},\n"
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
        "espeak-ng fallback and are dropped when it is absent, and a line made only of such words (or only of symbols) produces no audio and "
        "is named in Section 6's warning; a line above 510 phonemes is split by the library into several chunks; "
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
        "**Troubleshooting.** `This notebook needs a Linux x86_64 runtime` in Section 1: open it in Google Colab, Kaggle "
        "or Linux Jupyter; the locked environment is built from manylinux wheels. `The pinned uv wheel failed its "
        "size/SHA-256 check` in Section 1: the download was cut short or altered — run the cell again. A pip/uv error "
        "naming a hash in Section 1: a package download did not match the lock — run the cell again; if it repeats, "
        "the network is altering downloads. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` (or the affected `voices/*.pt`) and rerun "
        "Section 3. A `ValueError` naming `MAX_TEXT_CHARS`, the speed range or the voice in Section 5: fix the form "
        "parameter or shorten the BYOD file, then select Section 4 and choose **Runtime → Run after**. In Section 4 with "
        "`USE_BYOD = True`: `no file uploaded` — the upload dialog was cancelled, run the cell again and choose one file; "
        "`is not UTF-8 text` — save the file as UTF-8 and supply it again; `needs a file: this runtime has no Colab "
        "upload dialog` or `BYOD_PATH ... is not a file` — put the path of your `.txt` file in `BYOD_PATH`. "
        "`espeak_fallback: False` in Section 3 with words missing from the `phonemes` string in Section 6: the bundled "
        "espeak-ng library did not bind on this host — restrict the text to dictionary words. `WARNING: ... line(s) "
        "produced no audio` in Section 6: those lines held only symbols or only words the G2P could not pronounce — reword them; "
        "`RuntimeError: no audio produced` means every line was like that.\n\n"
        "**Next experiments.** Change `VOICE` to another `af_`/`am_` pack and compare the renderings of the same "
        "sentence; set `SPEED` to 0.8 and 1.5 and compare `duration_s`; run the same seed twice and diff the two WAV "
        "digests to see the seeded determinism on your host; supply a paragraph with names and numbers via `USE_BYOD` "
        "and inspect the `phonemes` string for dropped words and the `chunks_per_line` count for long lines; re-transcribe the WAV with the `whisper-asr-pipeline` "
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
