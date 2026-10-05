"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.

2026-10-05 fleet-sweep fix (SWP-R, SWP-G, SWP-B): generator /2.2 keys ``isolated_runtime`` (nothing is installed into
the kernel; a hash-locked uv environment runs every later cell, so Run all needs no restart), ``infrastructure_labels``
and ``guided`` (audience, how-to-use, roadmap, task contract; predictions and worked checkpoints from the recorded
2026-09-13 run, troubleshooting, glossary and a conclusion template). BYOD takes a path or one uploaded text file.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "kokoro_tts_pipeline",
    "repo_name": "kokoro-tts-pipeline",
    "stem": "kokoro_tts",
    "notebook_name": "kokoro_tts_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline, siglip-v1-zero-shot-pipeline): a
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins plus
    # tutorials/requirements-colab-extra.in with `uv pip compile pyproject.toml tutorials/requirements-colab-extra.in
    # --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all:
    # -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and "
        "digest-verifies the pinned snapshot, authors the tutorial sample automatically, validates it into an input manifest "
        "before the model runs, synthesises the speech locally, writes the evaluation report, and exports machine-readable "
        "outputs with provenance. The default path needs no repository clone, no DIMER worker or service, no credential, no "
        "upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 — with `BYOD_PATH` set to one UTF-8 text "
        "file (Kaggle or Jupyter), or left empty for the Colab upload dialog — and re-run from that cell. Your text passes "
        "through the same validation, synthesis, evaluation-report and export cells as the sample; the expected input "
        "format, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and the text stays "
        "inside this runtime. BYOD is optional and never part of the default path."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has run a Colab or Jupyter notebook, and wants to see how a small open text-to-speech model turns written English into audio, what the pipeline checks before and after synthesis, and why the run ends without an accuracy number. No speech-processing background is assumed: *phoneme*, *G2P*, *voice pack*, *vocoder* and the other terms are explained where they first matter and again in the **Glossary** at the end. A CPU runtime is enough; a T4 GPU is used automatically when present.\n\n"
                "**Input → Model → Output.**\n\n"
                "| | What it is in this notebook |\n|---|---|\n"
                "| Input | one English text (default: a 44-character pangram written in Section 4; BYOD: one UTF-8 text file of at most 2,000 characters), a voice name, a speed and a seed |\n"
                "| Model | `misaki` grapheme-to-phoneme (G2P) → phoneme string; a 128-d style vector read from the chosen voice pack; the StyleTTS 2 decoder predicts per-phoneme duration, pitch and energy; the ISTFTNet vocoder renders the waveform |\n"
                "| Output | one 24 kHz mono float32 waveform, written as a 16-bit WAV; the phoneme string of each segment; run-level facts (duration, peak amplitude); an evaluation report whose verdict is always `not-measurable` |\n\n"
                "**How to use this notebook.** Choose a runtime (CPU is enough; **Runtime → Change runtime type → T4 GPU** is used automatically when selected), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried pipeline module and the pinned model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the Colab T4 run of 13 September 2026, retained under `docs/verification/2026-09-13/` in the repository; a CPU run, another voice or another seed changes the sample-level numbers). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n"
                "**Roadmap:** 1–3 infrastructure → 4 author the text (or bring your own) → 5 validate the request into an input manifest *(core concept: ceilings and the one-language-per-instance voice rule)* → 6 synthesise, write the WAV and read the output *(core concept: phonemes, seeding and what the sanity checks prove)* → 7 the evaluation report and why no metric is reported *(evaluation practice)* → 8 export with provenance *(engineering)* → interpretation and limits, troubleshooting, glossary, conclusion."
            )
        ]
    },
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline does not change precision by device). The model card's CPU smoke loaded and verified the 58-file snapshot in 6.22 s and rendered 3.25 s of audio in 0.72 s, so the one-sentence default runs in seconds on a hosted CPU runtime. Section 1 builds a separate environment from the hash-locked pins (nothing is installed into the notebook's own Python, so no restart is needed); its PyTorch wheels and the 355 MB snapshot (327 MB checkpoint plus 54 voice packs) are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a phoneme string is; why a synthesised waveform has no ground truth to score against.",
        "- **Data:** the default sample is one synthetic English sentence authored in code, so nothing is downloaded and no private data is needed. Optional BYOD is gated off by default so the sample path can run top-to-bottom without interaction; it reads one file from `BYOD_PATH` (Kaggle, Jupyter) or, when that is empty on Colab, from the upload dialog. Expected BYOD input: one UTF-8 text file of at most 2,000 characters; the pipeline splits it on newlines and synthesises each line as a segment. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API. The text you submit will be spoken verbatim.",
        "- **Dependency network call:** outside this package's control, the `misaki` G2P library downloads the spaCy `en_core_web_sm` model once on first English use (spaCy installs it with the `pip` carried in the isolated environment's lock).",
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
                "dropped — Section 6 shows how to check. Set `BYOD_PATH` to the file's path on Kaggle or Jupyter; leave it empty "
                "on Colab to get the upload dialog. A cancelled or empty upload, a file that is not UTF-8, an empty file or one "
                "above `MAX_TEXT_CHARS` is refused with a message naming the file and the rule.\n\n"
                "**Predict:** how many characters and lines will the cell report for the default text, and will Section 5 accept it?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "The recorded run printed 44 characters and 1 line (`The quick brown fox jumps over the lazy dog.` — 35 letters, 8 spaces and a full "
                "stop), and Section 5 accepted it: 44 is far below the 2,000-character ceiling, and one line means one segment.\n\n"
                "</details>"
            ),
            "code": (
                "import hashlib\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "# Kaggle / Jupyter: the path of one UTF-8 text file (for example under /kaggle/input/). Empty: the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "VOICE = 'af_heart'  # @param {{type:\"string\"}}\n"
                "SPEED = 1.0  # @param {{type:\"number\"}}\n"
                "SEED = 0  # @param {{type:\"integer\"}}\n\n\n"
                "def read_byod_text(path_text):\n"
                "    \"\"\"One UTF-8 text file from BYOD_PATH, or from exactly one Colab upload; each refusal names the file and the rule.\"\"\"\n"
                "    if path_text.strip():\n"
                "        path = Path(path_text.strip()).expanduser()\n"
                "        if not path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{str(path)!r}} is not a file: give the path of one UTF-8 text file of at most {{MAX_TEXT_CHARS}} characters')\n"
                "        name, data = path.name, path.read_bytes()\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD = True but BYOD_PATH is empty and this runtime has no Colab upload dialog: set BYOD_PATH to one UTF-8 text file (on Kaggle, a file under /kaggle/input/)') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError(f'expected exactly one uploaded text file, got {{len(uploaded)}} ({{sorted(uploaded) or \"upload cancelled or empty\"}}): run this cell again and choose one file, or set BYOD_PATH')\n"
                "        name, data = next(iter(uploaded.items()))\n"
                "    try:\n"
                "        body = data.decode('utf-8-sig').strip()\n"
                "    except UnicodeDecodeError as exc:\n"
                "        raise ValueError(f'{{name}}: not UTF-8 text (undecodable byte at offset {{exc.start}}); save the file as UTF-8 and try again') from None\n"
                "    if not body:\n"
                "        raise ValueError(f'{{name}}: the file holds no text after stripping whitespace; the pipeline needs at least one character')\n"
                "    if len(body) > MAX_TEXT_CHARS:\n"
                "        raise ValueError(f'{{name}}: {{len(body)}} characters exceeds MAX_TEXT_CHARS = {{MAX_TEXT_CHARS}}; shorten the file')\n"
                "    return name, body\n\n\n"
                "if USE_BYOD:\n"
                "    sample_name, text = read_byod_text(BYOD_PATH)\n"
                "    sample_kind = 'BYOD file' if BYOD_PATH.strip() else 'BYOD upload'\n"
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
                "the returned `phonemes`.\n\n"
                "**Predict:** the cell also asks the validator about the British voice `bf_emma`. Will it be accepted? If not, "
                "what will the recorded message say?\n\n"
                "**What to notice:** the verdict for your own request, `voice_inventory` (how many packs passed digest "
                "verification), and the one entry under `findings`.\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "Rejected. The recorded run's manifest has verdict `accepted` for the pangram (44 characters, 1 segment, "
                "voice `af_heart`, speed 1.0, `lang_code` `a`, a 54-pack inventory) and one finding: `voice 'bf_emma' is not "
                "a lang_code='a' voice`. The first letter of a voice name is its language code; this instance loaded `a` "
                "(American English), so a `b` (British) voice is refused instead of being mixed in silently.\n\n"
                "</details>"
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
                "pinned `soundfile`; the inline player below appears in any notebook front end (Colab, Kaggle, Jupyter).\n\n"
                "**Predict:** at speed 1.0, roughly how many seconds of audio will the 44-character pangram produce, and how "
                "many samples is that at 24,000 Hz?\n\n"
                "**What to notice:** `num_samples` ÷ 24,000 = `duration_s`; all six `checks` true; the `phonemes` line — "
                "count whether every one of the nine words appears.\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "The recorded T4 run produced 78,000 samples, which is exactly 3.25 s at 24 kHz, with peak amplitude 0.337 and "
                "all six sanity checks true; `espeak_fallback` was `True`. The phoneme string was "
                "`ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.` — nine voiced words, none dropped. Duration and "
                "phonemes are properties of the text and speed; the peak (0.342 in the model card's unseeded CPU smoke) "
                "depends on the vocoder noise, the seed and the device, so your last digits may differ. None of these "
                "numbers says the speech sounds natural — listen to it.\n\n"
                "</details>"
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
                "\n\n"
                "class WavPlayer:\n"
                "    \"\"\"An HTML5 audio element carrying the written WAV; renders in Colab, Kaggle and Jupyter.\"\"\"\n\n"
                "    def __init__(self, path):\n"
                "        import base64\n\n"
                "        self.path = path\n"
                "        self.data = base64.b64encode(Path(path).read_bytes()).decode('ascii')\n\n"
                "    def _repr_html_(self):\n"
                "        return f'<audio controls src=\"data:audio/wav;base64,{{self.data}}\"></audio>'\n\n"
                "    def __repr__(self):\n"
                "        return f'<WAV player: {{self.path}}>'\n\n\n"
                "try:\n"
                "    display(WavPlayer(wav_path))\n"
                "except NameError:\n"
                "    print('inline player unavailable outside a notebook front end; open the WAV file instead')"
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
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict:** will the report contain any number you could call a quality score?\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "No. The recorded report has `metrics: []`, `baselines: []` and verdict `not-measurable`; it carries "
                "`n_segments` 1 and `duration_s` 3.25 as run-level facts, and its `needs` field names what would make the "
                "task measurable — MOS ratings from listeners, or an ASR round-trip word error rate over a reference "
                "sentence set, with the judge named.\n\n"
                "</details>"
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
                "reach the export.\n\n"
                "<details><summary>Check your reasoning</summary>\n\n"
                "Which field ties the audio to this record? `wav.sha256`. The recorded run's WAV digest was "
                "`daf4a8cf38a11e3c…`; a run on another device or PyTorch build gives a different waveform and therefore a "
                "different digest, which is why the record also carries the seed, the device and the runtime versions. "
                "`outputs/` should list four files: the input manifest, the evaluation report, the result JSON and the WAV.\n\n"
                "</details>"
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
        "## Troubleshooting\n\n"
        "Section 1 stops with `This notebook needs a Linux x86_64 runtime`: the locked environment holds manylinux x86_64 "
        "wheels — use Google Colab, Kaggle or a Linux Jupyter host. `The pinned uv wheel failed its size/SHA-256 check`: "
        "run Section 1 again; if it repeats, the download is being altered on the way. `The isolated environment's Python "
        "process exited`: the worker crashed, usually out of memory — restart the session and choose **Run all**. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` (or the affected `voices/*.pt`) and rerun "
        "Section 3. A `ValueError` naming `MAX_TEXT_CHARS`, the speed range or the voice in Section 5: fix the form "
        "parameter or shorten the BYOD file and rerun from Section 4. `espeak_fallback: False` in Section 3 with words "
        "missing from the `phonemes` string in Section 6: the bundled espeak-ng library did not bind on this host — "
        "restrict the text to dictionary words or install `espeak-ng` system-wide and reload. A long pause in Section 3 "
        "on first use: `misaki` is downloading the spaCy `en_core_web_sm` model. With `USE_BYOD = True`: "
        "`BYOD_PATH … is not a file` — fix the path; `expected exactly one uploaded text file` — the upload was cancelled, "
        "empty or held several files; `not UTF-8 text` — re-save the file as UTF-8. No player under Section 6: open "
        "`outputs/{stem}_sample.wav` from the file browser.\n\n"
        "## Change one thing (next experiments)\n\n"
        "Change `VOICE` to another `af_`/`am_` pack and compare the renderings of the same "
        "sentence; set `SPEED` to 0.8 and 1.5 and compare `duration_s`; run the same seed twice and diff the two WAV "
        "digests to see the seeded determinism on your host; upload a paragraph with names and numbers via `USE_BYOD` "
        "and inspect the `phonemes` string for dropped words; re-transcribe the WAV with the `whisper-asr-pipeline` "
        "sibling and compute WER against the input as the first step towards the intelligibility number the evaluation "
        "report asks for. None of these turns the sample result into evidence of production fitness.\n\n"
        "## Glossary\n\n"
        "- **Phoneme:** the smallest sound unit of a language; the model speaks phonemes, not letters.\n"
        "- **G2P (grapheme-to-phoneme):** turning written text into a phoneme string; here the `misaki` library, with a dictionary and spaCy part-of-speech tags.\n"
        "- **espeak-ng fallback:** a rule-based G2P used for words missing from `misaki`'s dictionary; `espeak_fallback` reports whether it bound on this host.\n"
        "- **Voice pack:** a pre-computed table of 128-d style vectors for one synthetic voice (`af_heart` = American English, female); the first letter is the language code.\n"
        "- **StyleTTS 2 decoder:** the network that predicts how long, how high and how loud each phoneme is.\n"
        "- **Vocoder (ISTFTNet):** the network that turns those predictions into a waveform; it adds random noise, which is why the seed matters.\n"
        "- **Sample rate:** samples per second of audio — 24,000 here, so 78,000 samples last 3.25 s.\n"
        "- **Peak amplitude:** the largest absolute sample value; 1.0 is digital full scale, above it the WAV would clip.\n"
        "- **PCM_16:** 16-bit integer WAV encoding used for the written file.\n"
        "- **MOS / WER:** Mean Opinion Score (listeners rate naturalness) and word error rate (an ASR re-transcription compared with the input) — the external judges a real evaluation needs.\n"
        "- **Weights-only loader:** PyTorch's restricted unpickler that only rebuilds tensors and primitive containers; used here after digest verification.\n"
        "- **Input manifest:** the JSON record of what was validated, against which ceilings, and with what verdict, written before the model runs.\n"
        "- **Isolated environment:** the separate hash-locked Python environment built in Section 1; every later cell runs there, so the kernel's own packages are never replaced.\n\n"
        "## Conclusion (your notes)\n\n"
        "1. In two sentences: what does the pipeline guarantee about the WAV it wrote, and what does it not guarantee?\n"
        "2. Which of your predictions were wrong, and what did the output show instead?\n"
        "3. One change you tried (voice, speed, seed or your own text) and what it changed in `duration_s`, the phonemes or the digest.\n"
        "4. What you would need before claiming the speech is intelligible to a given audience.\n\n"
        "**Your notes:**\n\n"
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
