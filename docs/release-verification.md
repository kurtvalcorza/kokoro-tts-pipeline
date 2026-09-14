# Release verification

`tutorials/kokoro_tts_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) remains a **release candidate**. A clean Python 3.12 GPU execution of the exact notebook blob was recorded on 2026-09-13; the result and retained artifacts are below. Static checks are not runtime evidence, and promotion still requires a reviewer to accept the recorded run.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/kokoro_tts_pipeline/pipeline.py` after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 58-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`, `list_voices`,
  `KokoroTTSPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `torch.manual_seed(SEED)` before
  `synthesize`, `evaluation_report`), the ceiling print, the contract sanity checks (sample rate, peak within full
  scale), the WAV write through `soundfile`, the `espeak_fallback` export, the four exports, the learner-facing
  statements (no adaptation, no intrinsic metric, verdict always `not-measurable`, non-deterministic vocoder,
  pickle trust boundary, 510-phoneme segment cap, no cloning, no speech-to-text) and the gated-off
  BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` /
  repository import on the primary path, a mutable `revision='main'`, `from kokoro import` / `from misaki import` /
  `KModel(` / `KPipeline(` / `huggingface_hub` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no
weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache, no pre-staged files under `weights/kokoro-82m/` beyond the committed ones | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with **no repository checkout**, an empty Hugging Face
   cache, no pre-staged `kokoro-v1_0.pth` or `voices/*.pt` under `weights/kokoro-82m/`, and no
   pre-installed spaCy `en_core_web_sm` (so the `misaki` first-use download is observed and recorded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `VOICE = 'af_heart'`, `SPEED = 1.0`, `SEED = 0`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= the
   `pyproject.toml` pins (`kokoro==0.9.4`, `misaki==0.9.4`,
   `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `numpy==2.5.3`, `huggingface-hub==0.36.2`, `soundfile==0.14.0`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executing (defining `KokoroTTSPipeline`, `validate_inputs`, `evaluation_report`,
     `list_voices` and the ceilings) with no import of the repository package;
   - the synthetic pangram authored in code with its text SHA-256 printed;
   - the inline `MANIFEST` asserted against the module identity and written to `weights/kokoro-82m/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reporting all 58 manifest entries fetched in an empty standalone weights directory from `hexgrad/Kokoro-82M`
     at the immutable revision (the checkpoint, 54 voice packs and three metadata files), `verify_snapshot` returning the
     58-entry manifest, `list_voices` returning 54 names including `af_heart`, and
     `KokoroTTSPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with
     `lang_code 'a'`, `source 'local-snapshot'` and the observed `espeak_fallback` value recorded;
   - ceilings `MAX_TEXT_CHARS = 2000`, `MIN_SPEED = 0.5`, `MAX_SPEED = 2.0`, `SAMPLE_RATE = 24000`,
     the nine `LANG_CODES` and `DEFAULT_LANG_CODE = 'a'` printed, and `validate_inputs` writing
     `outputs/kokoro_tts_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     `bf_emma` wrong-language probe) before model execution;
   - `synthesize` returning a 1-D float32 array at 24000 Hz with all six sanity checks true, one
     segment whose `phonemes` string voices every word of the pangram, and `outputs/kokoro_tts_sample.wav`
     written as 16-bit PCM;
   - `evaluation_report` writing `outputs/kokoro_tts_evaluation_report.json` with verdict `not-measurable` and an
     empty `metrics` list, and the "No metric is reported" line printed;
   - `outputs/kokoro_tts_result.json` written with the WAV path and digest, the audio facts, the
     segments, the request including the seed, `espeak_fallback`, `NOTEBOOK_SOURCE`, model identifier,
     immutable model revision, model licence, weight format and loader facts, snapshot summary, runtime versions,
     device and dtype;
6. verify the exports exist, that the WAV is audible speech of the pangram (a listener's observation,
   not a metric), and that the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, NumPy, `kokoro`,
   `misaki`, `soundfile`, device), model identifier and immutable revision, whether the model cache
   and weights directory were clean, whether the spaCy model download occurred, outcome, produced
   outputs, `num_samples`/`duration_s`/`peak_amplitude` and the phoneme string (as observations, not
   a metric), and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/kokoro_tts_colab.ipynb` | `bf509157c7133a863aa891bf7b88087b1464b946` / `ca60f5f4f3b6c3dbf48f44a0aeaeedab4c7880ad` | 2026-09-13 | Colab CLI → isolated Python 3.12.3, T4 | PASS — 8/8 cells; listening and evidence review pending; [Retained run](verification/2026-09-13/README.md) |

## Recorded executions

Notebook identity is the Git blob of `tutorials/kokoro_tts_colab.ipynb` at the source commit in the row below. The documentation commit recording the run does not change that notebook blob. Cell wall time is the sum of recorded code-cell times, including installation and model downloads; total time additionally includes environment setup and bookkeeping. These measurements describe this one run.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Cell wall / total | Outcome |
|---|---|---|---|---|---|
| 2026-09-13 | `bf509157c7133a863aa891bf7b88087b1464b946` / `ca60f5f4f3b6c3dbf48f44a0aeaeedab4c7880ad` | Colab CLI → fresh Python 3.12.3 venv/interpreter; Tesla T4, 15,360 MiB | Unchanged default sample, no repository checkout, empty per-model cache and weights | 184.518 s / 188.984 s | PASS — 8/8 cells; listening and evidence review pending; [Retained run](verification/2026-09-13/README.md) |

The run used PyTorch `2.14.0+cu130`, `cuda:0` and `float32`. All eight code cells completed, runtime pins matched, every snapshot file was SHA-256 verified, inputs were accepted, and the negative validation probe was recorded. Results, model identity/revision, observed output, warnings, package versions, notebook outputs, executor source and cleanup evidence are retained in [the run record](verification/2026-09-13/README.md).

The native hosted kernel was Python 3.13.15; its direct notebook attempt was aborted in installation after the Python-version mismatch was confirmed. The successful result above uses the repository-supported Python 3.12 interpreter on the Colab GPU. No completed native hosted-kernel run is claimed.

## Current status

Clean GPU execution evidence is now recorded for the exact notebook blob above. The registry status remains **Candidate** pending a reviewer’s acceptance of the evidence and an integrator’s promotion. This documentation change performs no promotion. The run is default-sample inference/contract evidence; it does not establish model quality or a benchmark result. Kokoro listening review is still pending; WAV structure and digest were checked, but audible intelligibility and perceptual quality were not assessed.

Current source update: snapshot validation now runs before model-library imports (Kokoro also validates the language first), so rejected requests fail with the intended validation error even when model libraries are absent. The standalone notebook was regenerated from this source. The retained 2026-09-13 GPU run identifies the earlier notebook blob; the regenerated notebook has not had a fresh GPU execution. Status remains **Candidate**.
