# Release verification

`tutorials/kokoro_tts_colab.ipynb` (`TASK-INFERENCE`, `GUIDED`, **standalone** carrier, DIMER Notebook Specification
2.2) is a **release candidate** until the exact notebook revision has executed top-to-bottom with **Run all**, in one
pass and with no manual restart, in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, the
generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence.
No `Run all` of the current notebook is recorded (see below).

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`,
  a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/kokoro_tts_pipeline/pipeline.py` after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 58-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; `NOTEBOOK_SOURCE` recorded in exports;
- the isolated runtime (review KTT-B1, KTT-M1): exactly two kernel cells — the install cell, which verifies the pinned
  `uv` wheel by size and SHA-256, creates a managed CPython 3.12.12 environment and installs the carried hash-locked
  `tutorials/requirements-colab.lock.txt` (including the spaCy `en_core_web_sm` 3.8.0 wheel) with
  `--require-hashes --only-binary :all:` on Linux x86_64 only, and the router that sends every later cell to one
  persistent worker in that environment; the worker's `google.colab` stubs carry a module spec; learner-facing text
  removed by the review fixes (the in-kernel install and its restart, the "truncates" statement, the runtime spaCy
  download) may not return;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`, `list_voices`,
  `KokoroTTSPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `torch.manual_seed(SEED)` before
  `synthesize`, `evaluation_report`), the ceiling print, the contract sanity checks (sample rate, peak within full
  scale, every non-blank line voiced or reported as skipped), the skipped-line warning, the WAV write through
  `soundfile`, the `espeak_fallback` and line-coverage exports, the four exports, the BYOD fields (`USE_BYOD`,
  `BYOD_PATH`) with named errors for a cancelled upload and non-UTF-8 text, the learner-facing statements (no
  adaptation, no intrinsic metric, verdict always `not-measurable`, non-deterministic vocoder, pickle trust boundary,
  long lines split into chunks, lines that phonemise to nothing skipped, no cloning, no speech-to-text) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` other than the locked
  spaCy model wheel / repository import on the primary path, a mutable `revision='main'`, `from kokoro import` /
  `from misaki import` / `KModel(` / `KPipeline(` / `huggingface_hub` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token, no document makes an
  unsupported release-grade, production-readiness or benchmark claim, and a document may call the notebook's
  `Run all` verified only if it names the current notebook blob and that blob is recorded in this file (KTT-m1);
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`,
`tests/test_kokoro_tts_colab_review_fixes.py`; injected runner, no weights). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | fresh Colab CPU or GPU runtime, Linux x86_64, default kernel (Python 3.13 at the time of the review) | The runtime the tutorial is written for; a one-pass **Run all** here is promotion evidence |
| Kaggle kernel or an equivalent fresh Linux container | fresh CPU or GPU container; the committed notebook executed verbatim, cell by cell, in one kernel, with **no repository checkout** | Clean-room executor of the same class; promotion evidence when it completes in one pass with no restart |
| Local harness (pre-flight only) | workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (install and routing skipped) | Builder pre-flight to catch defects; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green on it;
2. open that exact notebook revision in a new runtime (Colab with its default kernel, or Kaggle) with **no
   repository checkout**, an empty Hugging Face cache and no pre-staged `kokoro-v1_0.pth` or `voices/*.pt` under
   `weights/kokoro-82m/`;
3. choose **Run all** once, without editing implementation cells (form parameters at their defaults for the sample
   path: `USE_BYOD = False`, `BYOD_PATH = ''`, `VOICE = 'af_heart'`, `SPEED = 1.0`, `SEED = 0`); the run must
   complete in one pass with **no restart** (`restarted: false` in the record); a run that needed a restart is not
   promotion evidence;
4. verify that Section 1 prints `isolated_python` `3.12.12`, the kernel's Python version and 112 locked packages;
   that the runtime cell reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from`; and that the imported versions equal the inline `PINS` (= the `pyproject.toml`
   pins: `kokoro==0.9.4`, `misaki==0.9.4`, `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`,
   `numpy==2.5.3`, `huggingface-hub==0.36.2`, `soundfile==0.14.0`);
5. verify every default-path stage completes:
   - the carried module cell executing (defining `KokoroTTSPipeline`, `validate_inputs`, `evaluation_report`,
     `list_voices`, `text_lines`, `decode_byod_text` and the ceilings) with no import of the repository package;
   - the synthetic pangram authored in code with its text SHA-256 printed;
   - the inline `MANIFEST` asserted against the module identity and written to `weights/kokoro-82m/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reporting all 58 manifest entries fetched in an empty
     standalone weights directory from `hexgrad/Kokoro-82M` at the immutable revision, `verify_snapshot` returning
     the 58-entry manifest, `list_voices` returning 54 names including `af_heart`, and
     `KokoroTTSPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with
     `lang_code 'a'`, `source 'local-snapshot'` and the observed `espeak_fallback` value recorded;
   - ceilings `MAX_TEXT_CHARS = 2000`, `MIN_SPEED = 0.5`, `MAX_SPEED = 2.0`, `SAMPLE_RATE = 24000`,
     the nine `LANG_CODES` and `DEFAULT_LANG_CODE = 'a'` printed, and `validate_inputs` writing
     `outputs/kokoro_tts_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     `bf_emma` wrong-language probe) before model execution;
   - `synthesize` returning a 1-D float32 array at 24000 Hz with all six sanity checks true, one line with one chunk
     whose `phonemes` string voices every word of the pangram, `skipped_lines` empty, an inline audio player, and
     `outputs/kokoro_tts_sample.wav` written as 16-bit PCM;
   - `evaluation_report` writing `outputs/kokoro_tts_evaluation_report.json` with verdict `not-measurable` and an
     empty `metrics` list, and the "No metric is reported" line printed;
   - `outputs/kokoro_tts_result.json` written with the WAV path and digest, the audio facts, the chunks and line
     coverage, the request including the seed, `espeak_fallback`, `NOTEBOOK_SOURCE`, model identifier, immutable model
     revision, model licence, weight format and loader facts, snapshot summary, runtime versions, device and dtype;
6. BYOD gate: with `USE_BYOD = True`, run Section 4 onward (**Runtime → Run after**) with (a) a two-line UTF-8 text
   whose first line is a ~600-character paragraph and (b) a two-line text whose second line is `***` (symbols only;
   it phonemises to nothing on every host — an all-capital word such as `XQZV` does not, because `misaki` spells it
   letter by letter) — (a) must finish Section 6 with more than one chunk for line 0, (b) must finish with a warning
   naming line 1; and cancel the upload once, which must stop with `no file uploaded; rerun this cell and choose one
   UTF-8 .txt file`;
7. verify the exports exist, that the WAV is audible speech of the pangram (a listener's observation,
   not a metric), and that the interpretation section matches the observed path;
8. record the notebook Git blob id, commit, runtime (platform, kernel Python, isolated Python, PyTorch, NumPy,
   `kokoro`, `misaki`, `soundfile`, device), `restarted: false`, model identifier and immutable revision, whether the
   model cache and weights directory were clean, outcome, produced outputs, `num_samples`/`duration_s`/`peak_amplitude`
   and the phoneme string (as observations, not a metric), and any warning or applicable `SHOULD` deviation in the
   table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/kokoro_tts_colab.ipynb` | `bf509157c7133a863aa891bf7b88087b1464b946` / `ca60f5f4f3b6c3dbf48f44a0aeaeedab4c7880ad` | 2026-09-13 | Colab CLI driver → separately built Python 3.12.3 venv, T4; cells exec'd one by one | Completed 8/8 cells for that earlier blob — **not a `Run all` of the notebook in a supported kernel and not promotion evidence for the current notebook**; [Retained run](verification/2026-09-13/README.md) |

## Recorded executions

Notebook identity is the Git blob of `tutorials/kokoro_tts_colab.ipynb` at the source commit in the row below. The documentation commit recording the run does not change that notebook blob. Cell wall time is the sum of recorded code-cell times, including installation and model downloads; total time additionally includes environment setup and bookkeeping. These measurements describe this one run.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Cell wall / total | Outcome |
|---|---|---|---|---|---|
| 2026-09-13 | `bf509157c7133a863aa891bf7b88087b1464b946` / `ca60f5f4f3b6c3dbf48f44a0aeaeedab4c7880ad` | Colab CLI → fresh Python 3.12.3 venv/interpreter built by the driver, not by the notebook; Tesla T4, 15,360 MiB | Unchanged default sample of that earlier blob, no repository checkout, empty per-model cache and weights | 184.518 s / 188.984 s | Completed 8/8 cells — not a `Run all`; listening and evidence review pending; [Retained run](verification/2026-09-13/README.md) |

The run used PyTorch `2.14.0+cu130`, `cuda:0` and `float32`. All eight code cells completed, runtime pins matched, every snapshot file was SHA-256 verified, inputs were accepted, and the negative validation probe was recorded. Results, model identity/revision, observed output, warnings, package versions, notebook outputs, executor source and cleanup evidence are retained in [the run record](verification/2026-09-13/README.md).

The native hosted kernel was Python 3.13.15; the direct notebook attempt there was aborted in installation, because `kokoro` 0.9.4 and `misaki` 0.9.4 declare `Requires-Python <3.13`. The run above therefore exercised the notebook's cells, not its install path: the driver built the Python 3.12 environment itself. On a Python 3.12 kernel the former in-kernel install would also have replaced pre-imported NumPy/torch and stopped with a restart request, which is not a one-pass `Run all` either (review KTT-B1, KTT-M1). The current notebook replaces that install with an isolated `uv` environment whose managed Python 3.12.12 does not depend on the kernel; it has not been executed on a hosted runtime yet.

## Current status

**Candidate.** No `Run all` of the current notebook blob is recorded. The 2026-09-13 run is default-sample inference/contract evidence for an earlier blob; it does not establish model quality or a benchmark result, and Kokoro listening review is still pending (WAV structure and digest were checked, audible intelligibility and perceptual quality were not). Outstanding before promotion: a one-pass hosted `Run all` of the current blob on Colab (default kernel) and Kaggle with no restart, the BYOD gate in step 6, evidence review and the listening review. This documentation performs no promotion.
