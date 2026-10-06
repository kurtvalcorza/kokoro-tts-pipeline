# Kokoro-82M text-to-speech tutorial — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/kokoro-tts-pipeline`  
**Notebook:** `tutorials/kokoro_tts_colab.ipynb`  
**Reviewed commit:** `223df80b8125ce95cb9c8699f38f57d909cd1096` (`main`, merge of PR #7)  
**Notebook Git blob:** `6447a068cc7086a5a347b1cf0270b494c909ecbd`  
**Finding prefix:** `KTT`  
**Framework:** Notebook Review Framework v1; requirements baseline DIMER Notebook Specification **2.2** (2026-09-26, `ml-worker` `origin/main` `b1cfe13`)

## Executive assessment

The notebook teaches well. It says plainly that speech quality has no intrinsic metric here, and it never reports a
quality score. It explains the pickle trust boundary, seeds the stochastic vocoder, and limits its seed claim to one host
and one build. Every snapshot file is digest-verified before it is loaded. The cross-language voice rejection is shown
and recorded rather than swallowed. The carried module matches the repository (`tools/build_notebook.py --check` OK;
offline suite 30 passed).

**It cannot be installed on the runtime its badge opens.** The notebook installs `kokoro==0.9.4` and `misaki==0.9.4`
into the kernel with `pip`. On PyPI both declare `requires_python <3.13` (probe `P1`), and this repository's own
release record says the hosted Colab kernel is **Python 3.13.15** and that the direct notebook attempt there was
"aborted in installation". The only PASS on record ran the cells in a separately built Python 3.12.3 venv, which the
notebook itself does not create, and it ran an **earlier notebook blob** (`ca60f5f4…`, not `6447a068…`). This is
**KTT-B1**.

On a Python 3.12 kernel the same in-kernel install replaces pre-imported `torch`/`numpy`, so the stale-import guard
halts `Run all` and asks for a restart. The release record calls that halt correct (**KTT-M1**). The fleet's `uv`
isolated-environment pattern fixes both problems at once, because it brings its own managed Python 3.12.

Also found: four Minor findings and three Suggestions. The Minors cover the evidence claim, the BYOD sanity check,
BYOD recovery outside Colab, and spec-version drift.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `223df80` (merge of PR #7). Notebook last changed in `f06e57e` (torchvision/torchaudio pins); `metadata.dimer.generated_from.revision` = `943077dd` (the module's revision) |
| Profile / mode | `TASK-INFERENCE` / `GUIDED` (opening and `metadata.dimer`) |
| Spec declared / applied | 2.0 / 2.2 |
| Audience | Basic Python; knows what a phoneme string is |
| Prerequisites stated | "Google Colab or Jupyter, Python 3.12"; CPU float32 default, CUDA used when present |
| Supported runtime | Google Colab (Open-in-Colab badge, primary), Jupyter |
| Promised outcomes | Pinned install; carried module; 58-file digest-verified snapshot; synthetic pangram (or BYOD text file); validated input manifest with a recorded wrong-language rejection; seeded synthesis to a 24 kHz PCM16 WAV; `not-measurable` evaluation report naming external judges; export with provenance |
| Generator | `tools/build_notebook.py` (install cell, lines 40–70) from `tools/notebook_template.py` |

### Evidence actually obtained

**Source inspection.** All 19 cells (8 code, 11 markdown), the generator and template, `pyproject.toml`
(`requires-python = ">=3.12,<3.13"`), `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`,
`docs/release-verification.md`, and the retained 2026-09-13 run (`docs/verification/2026-09-13/`: README,
execution record, execution log). Also upstream `hexgrad/kokoro` `kokoro/pipeline.py` from GitHub `main` (last
commit `6d87f4a`, its `pyproject.toml` says version 0.9.4). This is **not** the PyPI wheel byte for byte; where a
finding relies on it, that is stated.

**Documented execution evidence.** One run: 2026-09-13, commit `bf509157` / **blob `ca60f5f4`**, Colab CLI driving a
**separately created Python 3.12.3 venv** on a T4, cells executed with `exec(compile(...))`, 8/8 PASS, 184.5 s cell
time, `espeak_fallback=True`, 78,000 samples, peak 0.337. The same record states that the native kernel was Python
3.13.15, that the direct notebook attempt "was aborted in installation", and that "the regenerated notebook has not had
a fresh GPU execution". **No execution evidence exists for the reviewed blob `6447a068`** (probe `P2`).

**Direct execution (CPU, Windows, conda env `eo-notebook-test`, numpy only; no kokoro/misaki, no weights).**
`run_probes.py` probes `P0`–`P5`: blob identity; PyPI `requires_python` for the two pinned packages evaluated
against Python 3.13.15 and 3.12.12; evidence blob vs reviewed blob; spec-version statements; the carried module (cell 5)
exercised with injected stand-in runners to test the cell-13 sanity checks and validation messages; learner-facing text
checks. Repository offline suite: `PYTHONPATH=src pytest tests` → 30 passed, exit 0; `tools/build_notebook.py --check`
→ up to date.

**Not verified.** Any synthesis with real weights; the BYOD upload; the native Colab install failure itself (inferred
from package metadata plus the repository's record; not reproduced here); listening review (pending in the
repository's own record).

### Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well oriented, with honest framing of the absent metric. The stated "Python 3.12" runtime contradicts what Colab provides, and nothing tells the learner (KTT-B1) |
| Clean default | Documented execution (earlier blob, isolated 3.12 venv) | **Not verified for this blob.** The supported in-kernel path is inferred to fail at install on Colab (KTT-B1) and to need a restart on a 3.12 kernel (KTT-M1) |
| Active learning | Direct execution (stand-in) | `VOICE`/`SPEED` reach `validate_inputs`/`synthesize`; a wrong-language voice gets a clear message. Real rerun with weights not verified |
| Reuse and recovery | Direct execution (stand-in) + source | BYOD not verified with weights. Long-line and out-of-dictionary BYOD text trips a false sanity-check failure (KTT-m2). A cancelled upload or a non-Colab Jupyter gives a bare error (KTT-m3) |

## 2. Promise and objective tracing

| Claim | Cell | Observable result | Status |
|---|---|---|---|
| Run all in a fresh supported runtime completes with no intervention | 0, 3 | Install on the Colab kernel | **Not delivered on Colab (KTT-B1); restart on 3.12 kernels (KTT-M1)** |
| Carried module verbatim, parity-tested | 4, 5 | `build_notebook.py --check` OK; 30 tests pass | Holds (direct execution) |
| 58-file snapshot digest-verified before load | 7 | `verify_snapshot` before `from_pretrained` | Holds (source; documented for earlier blob) |
| Validation before the model runs; wrong-language voice rejected and recorded | 11 | `voice 'bf_emma' is not a lang_code='a' voice` | Holds (direct execution, P4) |
| Seeded synthesis; seed controls one host/build only | 12, 13 | `torch.manual_seed(SEED)` before `synthesize` | Holds (source) |
| "The library truncates any single segment above 510 phonemes" | 8 | Upstream `en_tokenize` **splits** long lines into chunks of at most 510 phonemes | **Inaccurate; it also breaks the sanity check (KTT-m2)** |
| Evaluation always `not-measurable`, names MOS / ASR WER | 14, 15 | Report fields | Holds (source; documented for earlier blob) |
| BYOD: one UTF-8 text file ≤ 2,000 chars through the same stages | 0, 8, 9 | Long single line → RuntimeError in cell 13 (stand-in) | **Partly delivered (KTT-m2, KTT-m3)** |

| Objective | Learner activity | Evidence exercised |
|---|---|---|
| Install the pinned runtime | Run cell 3 | Fails on Colab (KTT-B1) |
| Validate the request into an input manifest | Read the printed manifest, including the rejection finding | Yes |
| Read the output contract | Printed facts, sanity checks, phonemes | Yes |
| Read why no metric is reported | Printed report and closing line | Yes |
| Change voice/speed and compare | "Next experiments" prose; the voice inventory is never printed (KTT-S1) | Optional, prose only (KTT-S2) |

## 3. Separate judgments

- **Technical correctness:** the carried code is sound and parity-tested. The installation strategy does not fit the
  supported runtime (KTT-B1, KTT-M1). The BYOD sanity check rests on a false segment-count assumption (KTT-m2).
- **Promise fulfilment:** the central promise, one-pass Run all on the Colab badge's runtime, is not met. The
  inference and reporting promises hold on the earlier blob's isolated-venv run.
- **Learner experience:** strong explanations and honest limits. The weak spots are the misleading 510-phoneme
  "truncation" sentence, the unlisted voices, and recovery on BYOD errors.
- **Spec conformance (2.2):** fails RUN1, RUN10, ENV6, REL1/REL2 (no evidence for this blob, and the recorded run is
  not a Run all of the notebook in a supported kernel), and REL11. The notebook declares 2.0, not 2.2 (KTT-m4).

## 4. Findings

### Blocker

#### KTT-B1 — The pinned install cannot succeed on the Colab kernel (Python 3.13.15): kokoro/misaki 0.9.4 require Python < 3.13

- **Cell/section:** Section 1 install cell (cell 3; generator `tools/build_notebook.py` lines 40–70), Prerequisites
  (cell 1; `tools/notebook_template.py` line 80), opening "Run all" paragraph (cell 0).
- **Observed issue:** cell 3 runs `pip install -q kokoro==0.9.4 misaki==0.9.4 …` into the kernel's own interpreter,
  and nothing checks the Python version first. On PyPI, `kokoro` 0.9.4 declares `requires_python <3.13,>=3.10` and
  `misaki` 0.9.4 declares `<3.13,>=3.8`. Neither allows 3.13.15. This repository's release record names the hosted Colab
  kernel as Python 3.13.15 and says the direct notebook attempt "was aborted in installation after the Python-version
  mismatch was confirmed". The recorded PASS is not the notebook's install path: an external driver built a separate
  Python 3.12.3 venv and exec'd the cells in it.
- **Consequence:** a learner who opens the badge and selects Run all gets an install failure in cell 3. pip finds no
  distribution compatible with the interpreter, `check=True` raises `CalledProcessError`, and no message mentions
  Python. Every later stage is unreachable. The prerequisite "Python 3.12" gives no remedy, because a Colab user cannot
  choose the kernel's Python.
- **Evidence:** direct (probe `P1`: live PyPI metadata; `SpecifierSet` rejects 3.13.15 for both packages); documented
  (`docs/release-verification.md`, "native hosted kernel was Python 3.13.15 … aborted in installation"); source (no
  version guard before `subprocess.run`). The pip failure on Colab is **inferred** from this metadata and the record.
  It was not reproduced here.
- **Recommended correction:** adopt the fleet's `uv` isolated-environment pattern. A carrier cell bootstraps `uv`,
  creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with
  `uv pip install --require-hashes --only-binary :all:`, and runs the synthesis workload in that environment. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (also
  `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`). Implement it in
  the generator (`tools/build_notebook.py` install-cell block, `tools/notebook_template.py`). Update the Prerequisites
  so they say the notebook supplies its own Python 3.12, whatever the kernel version.
- **Acceptance check:** on a fresh Google Colab runtime with the default kernel (record its Python version), Run all of
  the new notebook blob completes every code cell with no error and no restart. The record names the blob, the kernel
  Python and the isolated-environment Python. Also, `grep "sys.executable, '-m', 'pip', 'install'"` finds no match in
  the notebook.
- **Spec:** RUN1 (MUST), REL11 (MUST), ENV1 (MUST), ENV4 (MUST).

### Major

#### KTT-M1 — On a Python 3.12 kernel, Run all needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (generator `tools/build_notebook.py` lines 52–70); Troubleshooting (cell 18).
- **Observed issue:** the cell snapshots the distributions already imported, installs exact pins (`torch==2.14.0`,
  `numpy==2.5.3`, `torchvision`, `torchaudio`) into the kernel, and raises
  `RuntimeError(... 'Restart the runtime, then rerun from the top.')` when any of them changed. Hosted kernels pre-import
  NumPy (and usually torch) at versions other than these pins. The release record says the fresh-container executor
  is "needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins,
  because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path". Sibling fleet notebooks with
  the same install cell stop at this guard on pass 1 of a Kaggle T4 run (numpy 2.0.2 → 2.5.3) and pass only after a
  restart.
- **Consequence:** even where KTT-B1 does not apply, Run all does not complete in one pass. The learner must restart
  and run again, which the opening says is unnecessary.
- **Evidence:** source inspection; the repository's own record (statement above). Not executed for this repository's
  blob; no hosted in-kernel run of this notebook exists.
- **Recommended correction:** the same `uv` isolated-environment pattern as KTT-B1 (reference
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`), so the kernel's
  pre-loaded NumPy/torch are never replaced. Do not add another install guard. Drop the restart line from
  Troubleshooting once it no longer applies.
- **Acceptance check:** the KTT-B1 Colab run plus a fresh Kaggle T4 (Python 3.12 kernel) run of the same blob both
  complete in one pass, with no `Core dependencies changed` error and no restart, and both are recorded in
  `docs/release-verification.md`.
- **Spec:** RUN10 (MUST), ENV6 (MUST), RUN1 (MUST).

### Minor

#### KTT-m1 — No execution evidence for the reviewed blob, yet `tutorials/README.md` marks Run-all "verified"

- **Cell/section:** `tutorials/README.md` table (Run-all column), `README.md` line 60, `STATUS.md`, `docs/release-verification.md`.
- **Observed issue:** the only recorded run is blob `ca60f5f4` at `bf509157`. The reviewed blob is `6447a068`, which
  differs in the carried module (imports moved after validation), the PINS (torchvision/torchaudio added), and the spec
  metadata. That run was an external driver exec'ing cells in an isolated 3.12.3 venv, not Run all of the notebook in
  the supported kernel. The record says this itself ("the regenerated notebook has not had a fresh GPU execution"). The
  tutorials table still says "verified — clean-runtime `Run all` execution recorded (REL1/REL8)".
- **Consequence:** a reader of the table believes the current notebook is Run-all verified, and it is not.
- **Evidence:** direct (probe `P2`: reviewed blob absent from the record; record states no fresh run; claim present).
- **Recommended correction:** after KTT-B1/M1, record a Run all of the new blob on Colab (and Kaggle), and until then
  change the Run-all cell to "pending — last recorded run is blob `ca60f5f4` in an isolated 3.12 venv". Also make the
  validator (`tools/validate_release_assets.py`) refuse a "verified" Run-all token when the current blob is not in the
  record.
- **Acceptance check:** every "verified" Run-all statement in `tutorials/README.md`, `README.md` and `STATUS.md` names a
  blob equal to `git rev-parse HEAD:tutorials/kokoro_tts_colab.ipynb`, and that blob has a row in
  `docs/release-verification.md` whose executor is Run all in a supported kernel.
- **Spec:** REL1, REL2, REL10 (MUST).

#### KTT-m2 — The BYOD sanity check fails on long lines and out-of-dictionary lines; the 510-phoneme sentence is wrong

- **Cell/section:** cell 8 BYOD paragraph (`tools/notebook_template.py` line 98); cell 13 `one_segment_per_line`
  (`tools/notebook_template.py` line 192); cell 12 prose ("one `graphemes`/`phonemes` pair per newline-split segment").
- **Observed issue:** cell 13 requires `len(result['segments'])` to equal the number of non-blank lines, and raises
  `RuntimeError('synthesize output failed a sanity check: …')` otherwise. In upstream `KPipeline.__call__`, English
  text goes through `en_tokenize`, which **splits** a line whose phonemes exceed 510 into several chunks, each yielded
  as its own result. A chunk with empty phonemes is skipped (`if not ps: continue`), for example a line of only
  out-of-dictionary words when espeak-ng did not bind. Either case changes the segment count. Cell 8 says instead that
  the library "truncates any single segment above 510 phonemes with a warning". A 566-character single-line paragraph is
  accepted by `validate_inputs` (`MAX_TEXT_CHARS` = 2,000).
- **Consequence:** a learner uploading an ordinary paragraph without line breaks (well within the stated 2,000-character
  limit), or a line of names and numbers on a host without espeak-ng, gets a hard RuntimeError listing booleans. It
  reads as a pipeline failure and suggests no fix. The learner is also taught that long text is truncated, when it is in
  fact chunked.
- **Evidence:** source inspection of upstream `hexgrad/kokoro` `main` `kokoro/pipeline.py` (`en_tokenize`, and the
  `__call__` English branch). That source declares version 0.9.4, but it is **inferred**, not confirmed, to match the
  pinned PyPI wheel. Direct execution with stand-in runners (probe `P4`): the long-line stand-in gives 2 segments for 1
  line and `cell13_would_raise: true`; the out-of-dictionary stand-in gives 1 segment for 2 lines and
  `cell13_would_raise: true`; two short lines pass. Not run with real weights.
- **Recommended correction:** in the template, check segment coverage rather than an exact count. For example, every
  non-blank line index appears in the results (expose `text_index` from the runner), or the count is ≥ 1 with a printed
  warning naming any line that produced no audio. Rewrite cell 8's sentence to "a line above 510 phonemes is split into
  several chunks; a line that phonemises to nothing is skipped". Add a Troubleshooting entry for a skipped line.
- **Acceptance check:** with the real pipeline, a single-line BYOD text of about 600 characters and a two-line text
  whose second line is `XQZV BRRT 12345` (with espeak disabled) both finish cell 13 without RuntimeError. The second
  prints a warning that names the skipped line. Cell 8 no longer contains "truncates any single segment".
- **Spec:** DAT12 (MUST: limits stated before upload), UX10 (SHOULD).

#### KTT-m3 — BYOD recovery: Colab-only upload and a bare `StopIteration` on a cancelled upload

- **Cell/section:** cell 9 (`tools/notebook_template.py` lines 106–116).
- **Observed issue:** `from google.colab import files` is unconditional in the BYOD branch, yet the Prerequisites list
  "Jupyter" as a supported runtime. A cancelled or empty upload makes `next(iter(uploaded))` raise a bare
  `StopIteration`. A file that is not UTF-8 raises `UnicodeDecodeError`. None of these failures names a remedy.
- **Consequence:** on Jupyter, BYOD fails with `ModuleNotFoundError: google.colab` and no alternative is given. On
  Colab, cancelling the dialog produces an opaque error.
- **Evidence:** source inspection; direct (probe `P4`: `next(iter({}))` gives a StopIteration with no message).
- **Recommended correction:** offer a `BYOD_PATH` form field used when `google.colab` is not importable. Catch the empty
  upload and raise `ValueError('no file uploaded; rerun this cell and choose one UTF-8 .txt file')`, and catch
  `UnicodeDecodeError` with a similar message.
- **Acceptance check:** with `USE_BYOD=True`: cancelling the upload prints that message; on plain Jupyter, setting
  `BYOD_PATH` to a local `.txt` completes cells 9–17; a Latin-1 file produces an error naming UTF-8.
- **Spec:** UX10 (SHOULD), DAT12 (MUST).

#### KTT-m4 — Spec-version drift: the notebook declares 2.0, the baseline is 2.2, and the release doc says 1.1

- **Cell/section:** cell 0, `metadata.dimer.notebook_spec`, `tutorials/README.md`, `docs/release-verification.md` line 13.
- **Observed issue:** the notebook declares NOTEBOOK_SPEC 2.0 while the fleet baseline is 2.2 (2026-09-26).
  `docs/release-verification.md` says `metadata.dimer` declares "spec `1.1`", which is untrue at this blob.
  `tutorials/README.md` conformance notes still use 1.1 requirement IDs.
- **Consequence:** reviewers and the validator check against stale requirement sets.
- **Evidence:** direct (probe `P3`).
- **Recommended correction:** regenerate against 2.2, update the validator's expected version, and fix the 1.1
  statement.
- **Acceptance check:** `metadata.dimer.notebook_spec == "2.2"`, the opening says 2.2, and no file says the metadata
  declares 1.1.
- **Spec:** the conformance declaration section of NOTEBOOK_SPEC 2.2.

### Suggestions

- **KTT-S1 — Print the usable voice list.** The "Next experiments" section asks the learner to "change `VOICE` to
  another `af_`/`am_` pack", but `pipe.voices` is never printed and `VOICE` is a free-text field. Print the `a`-language
  voices after loading in cell 7, or make the form a dropdown.
- **KTT-S2 — Make one experiment executable.** "Next experiments" is prose only. A gated-off cell that renders `SPEED`
  0.8 and 1.5 and prints `duration_s` for both would give the GUIDED mode a concrete comparison (UX5, SHOULD).
- **KTT-S3 — Close the listening review.** The release procedure (step 6) asks for a listener's observation that the WAV
  is the pangram. The record says that review is still pending. A one-line cue in cell 12 ("you should hear …") would
  give the learner the same check.

## 5. Readiness

**Needs revision.** One Blocker (KTT-B1) and one Major (KTT-M1), both fixed by the same `uv` isolated-environment
carrier. There is no execution evidence for the reviewed blob (KTT-m1), and the applicable MUSTs RUN1, RUN10, ENV6,
REL1, REL2 and REL11 are unmet. Remaining gates after the fix: a recorded one-pass Run all of the new blob on Colab
(default kernel) and Kaggle, the KTT-m2 BYOD acceptance run with real weights, and the listening review.

## 6. Verified vs inferred

- **Verified (direct):** blob identity; PyPI `requires_python` of both pins excludes 3.13.15; the reviewed blob is not
  in the release record; generator parity and 30 offline tests pass; the validation messages; the stand-in behaviour of
  the cell-13 segment check.
- **Documented:** the earlier blob's isolated-venv PASS; the Colab kernel being 3.13.15; the aborted native install.
- **Inferred:** that pip on the Colab kernel fails at cell 3 (from metadata plus the record, not reproduced); that the
  upstream `main` chunking code matches the 0.9.4 wheel; that a 3.12 hosted kernel triggers the restart guard for this
  notebook.
- **Finding most likely to be wrong:** KTT-m2's mechanism. It rests on upstream GitHub `main`, not the installed
  0.9.4 wheel. If the wheel truncates rather than chunks, the long-line half of the finding disappears, though the
  out-of-dictionary half and the wrong prose would remain.
