# kokoro-tts-pipeline

DIMER pipeline scaffold for **hexgrad/Kokoro-82M** — Text-to-Speech.

| | |
|---|---|
| Upstream model | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| Pinned revision | `f3ff3571791e39611d31c381e3a41a3af07b4987` (resolved 2026-09-12) |
| Upstream license | `apache-2.0` (verified on the Hub 2026-09-12; re-check at the pinned revision before release) |
| Weight files to stage | `kokoro-v1_0.pth + voices/*.pt` |
| Status | scaffold only — no weights downloaded, no pipeline code yet |

Weights are staged under `weights/` and are git-ignored. This repository follows the
MODEL_CARD_SPEC 1.0 / NOTEBOOK_SPEC 1.0 conventions used by the other `*-pipeline` repos.
