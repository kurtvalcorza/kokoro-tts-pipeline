# Kokoro-82M v1.0 — Colab T4 execution evidence

Execution date (UTC): 2026-09-13. Outcome: **PASS — 8/8 unchanged code cells**, with GPU device `cuda:0`.

| Provenance | Value |
|---|---|
| Tested repository commit | `bf509157c7133a863aa891bf7b88087b1464b946` |
| Source notebook Git blob | `ca60f5f4f3b6c3dbf48f44a0aeaeedab4c7880ad` |
| Source notebook SHA-256 | `d3bc88189779e9a8db596e9d752e38082c1ac800650d8a907f9bcf0d70ceae72` |
| Embedded source revision | `8e72764628b91a6c2bd5f25d437b0755ac7cbf38` |
| Model | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| GPU / driver | `Tesla T4, 15360 MiB, 580.82.07` |
| Runtime | `{"device": "cuda:0", "dtype": "float32", "kokoro": "0.9.4", "misaki": "0.9.4", "python": "3.12.3", "soundfile": "0.14.0", "torch": "2.14.0+cu130"}` |
| Sum of code-cell wall times | 184.518 s |
| Total with environment setup and bookkeeping | 188.984 s |

## Execution method

Colab CLI 0.6.0 ran a driver from WSL `claude-science` on one Tesla T4 VM. The driver created a separate Python 3.12.3 virtual environment for this notebook and launched a new interpreter. Each original code cell was executed sequentially with `exec(compile(...))`; source cells and default form parameters were unchanged. The executor source is retained as [executor-source.txt](executor-source.txt), an evidence artifact rather than repository tooling. The VM had no repository checkout. The weights directory and isolated Hugging Face cache were empty before this notebook ran; all snapshot entries were downloaded and SHA-256 verified by the embedded pipeline.

The hosted Colab kernel used Python 3.13.15. An earlier direct `.ipynb` attempt was aborted during installation after that mismatch was confirmed. The successful result here uses the repository-supported Python 3.12 interpreter. A completed native hosted-kernel run is not claimed.

CLI transport prerequisite: PyPI `jupyter-kernel-client==1.0.2` lacked `KernelClient`; the CLI environment used Google’s fork at `f18e982c3265df5e923aa9def101ab3fd737e139` (distribution 0.8.0). This affects the CLI host, not the notebook runtime pins.

## Observations

- Voice `af_heart`, language `a`, speed 1.0, seed 0.
- 78000 samples; 3.25 s audio; peak 0.337203; synthesis 2.667 s.
- 58 verified snapshot entries; 54 voice packs; `espeak_fallback=True`.
- WAV SHA-256: `daf4a8cf38a11e3c19fc316ac3ff5a9b5524e8ef33d18846c5d1fcc41f67fc55`.
- Phonemes: `ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.`.

All six waveform sanity checks passed. The downloaded WAV was independently checked for its digest, 24 kHz rate, mono channel, PCM16 sample width and 78,000 frames. `misaki` downloaded and installed spaCy `en_core_web_sm==3.8.0` on first use (see the execution log). Warnings concerned one-layer LSTM dropout and deprecated `weight_norm` / `torch.jit.script`; none prevented execution. The plain interpreter had no inline audio player, so the WAV is retained for playback.

**Listening review is pending.** Neither audible intelligibility nor perceptual quality was verified. The evaluation verdict is `not-measurable`, with no MOS or ASR-based WER.

## Verification and retained files

The read-back checks confirmed the exact source hash and Git blob, unchanged code cells, eight error-free executed cells, runtime pins, CUDA inference, accepted inputs, and the recorded rejection probe. Model-specific sanity checks and exported artifacts were checked after download. See [execution-record.json](execution-record.json) for the individual checks and per-cell timings.

- [Executed notebook](kokoro_tts_colab_output.ipynb)
- [Execution log](execution.log)
- [Installed distributions](packages-after.json)
- [Artifact SHA-256 manifest](artifacts.sha256)
- [kokoro_tts_evaluation_report.json](outputs/kokoro_tts_evaluation_report.json)
- [kokoro_tts_input_manifest.json](outputs/kokoro_tts_input_manifest.json)
- [kokoro_tts_result.json](outputs/kokoro_tts_result.json)
- [kokoro_tts_sample.wav](outputs/kokoro_tts_sample.wav)

The CLI stopped the shared runtime after all four notebook tests; a subsequent `colab sessions` call returned no active sessions. Both cleanup outputs are retained in the execution record. Repository release status remains **Candidate** pending evidence review; this record does not promote it.
