# Release status

Current status: **Candidate / source-complete**. The pipeline package, offline unit tests (14), one CPU smoke run (float32, Windows venv, in-dictionary English sentence, 2026-09-12) and `MODEL_CARD.md` (MODEL_CARD_SPEC 1.1) exist and pass the static gate. This is the card pass only: no tutorial notebook exists yet, and the repository is not release-grade until a `NOTEBOOK_SPEC` 1.0 tutorial and its clean-runtime execution evidence are recorded. Known environment caveat: `misaki` auto-installs spaCy `en_core_web_sm` on first English use.
