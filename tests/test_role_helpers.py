"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from kokoro_tts_pipeline import (
    DEFAULT_LANG_CODE,
    DEFAULT_VOICE,
    INPUT_SCHEMA,
    LANG_CODES,
    MAX_SPEED,
    MAX_TEXT_CHARS,
    MIN_SPEED,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    evaluation_report,
    validate_inputs,
)

VOICES = ("af_heart", "am_adam", "bf_emma")


def _result(n_segments: int = 1, duration_s: float = 3.25) -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "num_samples": int(duration_s * SAMPLE_RATE),
        "duration_s": duration_s,
        "peak_amplitude": 0.342,
        "segments": [{"graphemes": "hi", "phonemes": "haɪ"} for _ in range(n_segments)],
        "voice": DEFAULT_VOICE,
        "lang_code": DEFAULT_LANG_CODE,
        "speed": 1.0,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs("Line one.\nLine two.", voices=VOICES, names=["pangram"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["speed"] == [MIN_SPEED, MAX_SPEED]
    assert manifest["schema"]["lang_codes"] == list(LANG_CODES)
    assert manifest["schema"]["sample_rate_hz"] == SAMPLE_RATE
    assert manifest["inputs"] == [{"id": "pangram", "chars": 19, "segments": 2}]
    assert manifest["voice"] == DEFAULT_VOICE
    assert manifest["speed"] == 1.0
    assert manifest["lang_code"] == DEFAULT_LANG_CODE
    assert manifest["voice_inventory"] == 3
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_speed() -> None:
    manifest = validate_inputs("Hello.", "am_adam", speed=1.5, voices=VOICES)
    assert [entry["id"] for entry in manifest["inputs"]] == ["text-0"]
    assert (manifest["voice"], manifest["speed"]) == ("am_adam", 1.5)


def test_validate_inputs_rejects_like_synthesize() -> None:
    with pytest.raises(TypeError, match="text must be a str"):
        validate_inputs(None, voices=VOICES)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text must not be empty"):
        validate_inputs("   ", voices=VOICES)
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs("x" * (MAX_TEXT_CHARS + 1), voices=VOICES)
    with pytest.raises(ValueError, match="manifest voices"):
        validate_inputs("Hello.", "af_nonexistent", voices=VOICES)
    with pytest.raises(ValueError, match="is not a lang_code"):
        validate_inputs("Hello.", "bf_emma", voices=VOICES)
    with pytest.raises(TypeError, match="speed must be a number"):
        validate_inputs("Hello.", speed=True, voices=VOICES)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MIN_SPEED"):
        validate_inputs("Hello.", speed=MAX_SPEED + 0.1, voices=VOICES)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs("Hello.", voices=VOICES, names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result(2))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_segments"] == 2
    assert report["duration_s"] == 3.25
    assert report["sample_kind"] == "synthetic"
    assert "no reference recording" in report["reason"]
    assert "Mean Opinion Score" in report["needs"]
    assert "word error rate" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_references_are_supplied() -> None:
    report = evaluation_report(_result(), ["reference.wav"], sample_kind="BYOD upload")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD upload"
    assert "references were supplied but no metric helper exists" in report["reason"]
