import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from kokoro_tts_pipeline import (
    DEFAULT_LANG_CODE,
    DEFAULT_VOICE,
    DEFAULT_WEIGHTS_DIR,
    LANG_CODES,
    LOADER_WEIGHTS_ONLY,
    MAX_SPEED,
    MAX_TEXT_CHARS,
    MIN_SPEED,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    SAMPLE_RATE,
    WEIGHT_FORMAT,
    KokoroTTSPipeline,
    list_voices,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
VOICES = ("af_heart", "af_bella", "bm_george")


def _fake_runner(text, voice_path, speed):
    assert voice_path.endswith(".pt")
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            continue
        yield line, f"ph{i}", np.full(2400, 0.25 * (i + 1), dtype=np.float32)


def _pipeline(**overrides) -> KokoroTTSPipeline:
    kwargs = {"lang_code": "a", "device": "cpu", "source": "injected", "weights_dir": Path("w")}
    kwargs.update(overrides)
    return KokoroTTSPipeline(_fake_runner, VOICES, **kwargs)


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "kokoro-v1_0.pth").write_bytes(payload)
    (root / "voices").mkdir(exist_ok=True)
    (root / "voices" / "af_heart.pt").write_bytes(b"pack")
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "kokoro-v1_0.pth", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
            {"path": "voices/af_heart.pt", "bytes": 4, "sha256": hashlib.sha256(b"pack").hexdigest()},
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "hexgrad/Kokoro-82M"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert SAMPLE_RATE == 24000
    assert WEIGHT_FORMAT == "pytorch-pickle" and LOADER_WEIGHTS_ONLY is True
    assert DEFAULT_LANG_CODE in LANG_CODES and DEFAULT_VOICE[0] == DEFAULT_LANG_CODE
    assert 0 < MIN_SPEED < 1.0 <= MAX_SPEED


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY
    voices = list_voices(manifest)
    assert DEFAULT_VOICE in voices and len(voices) == 54


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)
    assert list_voices(result) == ("af_heart",)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_voice_pack(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "voices" / "af_heart.pt").write_bytes(b"hack")
    with pytest.raises(ValueError, match="voices/af_heart.pt: sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "voices" / "af_heart.pt").write_bytes(b"short-pack")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "voices" / "af_heart.pt").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_from_pretrained_refuses_without_snapshot(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no verified snapshot"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no verified snapshot"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=True)


def test_from_pretrained_refuses_tampered_snapshot_before_loading(tmp_path, forbid_model_imports):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path)
    with pytest.raises(AssertionError, match="model dependency imported before rejection: torch"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_rejects_unknown_lang_code(tmp_path, forbid_model_imports):
    with pytest.raises(ValueError, match="LANG_CODES"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, lang_code="xx")


def test_synthesize_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.synthesize(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.synthesize("   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.synthesize("a" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="manifest voices"):
        pipe.synthesize("hello", voice="af_nobody")
    with pytest.raises(ValueError, match="manifest voices"):
        pipe.synthesize("hello", voice=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lang_code"):
        pipe.synthesize("hello", voice="bm_george")
    with pytest.raises(TypeError):
        pipe.synthesize("hello", speed="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MAX_SPEED"):
        pipe.synthesize("hello", speed=MAX_SPEED + 0.1)
    with pytest.raises(ValueError, match="MIN_SPEED"):
        pipe.synthesize("hello", speed=MIN_SPEED - 0.1)


def test_synthesize_output_fields_and_segment_concatenation():
    pipe = _pipeline()
    result = pipe.synthesize("Hello there.\nSecond line.", voice="af_bella", speed=1.25)
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["sample_rate"] == SAMPLE_RATE
    assert result["audio"].dtype == np.float32 and result["audio"].ndim == 1
    assert result["num_samples"] == 4800
    assert result["duration_s"] == pytest.approx(0.2)
    assert result["peak_amplitude"] == pytest.approx(0.5)
    assert [s["graphemes"] for s in result["segments"]] == ["Hello there.", "Second line."]
    assert result["voice"] == "af_bella" and result["lang_code"] == "a" and result["speed"] == 1.25
    assert result["espeak_fallback"] is None
    assert result["device"] == "cpu" and result["source"] == "injected"


def test_synthesize_raises_when_no_audio():
    pipe = KokoroTTSPipeline(lambda *args: iter(()), VOICES, "a", "cpu")
    with pytest.raises(RuntimeError, match="no audio produced"):
        pipe.synthesize("hello")
