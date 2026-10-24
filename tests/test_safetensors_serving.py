"""The model and voices are served from pinned safetensors conversions; the pickles are read only to convert.

The digest and refusal tests need nothing but the package. The conversion tests swap the 82M model for
a small stand-in and need torch and safetensors. The last two run only where the real snapshot and its
conversions are staged under weights/ (and, for serving, where the kokoro library is installed).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from kokoro_tts_pipeline import pipeline as pipeline_module
from kokoro_tts_pipeline.pipeline import (
    CONFIG_FILE,
    CONVERTED_VOICES_FILE,
    CONVERTED_VOICES_SHA256,
    CONVERTED_WEIGHTS_FILE,
    CONVERTED_WEIGHTS_SHA256,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    VOICES_DIR,
    WEIGHTS_FILE,
    KokoroTTSPipeline,
)

REAL_FILES = (WEIGHTS_FILE, CONVERTED_WEIGHTS_FILE, CONVERTED_VOICES_FILE, f"{VOICES_DIR}/af_heart.pt")
REAL_STAGED = all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in REAL_FILES)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entry(root: Path, relative: str) -> dict:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_manifest(root: Path, relatives) -> None:
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [_entry(root, relative) for relative in relatives],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_converted_files_are_pinned_safetensors() -> None:
    for name, digest in (
        (CONVERTED_WEIGHTS_FILE, CONVERTED_WEIGHTS_SHA256),
        (CONVERTED_VOICES_FILE, CONVERTED_VOICES_SHA256),
    ):
        assert name.endswith(".safetensors")
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert CONVERTED_WEIGHTS_FILE.removesuffix(".safetensors") == WEIGHTS_FILE.removesuffix(".pth")
    assert pipeline_module.WEIGHT_FORMAT == "safetensors"
    assert pipeline_module.SOURCE_WEIGHT_FORMAT == "pytorch-pickle"


def test_validator_declares_the_converted_digests() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert set(validator.EXTERNAL_WEIGHT_DIGESTS) == {CONVERTED_WEIGHTS_SHA256, CONVERTED_VOICES_SHA256}


def test_serving_path_never_unpickles() -> None:
    serving = inspect.getsource(KokoroTTSPipeline.from_pretrained)
    serving += inspect.getsource(KokoroTTSPipeline.synthesize)
    assert "torch.load(" not in serving and "load_file(" in serving and "strict=True" in serving
    # KModel(...) would unpickle a checkpoint itself; the voice is handed to the library as a tensor.
    assert "KModel(" not in serving and "voice=packs[voice]" in serving
    assert '.pt"' not in serving and "VOICES_DIR" not in serving


def test_altered_conversion_is_refused_before_model_imports(tmp_path, forbid_model_imports) -> None:
    (tmp_path / CONFIG_FILE).write_text("{}", encoding="utf-8")
    (tmp_path / CONVERTED_WEIGHTS_FILE).write_bytes(b"not the pinned conversion")
    (tmp_path / CONVERTED_VOICES_FILE).write_bytes(b"not the pinned voices")
    _write_manifest(tmp_path, [CONFIG_FILE])
    with pytest.raises(ValueError, match="sha256"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_altered_config_is_refused_before_model_imports(tmp_path, forbid_model_imports) -> None:
    (tmp_path / CONFIG_FILE).write_text("{}", encoding="utf-8")
    _write_manifest(tmp_path, [CONFIG_FILE])
    (tmp_path / CONFIG_FILE).write_text('{"n_token": 1}', encoding="utf-8")
    for name in (CONVERTED_WEIGHTS_FILE, CONVERTED_VOICES_FILE):
        (tmp_path / name).write_bytes(b"present")
    with pytest.raises(ValueError, match="config.json: size or sha256"):
        KokoroTTSPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A manifest-pinned stand-in snapshot in the upstream layout and the digests its conversion must have."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(3, 4)
            self.norm = torch.nn.InstanceNorm1d(4, affine=True)

    def build(_config=None):
        blocks = torch.nn.ModuleDict({"blk": Block()})
        return torch.nn.ModuleDict({"enc": torch.nn.Linear(3, 4), "dec": blocks})

    torch.manual_seed(0)
    source = build()
    # Upstream layout: a state dict per component, every key prefixed "module.", InstanceNorm affine omitted.
    checkpoint = {
        name: {
            f"module.{k}": v
            for k, v in getattr(source, name).state_dict().items()
            if ".norm." not in f".{k}"
        }
        for name in ("enc", "dec")
    }
    root = tmp_path / "weights"
    (root / VOICES_DIR).mkdir(parents=True)
    torch.save(checkpoint, root / WEIGHTS_FILE)
    voices = {"af_heart": torch.randn(5, 1, 8), "bm_george": torch.randn(5, 1, 8)}
    for name, pack in voices.items():
        torch.save(pack, root / VOICES_DIR / f"{name}.pt")
    (root / CONFIG_FILE).write_text("{}", encoding="utf-8")
    relatives = [CONFIG_FILE, WEIGHTS_FILE] + [f"{VOICES_DIR}/{name}.pt" for name in voices]
    _write_manifest(root, relatives)
    full = source.state_dict()
    for name, tensors in (("weights", full), ("voices", voices)):
        pipeline_module._write_converted(tensors, tmp_path / f"{name}.safetensors")
    monkeypatch.setattr(pipeline_module, "CONVERTED_WEIGHTS_SHA256", _sha(tmp_path / "weights.safetensors"))
    monkeypatch.setattr(pipeline_module, "CONVERTED_VOICES_SHA256", _sha(tmp_path / "voices.safetensors"))
    monkeypatch.setattr(pipeline_module, "_build_model", build)
    return torch, root, full, voices, checkpoint, build


def test_conversion_is_byte_reproducible_and_complete(stand_in) -> None:
    torch, root, full, voices, _checkpoint, _build = stand_in
    from safetensors.torch import load_file

    first = pipeline_module.convert_checkpoint(root)["converted"]
    for name in (CONVERTED_WEIGHTS_FILE, CONVERTED_VOICES_FILE):
        (root / name).unlink()
    second = pipeline_module.convert_checkpoint(root)["converted"]
    assert first == second
    assert first[CONVERTED_WEIGHTS_FILE]["sha256"] == pipeline_module.CONVERTED_WEIGHTS_SHA256
    assert first[CONVERTED_VOICES_FILE]["sha256"] == pipeline_module.CONVERTED_VOICES_SHA256
    weights = load_file(str(root / CONVERTED_WEIGHTS_FILE))
    assert set(weights) == set(full) and all(torch.equal(weights[k], full[k]) for k in full)
    packs = load_file(str(root / CONVERTED_VOICES_FILE))
    assert set(packs) == set(voices) and all(torch.equal(packs[k], voices[k]) for k in voices)
    assert not list(root.glob("*.partial"))


def test_conversion_with_another_digest_is_refused_and_leaves_nothing(stand_in, monkeypatch) -> None:
    _torch, root, *_ = stand_in
    monkeypatch.setattr(pipeline_module, "CONVERTED_VOICES_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline_module.convert_checkpoint(root)
    assert not (root / CONVERTED_WEIGHTS_FILE).exists() and not (root / CONVERTED_VOICES_FILE).exists()
    assert not list(root.glob("*.partial"))


def test_model_state_refuses_a_missing_trained_tensor(stand_in) -> None:
    _torch, _root, _full, _voices, checkpoint, build = stand_in
    damaged = {name: dict(state) for name, state in checkpoint.items()}
    del damaged["enc"]["module.bias"]
    with pytest.raises(ValueError, match="not a default InstanceNorm parameter"):
        pipeline_module._model_state(damaged, build())


def test_model_state_refuses_an_unexpected_key(stand_in) -> None:
    torch, _root, _full, _voices, checkpoint, build = stand_in
    damaged = {name: dict(state) for name, state in checkpoint.items()}
    damaged["enc"]["module.extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected keys"):
        pipeline_module._model_state(damaged, build())


@pytest.mark.skipif(not REAL_STAGED, reason="real snapshot and conversions not staged")
def test_real_conversion_equals_the_source_snapshot() -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    manifest = pipeline_module.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    pipeline_module.verify_converted(DEFAULT_WEIGHTS_DIR)
    checkpoint = torch.load(DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE))
    for component, state in checkpoint.items():
        for key, value in state.items():
            assert torch.equal(converted[f"{component}.{key.removeprefix('module.')}"], value)
    extra = set(converted) - {f"{c}.{k.removeprefix('module.')}" for c, s in checkpoint.items() for k in s}
    assert extra and all(key.endswith((".norm.weight", ".norm.bias")) for key in extra)
    packs = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_VOICES_FILE))
    names = pipeline_module.list_voices(manifest)
    assert tuple(sorted(packs)) == names
    for name in names:
        path = DEFAULT_WEIGHTS_DIR / VOICES_DIR / f"{name}.pt"
        source = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(packs[name], source) and packs[name].dtype == source.dtype


@pytest.mark.skipif(not REAL_STAGED, reason="real snapshot and conversions not staged")
def test_real_serving_path_never_calls_torch_load(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kokoro")

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    pipe = KokoroTTSPipeline.from_pretrained(device="cpu")
    result = pipe.synthesize("Hello from the converted voice.", voice="af_heart")
    assert result["num_samples"] > 0 and len(pipe.voices) == 54
