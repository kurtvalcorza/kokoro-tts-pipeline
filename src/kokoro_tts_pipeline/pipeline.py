"""Text-to-speech with the pinned ``hexgrad/Kokoro-82M`` snapshot (24 kHz mono float32).

The checkpoint ``kokoro-v1_0.pth`` and every ``voices/*.pt`` pack are PyTorch pickle files, not SafeTensors.
The trust boundary is therefore: (1) every file is SHA-256-verified against the manifest before it is opened,
and (2) the ``kokoro`` library deserialises both with ``torch.load(..., weights_only=True)`` (recorded in
``LOADER_WEIGHTS_ONLY``), which restricts unpickling to tensors and primitive containers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "kokoro-82m"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "kokoro-v1_0.pth"
CONFIG_FILE = "config.json"
VOICES_DIR = "voices"

WEIGHT_FORMAT = "pytorch-pickle"  # .pth checkpoint and .pt voice packs; not SafeTensors
# kokoro 0.9.4 model.py:68 and pipeline.py:147 call torch.load(..., weights_only=True)
LOADER_WEIGHTS_ONLY = True
SAMPLE_RATE = 24000  # Hz, mono, float32 (upstream README)
LANG_CODES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")  # kokoro.pipeline.LANG_CODES keys
DEFAULT_LANG_CODE = "a"  # American English
DEFAULT_VOICE = "af_heart"
MAX_TEXT_CHARS = 2000  # characters per synthesize() call; the library chunks at 510 phonemes per segment
MIN_SPEED = 0.5
MAX_SPEED = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def list_voices(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Voice names (``af_heart`` …) from the manifest's ``voices/*.pt`` entries, sorted."""
    prefix = f"{VOICES_DIR}/"
    paths = [entry["path"] for entry in manifest["files"]]
    names = [path for path in paths if path.startswith(prefix) and path.endswith(".pt")]
    return tuple(sorted(name[len(prefix) : -3] for name in names))


@dataclass
class KokoroTTSPipeline:
    """``_runner(text, voice_path, speed)`` yields ``(graphemes, phonemes, audio_1d_float32)`` per segment."""

    _runner: Callable[..., Any]
    voices: tuple[str, ...]
    lang_code: str = DEFAULT_LANG_CODE
    device: str = "cpu"
    source: str = "injected"
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    espeak_fallback: bool | None = None  # None = unknown (injected runner)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        lang_code: str = DEFAULT_LANG_CODE,
    ) -> KokoroTTSPipeline:
        import torch
        from kokoro import KModel, KPipeline

        if lang_code not in LANG_CODES:
            raise ValueError(f"lang_code must be one of LANG_CODES {LANG_CODES}, got {lang_code!r}")
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no verified snapshot at {root} and no Hub path is offered for pickle checkpoints; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        stage_missing_files(root, allow_download=allow_download)
        manifest = verify_snapshot(root)
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        kmodel = KModel(repo_id=MODEL_ID, config=str(root / CONFIG_FILE), model=str(root / WEIGHTS_FILE))
        kmodel = kmodel.to(resolved_device).eval()
        kpipeline = KPipeline(lang_code=lang_code, repo_id=MODEL_ID, model=kmodel, device=resolved_device)
        fallback = getattr(getattr(kpipeline, "g2p", None), "fallback", None)

        def runner(text: str, voice_path: str, speed: float) -> Any:
            with torch.inference_mode():
                for result in kpipeline(text, voice=voice_path, speed=speed, split_pattern=r"\n+"):
                    audio = result.audio
                    array = audio.cpu().numpy() if audio is not None else None
                    yield result.graphemes, result.phonemes, array

        return cls(
            runner, list_voices(manifest), lang_code, resolved_device, "local-snapshot", root,
            fallback is not None if lang_code in "ab" else None,
        )

    def _validate(self, text: Any, voice: Any, speed: Any) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        if not text.strip():
            raise ValueError("text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds MAX_TEXT_CHARS={MAX_TEXT_CHARS}: {len(text)}")
        if not isinstance(voice, str) or voice not in self.voices:
            raise ValueError(f"voice must be one of the {len(self.voices)} manifest voices, got {voice!r}")
        if voice[0] != self.lang_code:
            raise ValueError(f"voice {voice!r} is not a lang_code={self.lang_code!r} voice")
        if isinstance(speed, bool) or not isinstance(speed, int | float):
            raise TypeError("speed must be a number")
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between MIN_SPEED={MIN_SPEED} and MAX_SPEED={MAX_SPEED}")

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE, *, speed: float = 1.0) -> dict[str, Any]:
        """Synthesise ``text`` with a named voice pack; ``audio`` is a 1-D float32 array at 24 kHz."""
        self._validate(text, voice, speed)
        voice_path = str(self.weights_dir / VOICES_DIR / f"{voice}.pt")
        chunks: list[np.ndarray] = []
        segments: list[dict[str, str]] = []
        for graphemes, phonemes, audio in self._runner(text, voice_path, float(speed)):
            if audio is None:
                continue
            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            chunks.append(array)
            segments.append({"graphemes": str(graphemes), "phonemes": str(phonemes)})
        if not chunks:
            raise RuntimeError("no audio produced: every segment was empty after grapheme-to-phoneme")
        wave = np.concatenate(chunks)
        return {
            "audio": wave,
            "sample_rate": SAMPLE_RATE,
            "num_samples": int(wave.shape[0]),
            "duration_s": float(wave.shape[0] / SAMPLE_RATE),
            "peak_amplitude": float(np.abs(wave).max()),
            "segments": segments,
            "voice": voice,
            "lang_code": self.lang_code,
            "speed": float(speed),
            "espeak_fallback": self.espeak_fallback,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
