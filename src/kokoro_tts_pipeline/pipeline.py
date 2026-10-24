"""Text-to-speech with the pinned ``hexgrad/Kokoro-82M`` snapshot (24 kHz mono float32).

The upstream checkpoint ``kokoro-v1_0.pth`` and every ``voices/*.pt`` pack are PyTorch pickle files, not
SafeTensors. They are staged once, SHA-256-verified against the manifest before any is opened, read once with
``torch.load(..., weights_only=True)`` (recorded in ``LOADER_WEIGHTS_ONLY``), which restricts unpickling to
tensors and primitive containers, and converted into two safetensors files whose SHA-256 digests are pinned
below: ``kokoro-v1_0.safetensors`` (the model's state dict) and ``voices.safetensors`` (one tensor per voice,
keyed by voice name). The model and the voices are served only from those files; once they exist the pickles
are never read again.

The ``kokoro`` library's own ``KModel.__init__`` always unpickles a checkpoint (it downloads one when none is
given), so ``_build_kmodel`` is a small shim that builds the same modules from ``config.json`` without loading
weights, mirroring ``kokoro==0.9.4`` ``model.py`` lines 41-64. ``load_state_dict(strict=True)`` then loads the
converted file. Voice packs are handed to ``KPipeline`` as tensors, which its ``load_voice`` returns
unchanged, so the library never opens a ``.pt`` either.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
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

WEIGHT_FORMAT = "safetensors"  # what is served: the pinned conversions below
SOURCE_WEIGHT_FORMAT = "pytorch-pickle"  # the upstream .pth checkpoint and .pt voice packs; not SafeTensors
# The one-time conversion reads the pickles with torch.load(..., weights_only=True), as kokoro 0.9.4 itself
# would.
LOADER_WEIGHTS_ONLY = True
KOKORO_VERSION = "0.9.4"  # the library version `_build_kmodel` mirrors; any other version is refused
# The served files, converted once from the verified pickles. Their digests are pinned, so the conversion is
# byte-reproducible and a different file is refused.
CONVERTED_WEIGHTS_FILE = "kokoro-v1_0.safetensors"
CONVERTED_WEIGHTS_SHA256 = "bf109333168098b2200e878959872ba8a095b089d9c18f9d0d36b25cded3ca5f"
CONVERTED_VOICES_FILE = "voices.safetensors"
CONVERTED_VOICES_SHA256 = "35597f0f9fab89c0da7366cfc8550ff78e10c453dc15d15eaba374ff1f0fba1e"
CONVERTED_METADATA = {"format": "pt"}
# Checkpoint entries are prefixed "module." (saved from a DataParallel wrapper); the library strips it.
CHECKPOINT_PREFIX = "module."
# The checkpoint omits the affine parameters of the InstanceNorm layers inside the AdaIN blocks; kokoro loads
# with strict=False and leaves them at their constructed values (weight 1, bias 0). The conversion stores
# them, so serving can load strict=True; it refuses any other missing key or any non-default value.
DEFAULT_NORM_SUFFIXES = (".norm.weight", ".norm.bias")
SAMPLE_RATE = 24000  # Hz, mono, float32 (upstream README)
LANG_CODES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")  # kokoro.pipeline.LANG_CODES keys
DEFAULT_LANG_CODE = "a"  # American English
DEFAULT_VOICE = "af_heart"
MAX_TEXT_CHARS = 2000  # characters per synthesize() call; the library chunks at 510 phonemes per segment
SPLIT_PATTERN = r"\n+"  # the library splits the text into lines on this pattern before phonemising each line
MAX_SEGMENT_PHONEMES = 510  # a line whose phonemes exceed this is split by the library into several chunks
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


def _verify_entry(root: Path, manifest: Mapping[str, Any], relative_path: str) -> Path:
    """Check one manifest-listed file's size and SHA-256 (``verify_snapshot`` for a single entry)."""
    entries = [entry for entry in manifest["files"] if entry["path"] == relative_path]
    if len(entries) != 1:
        raise ValueError(f"manifest lists {relative_path} {len(entries)} times, expected once")
    file_path = root / relative_path
    if not file_path.is_file():
        raise FileNotFoundError(f"snapshot file missing: {file_path}")
    if file_path.stat().st_size != entries[0]["bytes"] or _sha256(file_path) != entries[0]["sha256"]:
        raise ValueError(f"{relative_path}: size or sha256 differs from the manifest")
    return file_path


def _read_manifest(root: Path) -> dict[str, Any]:
    with open(root / MANIFEST_NAME, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}"
        )
    return manifest


def _build_kmodel(config: Mapping[str, Any]) -> Any:
    """A ``kokoro.KModel`` with freshly constructed modules and no weights loaded.

    ``KModel.__init__`` in kokoro 0.9.4 builds these modules and then always calls ``torch.load`` on a
    checkpoint. This shim builds the same modules with the same arguments (``model.py`` lines 41-64,
    ``disable_complex=False``) and skips that load, so the caller can ``load_state_dict`` from safetensors. It
    refuses any other kokoro version.
    """
    from importlib.metadata import version

    installed = version("kokoro")
    if installed != KOKORO_VERSION:
        raise RuntimeError(
            f"_build_kmodel mirrors kokoro=={KOKORO_VERSION}; kokoro=={installed} is installed"
        )
    import torch
    from kokoro import KModel
    from kokoro.istftnet import Decoder
    from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
    from transformers import AlbertConfig

    model = KModel.__new__(KModel)
    torch.nn.Module.__init__(model)
    model.repo_id = MODEL_ID
    model.vocab = config["vocab"]
    model.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
    model.bert_encoder = torch.nn.Linear(model.bert.config.hidden_size, config["hidden_dim"])
    model.context_length = model.bert.config.max_position_embeddings
    model.predictor = ProsodyPredictor(
        style_dim=config["style_dim"],
        d_hid=config["hidden_dim"],
        nlayers=config["n_layer"],
        max_dur=config["max_dur"],
        dropout=config["dropout"],
    )
    model.text_encoder = TextEncoder(
        channels=config["hidden_dim"],
        kernel_size=config["text_encoder_kernel_size"],
        depth=config["n_layer"],
        n_symbols=config["n_token"],
    )
    model.decoder = Decoder(
        dim_in=config["hidden_dim"],
        style_dim=config["style_dim"],
        dim_out=config["n_mels"],
        disable_complex=False,
        **config["istftnet"],
    )
    return model


def _model_state(checkpoint: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """The full state dict ``model`` holds after loading ``checkpoint`` the way kokoro 0.9.4 does.

    Each top-level entry is one sub-module's state dict with ``CHECKPOINT_PREFIX`` on every key; it is
    stripped and loaded ``strict=False``. Any unexpected key, any missing key that is not an InstanceNorm
    affine parameter, or a missing parameter whose value is not the constructed default (weight 1, bias 0) is
    refused.
    """
    import torch

    if set(checkpoint) != {name for name, _ in model.named_children()}:
        raise ValueError(f"checkpoint components {sorted(checkpoint)} do not match the model's")
    for component, state in checkpoint.items():
        if not isinstance(state, Mapping) or any(not isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError(f"checkpoint entry {component!r} is not a state dict of tensors")
        stripped = {k.removeprefix(CHECKPOINT_PREFIX): v for k, v in state.items()}
        result = getattr(model, component).load_state_dict(stripped, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"{component}: unexpected keys {sorted(result.unexpected_keys)[:5]}")
        for key in result.missing_keys:
            value = getattr(model, component).state_dict()[key]
            default = 1.0 if key.endswith(".weight") else 0.0
            if not key.endswith(DEFAULT_NORM_SUFFIXES) or not bool(torch.all(value == default)):
                raise ValueError(f"{component}: missing key {key!r} is not a default InstanceNorm parameter")
    return dict(model.state_dict())


def _write_converted(tensors: Mapping[str, Any], target: Path) -> None:
    """Write tensors as safetensors, byte-reproducibly: each is cloned into its own contiguous CPU storage
    (safetensors refuses shared storage) and the one-key metadata header has a fixed order."""
    from safetensors.torch import save_file

    owned = {name: value.detach().to("cpu").contiguous().clone() for name, value in tensors.items()}
    save_file(owned, str(target), metadata=dict(CONVERTED_METADATA))


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check both converted serving files against their pinned SHA-256; raise on a missing or altered file."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    checked = {}
    for name, pinned in (
        (CONVERTED_WEIGHTS_FILE, CONVERTED_WEIGHTS_SHA256),
        (CONVERTED_VOICES_FILE, CONVERTED_VOICES_SHA256),
    ):
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"converted weights missing: {file_path}")
        digest = _sha256(file_path)
        if digest != pinned:
            raise ValueError(f"{name}: sha256 {digest} != pinned {pinned}")
        checked[name] = {"bytes": file_path.stat().st_size, "sha256": digest}
    return checked


def convert_checkpoint(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the verified pickle checkpoint and voice packs into the two pinned safetensors files, once.

    The snapshot must pass ``verify_snapshot`` first. Every pickle is read with ``torch.load(...,
    weights_only=True)``. The checkpoint is loaded into ``_build_kmodel`` as kokoro does (``_model_state``)
    and the model's full state dict is written to ``CONVERTED_WEIGHTS_FILE``; each ``voices/<name>.pt`` must
    be one float32 tensor and is written under the key ``<name>`` to ``CONVERTED_VOICES_FILE``. Each written
    file must hash to its pinned digest, or both are deleted and the conversion refused.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = verify_snapshot(root)
    import torch

    with open(root / CONFIG_FILE, encoding="utf-8") as fh:
        config = json.load(fh)
    checkpoint = torch.load(root / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{WEIGHTS_FILE} did not unpickle to a mapping of state dicts")
    state = _model_state(checkpoint, _build_model(config))
    voices: dict[str, Any] = {}
    for name in list_voices(manifest):
        pack = torch.load(root / VOICES_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
        if not isinstance(pack, torch.Tensor) or pack.dtype != torch.float32:
            raise ValueError(f"{VOICES_DIR}/{name}.pt did not unpickle to one float32 tensor")
        voices[name] = pack
    partials = []
    try:
        for name, tensors, pinned in (
            (CONVERTED_WEIGHTS_FILE, state, CONVERTED_WEIGHTS_SHA256),
            (CONVERTED_VOICES_FILE, voices, CONVERTED_VOICES_SHA256),
        ):
            partial = root / f"{name}.partial"
            partials.append(partial)
            _write_converted(tensors, partial)
            digest = _sha256(partial)
            if digest != pinned:
                raise ValueError(f"conversion into {name} hashed to {digest}, not the pinned {pinned}")
        for partial in partials:
            partial.replace(partial.with_name(partial.name.removesuffix(".partial")))
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
    return {"converted": verify_converted(root), "tensors": len(state), "voices": len(voices)}


def _build_model(config: Mapping[str, Any]) -> Any:
    """Module-level seam over ``_build_kmodel`` so the offline tests can swap in a small stand-in."""
    return _build_kmodel(config)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str, spoken verbatim; newline-split into segments by the library",
    "text_chars": [1, MAX_TEXT_CHARS],
    "speed": [MIN_SPEED, MAX_SPEED],
    "voice": (
        "one of the manifest voices/*.pt packs whose first letter equals the instance "
        f"lang_code ({DEFAULT_LANG_CODE} by default)"
    ),
    "lang_codes": list(LANG_CODES),
    "sample_rate_hz": SAMPLE_RATE,
    "preprocessing": (
        "grapheme-to-phoneme through misaki (espeak-ng fallback for out-of-dictionary words when it "
        "binds on the host), a 128-d style vector read from the voice pack by phoneme count; inside the "
        f"library a line above {MAX_SEGMENT_PHONEMES} phonemes is split into several chunks and a line that "
        "phonemises to nothing is skipped"
    ),
}


def text_lines(text: str) -> list[str]:
    """The lines the library synthesises, split as it splits them (``SPLIT_PATTERN`` on the stripped text).

    A line's position in this list is the ``line`` index reported for every chunk of audio it produced."""
    return re.split(SPLIT_PATTERN, text.strip())


def decode_byod_text(payload: bytes, name: str) -> str:
    """Decode an uploaded BYOD text file (BOM dropped, line endings normalised to ``\\n``, outer whitespace
    stripped); refuse a file that is not UTF-8 with a message naming the remedy."""
    try:
        return payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{name} is not UTF-8 text (undecodable byte at position {exc.start}): save it as UTF-8 "
            "(in most editors: Save as -> Encoding: UTF-8) and run this cell again"
        ) from None


def _check_inputs(text: Any, voice: Any, speed: Any, voices: Sequence[str], lang_code: str) -> None:
    """Raise TypeError/ValueError naming the first violated ceiling; return nothing."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text exceeds MAX_TEXT_CHARS={MAX_TEXT_CHARS}: {len(text)}")
    if not isinstance(voice, str) or voice not in voices:
        raise ValueError(f"voice must be one of the {len(voices)} manifest voices, got {voice!r}")
    if voice[0] != lang_code:
        raise ValueError(f"voice {voice!r} is not a lang_code={lang_code!r} voice")
    if isinstance(speed, bool) or not isinstance(speed, int | float):
        raise TypeError("speed must be a number")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between MIN_SPEED={MIN_SPEED} and MAX_SPEED={MAX_SPEED}")


def validate_inputs(
    text: str,
    voice: str = DEFAULT_VOICE,
    *,
    speed: float = 1.0,
    voices: Sequence[str],
    lang_code: str = DEFAULT_LANG_CODE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    ``voices`` is the instance's authoritative voice inventory — ``pipe.voices``, which
    ``list_voices`` derives from the digest-verified manifest — and is required because voice
    membership cannot be checked without it. Rejection is reported by raising exactly as
    ``synthesize`` would: both route through ``_check_inputs``.
    """
    _check_inputs(text, voice, speed, voices, lang_code)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry: synthesize takes one text per call")
    lines = [line for line in text_lines(text) if line.strip()]
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "text-0",
                "chars": len(text),
                "segments": len(lines) or 1,
            }
        ],
        "voice": voice,
        "speed": float(speed),
        "lang_code": lang_code,
        "voice_inventory": len(voices),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], references: Sequence[Any] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    Speech quality has no intrinsic metric and the repository ships no metric helper, so the
    verdict is always ``not-measurable`` (EVAL9). ``references`` exists for interface parity with
    the fleet's other pipelines and is recorded in ``reason`` rather than scored: a reference
    recording cannot be compared to a synthesised waveform sample-by-sample, and both of the real
    judges (listener MOS, ASR word error rate) live outside this repository.
    """
    supplied = references is not None
    return {
        "task": f"text-to-speech synthesis ({SAMPLE_RATE} Hz mono float32, named synthetic voice pack)",
        "score_semantics": (
            "the output is a waveform, not a prediction: duration, peak amplitude and the phoneme "
            "string are run-level facts, not quality scores, and none of them bounds naturalness "
            "or intelligibility"
        ),
        "sample_kind": sample_kind,
        "n_segments": len(result.get("segments", [])),
        "duration_s": float(result.get("duration_s", 0.0)),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "speech quality has no ground truth in this repository and no metric helper is shipped"
            + (
                "; references were supplied but no metric helper exists to score them here"
                if supplied
                else "; the evaluated sample has no reference recording"
            )
        ),
        "needs": (
            "an external judge: Mean Opinion Score ratings from human listeners for naturalness, or "
            "an independent speech recogniser to re-transcribe the waveform and compute word error "
            "rate against the input text for intelligibility — over a reference sentence set, with "
            "the judge named"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class KokoroTTSPipeline:
    """``_runner(text, voice, speed)`` yields ``(graphemes, phonemes, audio_1d[, line])`` per chunk.

    ``line`` is the index into ``text_lines(text)`` of the line the chunk came from; a runner that omits it
    (a 3-tuple) leaves line coverage unknown (``skipped_lines`` is then ``None``)."""

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
        if lang_code not in LANG_CODES:
            raise ValueError(f"lang_code must be one of LANG_CODES {LANG_CODES}, got {lang_code!r}")
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no verified snapshot at {root} and no Hub path is offered for pickle checkpoints; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        converted = (root / CONVERTED_WEIGHTS_FILE, root / CONVERTED_VOICES_FILE)
        if not all(file.is_file() for file in converted):
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            convert_checkpoint(root)
        manifest = _read_manifest(root)
        config_path = _verify_entry(root, manifest, CONFIG_FILE)
        # Validate language and verify the served files before importing model libraries.
        verify_converted(root)
        import torch
        from kokoro import KPipeline
        from safetensors.torch import load_file

        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        kmodel = _build_model(config)
        kmodel.load_state_dict(load_file(str(converted[0]), device="cpu"), strict=True)
        kmodel = kmodel.to(resolved_device).eval()
        packs = load_file(str(converted[1]), device="cpu")
        voices = list_voices(manifest)
        if tuple(sorted(packs)) != voices:
            raise ValueError(
                f"{CONVERTED_VOICES_FILE} holds {len(packs)} voices, the manifest lists {len(voices)}"
            )
        kpipeline = KPipeline(lang_code=lang_code, repo_id=MODEL_ID, model=kmodel, device=resolved_device)
        fallback = getattr(getattr(kpipeline, "g2p", None), "fallback", None)

        def runner(text: str, voice: str, speed: float) -> Any:
            with torch.inference_mode():
                # A tensor voice is returned unchanged by KPipeline.load_voice: no .pt file is opened.
                for result in kpipeline(text, voice=packs[voice], speed=speed, split_pattern=SPLIT_PATTERN):
                    audio = result.audio
                    array = audio.cpu().numpy() if audio is not None else None
                    yield result.graphemes, result.phonemes, array, result.text_index

        return cls(
            runner, voices, lang_code, resolved_device, "local-snapshot", root,
            fallback is not None if lang_code in "ab" else None,
        )

    def _validate(self, text: Any, voice: Any, speed: Any) -> None:
        _check_inputs(text, voice, speed, self.voices, self.lang_code)

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE, *, speed: float = 1.0) -> dict[str, Any]:
        """Synthesise ``text`` with a named voice pack; ``audio`` is a 1-D float32 array at 24 kHz."""
        self._validate(text, voice, speed)
        chunks: list[np.ndarray] = []
        segments: list[dict[str, Any]] = []
        line_known = True
        for item in self._runner(text, voice, float(speed)):
            graphemes, phonemes, audio = item[:3]
            line = item[3] if len(item) > 3 else None
            if audio is None:
                continue
            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            chunks.append(array)
            line_known = line_known and line is not None
            segments.append({"graphemes": str(graphemes), "phonemes": str(phonemes), "line": line})
        if not chunks:
            raise RuntimeError(
                "no audio produced: every line phonemised to nothing (words outside the dictionary are "
                "dropped when the espeak-ng fallback is not available); use dictionary words"
            )
        # A long line yields several chunks; a line that phonemises to nothing yields none: report it.
        lines = text_lines(text)
        voiced = {segment["line"] for segment in segments}
        skipped = None
        if line_known:
            skipped = [
                {"line": i, "text": part.strip()[:80]}
                for i, part in enumerate(lines)
                if part.strip() and i not in voiced
            ]
        wave = np.concatenate(chunks)
        return {
            "audio": wave,
            "sample_rate": SAMPLE_RATE,
            "num_samples": int(wave.shape[0]),
            "duration_s": float(wave.shape[0] / SAMPLE_RATE),
            "peak_amplitude": float(np.abs(wave).max()),
            "segments": segments,
            "lines": sum(1 for part in lines if part.strip()),
            "skipped_lines": skipped,
            "voice": voice,
            "lang_code": self.lang_code,
            "speed": float(speed),
            "espeak_fallback": self.espeak_fallback,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
