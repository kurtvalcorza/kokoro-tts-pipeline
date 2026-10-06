"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/kokoro_tts_colab.ipynb`
(review PR #8: KTT-B1, KTT-M1, KTT-m1..m4).

The notebook's own cells are executed from the committed JSON in a namespace of the carried module's names and
inert stand-ins (an injected runner instead of the model, a fake `soundfile`, a fake `google.colab`). Nothing here
loads the pinned checkpoint, and nothing needs torch: CI installs only numpy.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from kokoro_tts_pipeline import KokoroTTSPipeline, decode_byod_text, text_lines
from kokoro_tts_pipeline import pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "kokoro_tts_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
VOICES = ("af_heart", "af_bella", "bf_emma")
SPACY_WHEEL = "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_kokoro_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _kernel_cells() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]


def _namespace() -> dict:
    """The kernel globals the carried module cell defines (public and private names)."""
    return {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}


# --- KTT-B1 / KTT-M1: isolated uv environment with a managed Python 3.12.12 -----------------------------------------


def test_exactly_two_kernel_cells_build_and_route_to_an_isolated_python_3_12():
    kernel = _kernel_cells()
    assert len(kernel) == 2
    install, router = kernel
    for needed in ("MANAGED_PYTHON = '3.12.12'", '"--managed-python"', '"--require-hashes"', '"--only-binary", ":all:"', "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install, needed
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    # Nothing outside the two kernel cells runs in the kernel; the runtime cell's pip guard only runs when the kernel
    # itself is the runtime (DIMER_NOTEBOOK_CI_PREINSTALLED=1 skips it, and the isolated worker always sets that).
    runtime = _code_after("### Record the runtime")
    assert "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in runtime
    assert "env = dict(os.environ, MPLBACKEND=\"Agg\", PYTHONUNBUFFERED=\"1\", DIMER_NOTEBOOK_CI_PREINSTALLED=\"1\"" in router


def test_carried_lock_is_the_committed_lock_and_pins_kokoro_misaki_and_the_spacy_model():
    install = _kernel_cells()[0]
    carried = re.search(r"^LOCK_TEXT = r'''(.*?)'''$", install, re.M | re.S).group(1)
    committed = LOCK.read_text(encoding="utf-8")
    assert carried == committed
    build = _load_tool("build_notebook")
    pins = build._pins(ROOT)
    build.check_lock(pins, committed)  # every pyproject pin at its version, every entry hashed
    locked = build.lock_packages(committed)
    assert locked["kokoro"] == "0.9.4" and locked["misaki"] == "0.9.4" and locked["torch"] == "2.14.0"
    assert locked["num2words"] == "0.5.6" and "docopt" not in locked  # wheel-only resolution, stated in the prerequisites
    block = committed.split(SPACY_WHEEL, 1)[1].split("\n")[1]
    assert SPACY_WHEEL in committed and block.strip().startswith("--hash=sha256:")
    assert "espeakng-loader" in locked and "phonemizer-fork" in locked


def test_no_restart_or_in_kernel_install_text_and_the_runtime_prerequisite_names_python_3_13():
    md = _markdown()
    for stale in ("Restart the runtime", "restart the runtime", "Core dependencies changed", "installs the pinned dependencies", "Google Colab or Jupyter, Python 3.12", "is downloading the spaCy"):
        assert stale not in md, stale
    assert "**Linux x86_64** runtime" in md and "Colab's kernel runs Python 3.13" in md
    assert "the kernel's own Python version does not matter" in md
    assert "SyntaxWarning: invalid escape sequence" in md  # num2words 0.5.6, an expected warning


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: a library that calls importlib.util.find_spec("google.colab") raises on a spec-less stub."""
    router = _kernel_cells()[1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


# --- KTT-m2: long lines are chunked, empty lines are reported, prose matches ------------------------------------------


def _chunking_runner(text, voice_path, speed):
    """Mimics kokoro 0.9.4 KPipeline.__call__: lines split on SPLIT_PATTERN; a line above 200 characters yields two
    chunks (the real library splits above 510 phonemes); a line of only 'XQZV'-style words yields nothing."""
    for index, line in enumerate(re.split(r"\n+", text.strip())):
        if not line.strip() or set(line.split()) <= {"XQZV", "BRRT"}:
            continue
        pieces = [line[: len(line) // 2], line[len(line) // 2 :]] if len(line) > 200 else [line]
        for piece in pieces:
            yield piece, f"ph:{piece[:10]}", np.full(240, 0.1, dtype=np.float32), index


def _pipe(runner=_chunking_runner) -> KokoroTTSPipeline:
    return KokoroTTSPipeline(runner, VOICES, "a", "cpu", "injected", Path("w"))


def test_synthesize_reports_chunks_per_line_and_skipped_lines():
    long_line = "word " * 120
    result = _pipe().synthesize(long_line.strip() + "\nXQZV BRRT\nShort line.")
    assert [s["line"] for s in result["segments"]] == [0, 0, 2]
    assert result["lines"] == 3
    assert result["skipped_lines"] == [{"line": 1, "text": "XQZV BRRT"}]


def test_a_three_tuple_runner_leaves_line_coverage_unknown():
    def runner(text, voice_path, speed):
        yield "hi", "haɪ", np.full(10, 0.1, dtype=np.float32)

    result = _pipe(runner).synthesize("hi")
    assert result["segments"][0]["line"] is None and result["skipped_lines"] is None


def test_text_lines_splits_like_the_library():
    assert text_lines("  a\n\n\nb\n  \nc  ") == ["a", "b", "  ", "c"]
    assert pipeline_module.SPLIT_PATTERN == r"\n+" and pipeline_module.MAX_SEGMENT_PHONEMES == 510


class _FakeSoundfile:
    __version__ = "fake"

    @staticmethod
    def write(path, audio, rate, subtype):
        Path(path).write_bytes(b"RIFF" + np.asarray(audio, dtype=np.float32).tobytes())


def _section6(monkeypatch, tmp_path, text: str, **extra) -> dict:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    ns = _namespace()
    ns.update(text=text, VOICE="af_heart", SPEED=1.0, SEED=0, pipe=_pipe(), hashlib=__import__("hashlib"), soundfile=_FakeSoundfile, torch=types.SimpleNamespace(manual_seed=lambda seed: None), **extra)
    exec(compile(_code_after("## 6. Synthesise, write the WAV"), "<section 6>", "exec"), ns)
    return ns


def test_section6_accepts_a_600_character_single_line(monkeypatch, tmp_path, capsys):
    paragraph = ("The quick brown fox jumps over the lazy dog near the river bank. " * 10)[:600]
    ns = _section6(monkeypatch, tmp_path, paragraph)
    assert all(ns["checks"].values())
    assert len(ns["result"]["segments"]) == 2 and ns["result"]["lines"] == 1
    out = capsys.readouterr().out
    assert "'chunks_per_line': {0: 2}" in out and "WARNING" not in out


def test_section6_warns_and_names_a_line_that_phonemises_to_nothing(monkeypatch, tmp_path, capsys):
    ns = _section6(monkeypatch, tmp_path, "The quick brown fox.\nXQZV BRRT")
    assert all(ns["checks"].values())
    out = capsys.readouterr().out
    assert "WARNING: 1 line(s) produced no audio" in out and "{'line': 1, 'text': 'XQZV BRRT'}" in out


def test_section6_shows_an_html_audio_player_through_display(monkeypatch, tmp_path):
    shown = []
    _section6(monkeypatch, tmp_path, "The quick brown fox.", display=shown.append)
    assert len(shown) == 1 and shown[0]._repr_html_().startswith('<audio controls src="data:audio/wav;base64,')


def test_prose_describes_chunking_not_truncation():
    md = _markdown()
    assert "truncates any single segment" not in md and "are truncated by the library" not in md
    assert "a line above 510 phonemes is split" in md and "phonemises to nothing" in md
    assert "one segment per line" not in md


# --- KTT-m3: BYOD by path on any runtime; named errors ----------------------------------------------------------------


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)


def _section4(monkeypatch, tmp_path, byod_path: str = "") -> dict:
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. Author the sample text or optional BYOD")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    assert "BYOD_PATH = ''" in source
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def test_byod_path_works_without_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    path = tmp_path / "mine.txt"
    path.write_bytes("﻿Hello there.\r\nSecond line.\r\n".encode())  # BOM and Windows line endings
    ns = _section4(monkeypatch, tmp_path, str(path))
    assert ns["text"] == "Hello there.\nSecond line." and ns["sample_name"] == "mine.txt"
    assert ns["sample_kind"] == "BYOD file (BYOD_PATH)"


def test_byod_errors_name_the_fix(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    latin = tmp_path / "latin.txt"
    latin.write_bytes("café crème".encode("latin-1"))
    with pytest.raises(ValueError, match="is not UTF-8 text .* save it as UTF-8"):
        _section4(monkeypatch, tmp_path, str(latin))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'missing.txt' is not a file"):
        _section4(monkeypatch, tmp_path, "missing.txt")
    with pytest.raises(RuntimeError, match="no Colab upload dialog, so set BYOD_PATH"):
        _section4(monkeypatch, tmp_path)


def test_cancelled_and_multiple_uploads_are_named_and_one_upload_works(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}, {"a.txt": b"a", "b.txt": b"b"}, {"mine.txt": "Hej då.".encode()}])
    with pytest.raises(ValueError, match=r"^no file uploaded; rerun this cell and choose one UTF-8 \.txt file$"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="2 files uploaded"):
        _section4(monkeypatch, tmp_path)
    ns = _section4(monkeypatch, tmp_path)
    assert ns["text"] == "Hej då." and ns["sample_kind"] == "BYOD upload"


def test_decode_byod_text_strips_bom_and_whitespace():
    assert decode_byod_text("﻿  hi \n".encode(), "x.txt") == "hi"
    with pytest.raises(ValueError, match=r"x\.txt is not UTF-8 text \(undecodable byte at position 0\)"):
        decode_byod_text(b"\xff\xfe", "x.txt")


# --- KTT-m1 / KTT-m4: records and declarations ------------------------------------------------------------------------


def test_spec_is_declared_as_2_2_everywhere():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.2"
    assert "DIMER Notebook Specification 2.2" in _source(_cells()[0])
    for name in ("docs/release-verification.md", "tutorials/README.md", "README.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "spec `1.1`" not in text and "Specification 1.1" not in text and "Specification 2.0" not in text, name


def test_no_document_calls_run_all_verified_for_an_unrecorded_blob(tmp_path):
    validator = _load_tool("validate_release_assets")
    validator.validate_run_all_claims()  # the committed documents pass
    assert "verified — clean-runtime" not in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    copy = tmp_path / "repo"
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md", "tutorials/kokoro_tts_colab.ipynb"):
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, copy / name)
    registry = copy / "tutorials" / "README.md"
    registry.write_text(registry.read_text(encoding="utf-8") + "\n| x | verified — clean-runtime `Run all` execution recorded |\n", encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="KTT-m1"):
        validator.validate_run_all_claims(copy)
    blob = validator.notebook_blob(copy / "tutorials" / "kokoro_tts_colab.ipynb")
    registry.write_text(registry.read_text(encoding="utf-8") + f"\nblob `{blob}`\n", encoding="utf-8")
    record = copy / "docs" / "release-verification.md"
    record.write_text(record.read_text(encoding="utf-8") + f"\n| run | `{blob}` | PASS |\n", encoding="utf-8")
    validator.validate_run_all_claims(copy)


def test_the_2026_09_13_run_is_not_recorded_as_a_run_all_of_this_notebook():
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "correctly halts" not in record
    assert "not a `Run all`" in record
    for name in ("STATUS.md", "README.md"):
        assert "no `run all` of the current notebook is recorded" in (ROOT / name).read_text(encoding="utf-8").lower(), name
