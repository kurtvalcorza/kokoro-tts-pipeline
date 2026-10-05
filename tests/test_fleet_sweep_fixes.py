"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer, SWP-B BYOD).

Every test needs only CI's dependencies (pytest, numpy). The notebook's own cell sources are executed with
stand-ins; no model, no network and no torch are needed. The record is
``docs/reviews/2026-10-05-fleet-sweep/kokoro_tts_colab_Fixes.md``.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "kokoro_tts_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------------


def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift
    # spaCy installs misaki's en_core_web_sm with `python -m pip`; a uv environment has no pip unless it is locked.
    assert re.search(r"^pip==\S+ \\$", lock_text, re.M)


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


def test_swp_r_learner_text_no_longer_describes_an_in_kernel_install(notebook):
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "Core dependencies changed" not in markdown
    assert "pinned `torch==2.14.0` install" not in markdown
    assert "Run all completes in one pass" in markdown


# --- SWP-G: the guided layer and infrastructure labelling ---------------------------------------------------------


def test_swp_g_guided_layer_is_present(notebook):
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= 4
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 5
    assert markdown.count("**What to notice:**") >= 2


def test_swp_g_checkpoint_answers_agree_with_the_recorded_run(notebook):
    """The worked answers quote the retained 2026-09-13 run, not invented numbers."""
    record = ROOT / "docs" / "verification" / "2026-09-13" / "outputs"
    result = json.loads((record / "kokoro_tts_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((record / "kokoro_tts_input_manifest.json").read_text(encoding="utf-8"))
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert f"{result['audio']['num_samples']:,} samples" in markdown
    assert f"{result['audio']['duration_s']} s" in markdown
    assert f"peak amplitude {result['audio']['peak_amplitude']:.3f}" in markdown
    assert result["segments"][0]["phonemes"] in markdown
    assert result["wav"]["sha256"][:16] in markdown
    assert manifest["findings"][0]["message"] in markdown
    assert f"{manifest['inputs'][0]['chars']} characters" in markdown


def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(notebook):
    cells = _code_cells(notebook)
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


def test_swp_g_no_template_placeholders_leak(notebook):
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


# --- SWP-B: BYOD path, guarded upload, named refusals --------------------------------------------------------------


def _byod_namespace(notebook: dict, monkeypatch, *, path: str = "", upload=None, colab: bool = False) -> dict:
    source = _cell(notebook, "BYOD_PATH = ''")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    if colab:
        google = types.ModuleType("google")
        google.__path__ = []
        colab_mod = types.ModuleType("google.colab")
        files = types.ModuleType("google.colab.files")
        files.upload = upload
        colab_mod.files = files
        google.colab = colab_mod
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
        monkeypatch.setitem(sys.modules, "google.colab.files", files)
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter
    namespace = {"__name__": "__main__", "Path": Path, "MAX_TEXT_CHARS": 2000}
    exec(compile(source, "<section 4>", "exec"), namespace)
    return namespace


def test_swp_b_byod_path_works_outside_colab(notebook, tmp_path, monkeypatch, capsys):
    text_file = tmp_path / "lines.txt"
    text_file.write_text("﻿Hello there.\nSecond line.\n", encoding="utf-8")
    ns = _byod_namespace(notebook, monkeypatch, path=str(text_file))
    assert ns["sample_name"] == "lines.txt" and ns["sample_kind"] == "BYOD file"
    assert ns["text"] == "Hello there.\nSecond line."
    assert "'lines': 2" in capsys.readouterr().out


def test_swp_b_refusals_name_the_file_and_the_rule(notebook, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.txt.* is not a file"):
        _byod_namespace(notebook, monkeypatch, path=str(tmp_path / "missing.txt"))
    bad = tmp_path / "latin1.txt"
    bad.write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ValueError, match=r"latin1\.txt: not UTF-8 text"):
        _byod_namespace(notebook, monkeypatch, path=str(bad))
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"empty\.txt: the file holds no text"):
        _byod_namespace(notebook, monkeypatch, path=str(empty))
    long = tmp_path / "long.txt"
    long.write_text("a" * 2001, encoding="utf-8")
    with pytest.raises(ValueError, match=r"long\.txt: 2001 characters exceeds MAX_TEXT_CHARS = 2000"):
        _byod_namespace(notebook, monkeypatch, path=str(long))


def test_swp_b_no_path_outside_colab_is_explained(notebook, monkeypatch):
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _byod_namespace(notebook, monkeypatch)


def test_swp_b_cancelled_or_multiple_uploads_are_refused_and_one_upload_works(notebook, monkeypatch):
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        _byod_namespace(notebook, monkeypatch, colab=True, upload=lambda: {})
    with pytest.raises(RuntimeError, match="got 2"):
        _byod_namespace(notebook, monkeypatch, colab=True, upload=lambda: {"a.txt": b"a", "b.txt": b"b"})
    ns = _byod_namespace(notebook, monkeypatch, colab=True, upload=lambda: {"mine.txt": b"Good morning."})
    assert (ns["sample_name"], ns["text"], ns["sample_kind"]) == ("mine.txt", "Good morning.", "BYOD upload")


def test_swp_b_default_path_is_the_synthetic_pangram(notebook, monkeypatch):
    source = _cell(notebook, "BYOD_PATH = ''")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    namespace = {"__name__": "__main__", "Path": Path, "MAX_TEXT_CHARS": 2000}
    exec(compile(source, "<section 4>", "exec"), namespace)
    assert namespace["text"] == "The quick brown fox jumps over the lazy dog."
    assert len(namespace["text"]) == 44


def test_swp_b_inline_player_renders_without_ipython(notebook, tmp_path):
    """Section 6's player works in the isolated worker (no IPython there): an HTML5 element carrying the WAV."""
    source = _cell(notebook, "class WavPlayer:")
    player_src = source[source.index("class WavPlayer:") : source.index("try:\n    display(WavPlayer(wav_path))")]
    namespace = {"Path": Path}
    exec(compile(player_src, "<section 6 player>", "exec"), namespace)
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    html = namespace["WavPlayer"](str(wav))._repr_html_()
    assert html.startswith('<audio controls src="data:audio/wav;base64,UklGRjAwMDBXQVZF')
