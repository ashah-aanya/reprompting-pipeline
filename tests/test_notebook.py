"""The demo notebook is code too: check it is valid and that its cells really run, with no network.

`together` and `google.colab` are replaced with stubs: the fake Together client answers the notebook's
LLM calls with plain text and the judge's calls (recognized by the json_schema response_format) with a
report that passes every instruction.
"""
import json
import re
import sys
import types
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "reprompting_demo.ipynb"


def load_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return nb["cells"]


def runnable_source(cell):
    """Code cell source without Colab/IPython-only lines (%magics, !shell commands)."""
    lines = "".join(cell["source"]).splitlines()
    return "\n".join(l for l in lines if not l.lstrip().startswith(("%", "!")))


def test_notebook_has_no_outputs_and_no_aimon_references():
    text = NOTEBOOK.read_text()
    assert "aimon" not in text.lower()
    for cell in load_cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_every_code_cell_compiles():
    for i, cell in enumerate(load_cells()):
        if cell["cell_type"] == "code":
            compile(runnable_source(cell), f"<cell {i}>", "exec")


class FakeTogether:
    """Stand-in for together.Together."""

    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kwargs):
        if "response_format" in kwargs:  # a judge call: pass every numbered instruction
            last = kwargs["messages"][-1]["content"]
            block = last.split("INSTRUCTIONS:\n", 1)[1].split("\n\nRESPONSE:", 1)[0]
            items = [
                {"instruction": re.sub(r"^\d+\. ", "", line), "follow_probability": 0.95, "explanation": "ok"}
                for line in block.splitlines() if line.strip() and line != "(none)"
            ]
            content = json.dumps({"instruction_adherence": items, "groundedness": [], "toxicity": []})
        else:  # the notebook's own llm_fn call
            content = "A friendly, accurate answer."
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_notebook_runs_end_to_end_against_stubs(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "together", types.SimpleNamespace(Together=FakeTogether))
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "fake-key")
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    namespace = {"__name__": "notebook"}
    for i, cell in enumerate(load_cells()):
        if cell["cell_type"] == "code":
            exec(compile(runnable_source(cell), f"<cell {i}>", "exec"), namespace)

    result = namespace["result"]
    assert result["best_response"] == "A friendly, accurate answer."
    assert result["summary"] == "1 iteration, 0 failed instructions remaining"
    assert len(result["telemetry"]) == 1
    printed = capsys.readouterr().out
    assert "=== SUMMARY ===" in printed and "=== BEST RESPONSE ===" in printed
    # the notebook feeds the user's own example through the pipeline
    assert len(namespace["user_instructions"]) >= 3
