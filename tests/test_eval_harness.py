"""The eval harness is code too: check the case file is valid and that scoring behaves, with no network."""
import importlib.util
import json
from pathlib import Path

from helpers import ScriptedJudge, make_result

EVALS = Path(__file__).resolve().parent.parent / "evals"
spec = importlib.util.spec_from_file_location("run_judge_eval", EVALS / "run_judge_eval.py")
run_judge_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_judge_eval)


def test_case_file_is_valid_and_has_unique_ids():
    cases = run_judge_eval.load_cases()
    assert len(cases) >= 20
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["instructions"] and case["response"]
        assert set(case["expected"]) == {"adherence", "grounded", "toxic"}


def test_dry_run_exits_cleanly(capsys):
    assert run_judge_eval.main(["--dry-run"]) == 0
    assert "valid" in capsys.readouterr().out


class OracleJudge:
    """Answers exactly what the labels say, so every metric must come out perfect."""

    def __init__(self, cases):
        self.by_response = {c["response"]: c for c in cases}

    def evaluate(self, payload):
        case = self.by_response[payload["generated_text"]]
        return make_result(
            adherence=[(i, 0.95 if ok else 0.05) for i, ok in zip(case["instructions"], case["expected"]["adherence"])],
            groundedness=[("claim", 0.95 if case["expected"]["grounded"] else 0.05)],
            toxicity=[("span", 0.9)] if case["expected"]["toxic"] else [],
        )


def test_scoring_is_perfect_for_an_oracle_judge():
    cases = run_judge_eval.load_cases()
    metrics, mismatches = run_judge_eval.score_cases(cases, OracleJudge(cases))
    assert mismatches == [] and metrics["judge_errors"] == 0
    assert metrics["adherence_accuracy"] == 1.0
    assert metrics["groundedness_accuracy"] == 1.0
    assert metrics["toxicity_accuracy"] == metrics["toxicity_precision"] == metrics["toxicity_recall"] == 1.0


def test_scoring_reports_mismatches_and_judge_errors():
    cases = run_judge_eval.load_cases()[:2]
    # Always says: everything followed, grounded, not toxic -> wrong wherever the labels say otherwise.
    lenient = ScriptedJudge(*[make_result(adherence=[(i, 0.99) for i in c["instructions"]]) for c in cases])
    metrics, mismatches = run_judge_eval.score_cases(cases, lenient)
    # cases[:2] = orion-pass (labels: all followed, grounded) and orion-price (labels: price rule violated, ungrounded)
    assert metrics["adherence_accuracy"] == 0.75   # 3 of 4 instructions right; misses the price violation
    assert metrics["groundedness_accuracy"] == 0.5  # misses the unsupported $999 claim
    assert metrics["toxicity_accuracy"] == 1.0
    assert any("orion-price" in m and "adherence" in m for m in mismatches)
    assert any("orion-price" in m and "grounded" in m for m in mismatches)

    broken = ScriptedJudge(RuntimeError("boom"))
    metrics, _ = run_judge_eval.score_cases(cases, broken)
    assert metrics["judge_errors"] == 2 and "boom" in metrics["errors"][0]
