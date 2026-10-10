"""Measure how well the judge agrees with hand-labeled cases.

    python evals/run_judge_eval.py --dry-run             # validate evals/judge_cases.jsonl only (no network)
    TOGETHER_API_KEY=... python evals/run_judge_eval.py  # score the default judge model
    TOGETHER_API_KEY=... python evals/run_judge_eval.py --model <together/model-id> --out results.json

What is scored (all labels are objective: counts, quotes, or a lookup in the case's context):
    adherence : each instruction's pass/fail            -> per-instruction accuracy
    grounded  : "does the response contain any claim the context does not support?"
                (only cases that have a context)       -> case-level accuracy
    toxic     : "is any span flagged above TOXICITY_THRESHOLD?" -> accuracy, precision, recall

Use it to compare judge models and to tune the prompt in src/reprompting_pipeline/judge.py. With few
cases the numbers are rough; add cases to judge_cases.jsonl for the failure modes you care about.
"""
import argparse
import json
import sys
from pathlib import Path

CASES_PATH = Path(__file__).with_name("judge_cases.jsonl")


def load_cases(path=CASES_PATH):
    cases = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        expected = case["expected"]
        if len(expected["adherence"]) != len(case["instructions"]):
            raise ValueError(f"line {line_no} ({case.get('id')}): adherence labels != instructions")
        cases.append(case)
    return cases


def score_cases(cases, judge, toxicity_threshold=0.25):
    """Run `judge` over `cases`; return (metrics dict, list of mismatch descriptions)."""
    adherence_hits = adherence_total = 0
    grounded_hits = grounded_total = 0
    tp = fp = fn = tn = 0
    errors, mismatches = [], []

    for case in cases:
        payload = {
            "context": case["context"],
            "user_query": case["user_query"],
            "instructions": case["instructions"],
            "generated_text": case["response"],
            "system_prompt": "",
        }
        try:
            result = judge.evaluate(payload)
        except Exception as e:  # a judge that cannot produce a report is a failure to record, not to hide
            errors.append(f"{case['id']}: {type(e).__name__}: {e}")
            continue

        # adherence, per instruction
        predicted = {i["instruction"]: i["label"] for i in result.instruction_adherence["instructions_list"]}
        for instruction, want in zip(case["instructions"], case["expected"]["adherence"]):
            adherence_total += 1
            got = predicted.get(instruction)
            if got is want:
                adherence_hits += 1
            else:
                mismatches.append(f"{case['id']}: adherence {instruction!r} expected {want}, got {got}")

        # groundedness, per case (skipped when there is no context to ground against)
        if case["context"].strip():
            grounded_total += 1
            got = all(c["label"] for c in result.groundedness["instructions_list"])
            if got is case["expected"]["grounded"]:
                grounded_hits += 1
            else:
                mismatches.append(f"{case['id']}: grounded expected {case['expected']['grounded']}, got {got}")

        # toxicity, per case
        got = any(i["follow_probability"] > toxicity_threshold for i in result.toxicity["instructions_list"])
        want = case["expected"]["toxic"]
        tp += got and want
        fp += got and not want
        fn += (not got) and want
        tn += (not got) and not want
        if got is not want:
            mismatches.append(f"{case['id']}: toxic expected {want}, got {got}")

    def ratio(num, den):
        return round(num / den, 3) if den else None

    metrics = {
        "cases": len(cases),
        "judge_errors": len(errors),
        "adherence_accuracy": ratio(adherence_hits, adherence_total),
        "adherence_n": adherence_total,
        "groundedness_accuracy": ratio(grounded_hits, grounded_total),
        "groundedness_n": grounded_total,
        "toxicity_accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "toxicity_precision": ratio(tp, tp + fp),
        "toxicity_recall": ratio(tp, tp + fn),
        "errors": errors,
    }
    return metrics, mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--model", help="Together model id for the judge (default: RepromptingConfig default)")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--out", help="write metrics + mismatches as JSON to this path")
    parser.add_argument("--dry-run", action="store_true", help="validate the cases file and exit")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if args.dry_run:
        print(f"OK: {len(cases)} cases valid")
        return 0

    from reprompting_pipeline.config import RepromptingConfig
    from reprompting_pipeline.judge import TogetherJudge
    from reprompting_pipeline.utils import TOXICITY_THRESHOLD

    config = RepromptingConfig(**({"judge_model": args.model} if args.model else {}))
    print(f"Judge model: {config.judge_model}  |  cases: {len(cases)}")
    metrics, mismatches = score_cases(cases, TogetherJudge(config), toxicity_threshold=TOXICITY_THRESHOLD)

    for key, value in metrics.items():
        if key != "errors":
            print(f"{key:>22}: {value}")
    for line in metrics["errors"] + mismatches:
        print("  -", line)
    if args.out:
        Path(args.out).write_text(json.dumps({"model": config.judge_model, **metrics, "mismatches": mismatches}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
