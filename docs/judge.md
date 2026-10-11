# The judge

The judge decides whether a response needs another pass. It is a single call to a (small) Together AI model that returns one JSON report. All of it lives in [`src/reprompting_pipeline/judge.py`](../src/reprompting_pipeline/judge.py).

## The report

The pipeline needs three verdicts per response (instruction adherence, groundedness, toxicity). The model returns `{"instruction", "follow_probability", "explanation"}`.

| Section | One item per | `follow_probability` means | Fails when |
|---|---|---|---|
| `instruction_adherence` | user instruction | P(the response follows it) | `p < 0.5` |
| `groundedness` | checkable factual claim in the response | P(the context supports the claim) | `p < 0.5` |
| `toxicity` | toxic span only (empty when clean) | P(the span is toxic) | `p > 0.25` |

`parse_judge_output` turns the raw text into a `JudgeResult` whose sections look like the rest of the pipeline expects:

```python
{"score": 0.5, "instructions_list": [
    {"instruction": "...", "label": False, "follow_probability": 0.1, "explanation": "..."}]}
```

## The prompt

The system prompt defines the three sections and the rules for each, followed by two worked examples as chat turns (one with a violation and an unsupported claim, one toxic). The model is also asked for JSON via Together's `response_format` with a JSON schema (`JUDGE_SCHEMA`), when supported. The prompt and schema are module-level constants, so they are easy to edit and diff.

## Using a different judge

Anything with `evaluate(payload) -> JudgeResult` works:

```python
from reprompting_pipeline import run_reprompting_pipeline
from reprompting_pipeline.judge import TogetherJudge

# Same prompt, different transport: inject your own completion function (messages -> text).
judge = TogetherJudge(complete_fn=my_openai_or_local_call)
run_reprompting_pipeline(llm_fn=my_llm, user_query="...", judge=judge)
```

`payload` has the keys `context`, `user_query`, `instructions`, `generated_text`, `system_prompt`.

## Measuring and tuning

`evals/judge_cases.jsonl` holds hand-labeled cases (the labels are objective: counts, quotes, or a lookup in the case's context). Score a judge model against them:

```bash
TOGETHER_API_KEY=... python evals/run_judge_eval.py --model <together/model-id> --out results.json
```

It reports per-instruction adherence accuracy, case-level groundedness accuracy, and toxicity accuracy/precision/recall, and lists disagreement, so you can see *which* cases a judge fails on.
