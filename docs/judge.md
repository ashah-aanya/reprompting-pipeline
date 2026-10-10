# The judge

The judge decides whether a response needs another pass. It is a single call to a (small) Together AI model that returns one JSON report. All of it lives in [`src/reprompting_pipeline/judge.py`](../src/reprompting_pipeline/judge.py).

## Why one structured call

The pipeline needs three verdicts per response (instruction adherence, groundedness, toxicity). One combined call is cheaper and faster than three, and the model sees the whole response once. The cost is that a small model may drop a field or return malformed JSON, so the output is never trusted blindly: it is parsed and validated, and unusable output triggers a retry.

## The report

The model returns an object with three lists. Each item is `{"instruction", "follow_probability", "explanation"}`.

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

Validation rules (each covered in `tests/test_judge_parsing.py`):

- Code fences and surrounding prose around the JSON are stripped.
- Probabilities are clamped to [0, 1]; numeric strings are accepted; items without a usable number are skipped.
- **`label` is always derived from `follow_probability`**, so the two can never disagree.
- Paraphrased instructions are snapped back to the user's exact text (the re-prompt builder matches instructions by exact string).
- No instructions given: adherence is forced empty. No context given: groundedness is forced empty.
- Missing groundedness/toxicity sections default to "nothing found" with a warning. A missing adherence section when instructions *were* given, non-JSON, or an empty reply raises `JudgeParseError`, and the pipeline retries the call.

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

It reports per-instruction adherence accuracy, case-level groundedness accuracy, and toxicity accuracy/precision/recall, and lists every disagreement, so you can see *which* cases a prompt change fixed or broke.

Caveats:

- The case set is small and toxicity cases are few, so treat the numbers as a rough signal and add cases for your own failure modes.
- The 0.5 and 0.25 thresholds came from the pipeline's original detector. If a model's probabilities are systematically over- or under-confident, adjust `TOXICITY_THRESHOLD` in `utils.py` or prefer a model whose scores separate the labeled cases well.
