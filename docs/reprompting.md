# Automated Re-Prompting Pipeline

The pipeline improves an LLM's output by auditing each response with a **judge** model and turning the audit into a targeted corrective prompt. It is stateless: each call to your model carries everything it needs, so it works with any LLM.

## How it works

![The re-prompting loop](img/reprompting-pipeline.png)

1. **Generate** (`P0 -> R0`): your prompt goes to the primary LLM, which produces an initial response.
2. **Judge** (`R0 -> E0`): the judge checks the response for instruction adherence, groundedness and toxicity, and returns a structured report with an explanation for each failure ([judge.md](judge.md)).
3. **Decide:** the loop stops when everything passes, when `max_iterations` is reached, or when 75% of `latency_limit_ms` is used. Toxicity keeps the loop going even if every instruction passed.
4. **Re-prompt** (`E0 -> CP1`): a corrective prompt names each failure with its explanation, sets a tone by severity, and reminds the model which instructions it already followed.
5. **Refine** (`CP1 -> R1 ...`): the LLM revises. Steps 2-5 repeat.
6. **Return:** the response with the lowest *residual error* across all iterations, not necessarily the last one.

## `run_reprompting_pipeline`

```python
from reprompting_pipeline import run_reprompting_pipeline

def run_reprompting_pipeline(
    llm_fn: Callable[[Template, str, str, str], str],
    user_query: str,
    system_prompt: str = None,
    context: str = None,
    user_instructions: List[str] = None,
    reprompting_config: RepromptingConfig = None,
    judge=None,
) -> dict
```

### Parameters

- **`llm_fn`**: your LLM function (see below).
- **`user_query`** *(str, required)*: the question or task. Must be non-empty.
- **`system_prompt`** *(str, optional)*: role/behavior definition. Replaced by `"[no system prompt provided]"` when empty.
- **`context`** *(str, optional)*: background text the answer should rely on (a document, policy, retrieved chunks). Replaced by `"[no context provided]"` when empty, and groundedness is then not checked.
- **`user_instructions`** *(List[str], optional)*: specific, checkable rules the response must follow. When empty, instruction adherence is not checked.
- **`reprompting_config`** *(RepromptingConfig, optional)*: limits and output options.
- **`judge`** *(optional)*: any object with `evaluate(payload) -> JudgeResult`. Defaults to a Together AI judge built from the config.

### Returns

A `dict`:

- **`best_response`** *(str)*: the best answer across iterations.
- **`summary`** *(str, if `return_summary`)*: e.g. `"2 iterations, 0 failed instructions remaining"`.
- **`telemetry`** *(list[dict], if `return_telemetry`)*: one entry per iteration:

| Key | Meaning |
|---|---|
| `iteration` | Iteration number, starting at 1. |
| `cumulative_latency_ms` | Time since the run started. |
| `scores` | `{"groundedness", "instruction_adherence", "toxicity"}`, each 0-1. Higher is better for the first two; **lower is better for toxicity**. |
| `response_feedback` | Failed checks, each `{type, source, instruction, score, explanation}`. |
| `residual_error` | 0 (best) to 1 (worst). See below. |
| `failed_instructions_count` | Number of failed checks in this iteration. |
| `stop_reason` | Why the loop continued or stopped: `all_instructions_adhered`, `max_iterations_reached`, `latency_limit_exceeded`, `instructions_failed_continue_reprompting`, `toxicity_detect_continue_reprompting`. |
| `prompt_template` | The prompt template (with placeholders) used for this iteration. |
| `response_text` | The raw LLM output. |

**Residual error** averages a penalty per check: a check with follow-probability `p >= 0.5` costs 0, otherwise `2 * (1 - p)`. Toxicity uses `1 - p_toxic`. The average is clamped to [0, 1].

## Writing `llm_fn`

`llm_fn` connects the pipeline to your model:

```python
def my_llm(recommended_prompt_template: Template, system_prompt, context, user_query) -> str: ...
```

1. The pipeline passes a `string.Template` (the first call uses a simple template, later calls a corrective one).
2. Fill the placeholders `system_prompt`, `context` and `user_query`. Prefer `safe_substitute()`: it doesn't raise if a placeholder is missing. You may also ignore the template and build your own prompt.
3. Send it to any model and return the reply as a plain `str`.

Returning a non-string raises `TypeError`.

## `RepromptingConfig`

| Name | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | int | 2 | Max LLM calls (1 initial + re-prompts). Must be >= 1. |
| `latency_limit_ms` | int \| None | None | Stop re-prompting once 75% of this budget is used. |
| `together_api_key` | str \| None | env `TOGETHER_API_KEY` | Judge key; read only when the default judge is created. |
| `judge_model` | str | see `config.py` | Together model used as the judge. |
| `judge_max_tokens` | int | 1500 | Max tokens for the judge's report. |
| `judge_temperature` | float | 0.0 | Judge sampling temperature. |
| `return_telemetry` | bool | False | Include `telemetry` in the result. |
| `return_summary` | bool | False | Include `summary` in the result. |
| `user_model_max_retries` | int | 2 | Total attempts per `llm_fn` call, with exponential backoff. |
| `feedback_model_max_retries` | int | 2 | Total attempts per judge call, with exponential backoff. |

### Error handling

If `llm_fn` or the judge still fails after its attempts are used up, the **last error is re-raised unchanged** (for example your own `RuntimeError`, or a network error from Together). A judge reply that cannot be parsed counts as a failed attempt and is retried.

## Example

```python
import json, os
from together import Together
from reprompting_pipeline import RepromptingConfig, run_reprompting_pipeline

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

def my_llm(template, system_prompt, context, user_query) -> str:
    prompt = template.safe_substitute(system_prompt=system_prompt, context=context, user_query=user_query)
    reply = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0,
    )
    return reply.choices[0].message.content

result = run_reprompting_pipeline(
    llm_fn=my_llm,
    user_query="Can I get a refund on my headphones?",
    system_prompt="You are a concise support agent.",
    context="Refund requests must be made within 30 days of purchase. Digital goods are non-refundable.",
    user_instructions=["Answer in one sentence.", "Start the answer with 'Yes' or 'No'."],
    reprompting_config=RepromptingConfig(max_iterations=3, return_telemetry=True, return_summary=True),
)

print(result["summary"])
print(result["best_response"])
print(json.dumps(result["telemetry"], indent=2))
```

## Using it with a RAG system

The pipeline doesn't care where `context` comes from. Retrieve with any framework (LlamaIndex, LangChain, your own), join the chunks into one string, and pass it as `context`. Groundedness is then checked against exactly what you retrieved.

## Limitations

- Needs clear, checkable instructions. Vague or contradictory ones cannot be fixed.
- Subjective attributes (tone, creativity) improve inconsistently.
- Each iteration adds one LLM call and one judge call. Tune `max_iterations` and `latency_limit_ms` to balance quality against speed and cost.
- The judge is an LLM: it can be wrong, and its probabilities are uncalibrated. See [judge.md](judge.md#measuring-and-tuning).
