# reprompting-pipeline

What if your Large Language Models (LLMs) could rewrite its own answers until it got things right … *automatically?*

Anyone who’s worked with LLMs knows the drill. You give a model a prompt with specific instructions and still, the response comes back missing something. So you tweak the prompt and try again, asking it to fix the tone, then rephrase for conciseness. Eventually, you may get what you need but only after multiple rounds.

Re-prompting automates that loop. Instead of relying on massive models with high latency or manually fine-tuning outputs, this is a lightweight, stateless system that sits on top of any black-box LLM. A judge model gives precise feedback, and the pipeline turns it into a targeted corrective prompt to extract better outputs. You give it your model as a plain Python function, so it works with OpenAI, Anthropic, Together, a local model, or anything else.

## Goals

The pipeline is designed to:

- Boost instruction adherence
- Minimize hallucinations
- Flag and fix toxicity
- Stay fast and token-efficient
- Keep all models stateless for scalability

## How it works

Think of our system as an editorial feedback loop for LLM responses. With each pass through, the original LLM draft is reviewed and returned with targeted feedback to fuel iterative improvement until it converges on an error-free output. Each iteration can optionally log a JSON blob of diagnostics for iteration progression, providing full transparency into how the model evolves with each re-prompt.

![The re-prompting loop](docs/img/reprompting-pipeline.png)

1. **Draft the initial response (P₀ → R₀).** The user sends a query, context, and instructions to a stateless LLM that generates a response that may fail to fully comply with the provided, especially critical, instructions.
2. **Audit with the judge (R₀ → E₀).** A small Together AI model checks the response for instruction adherence, groundedness (hallucination against your context) and toxicity, and explains each failure.
3. **Generate a corrective prompt (E₀ → P₁).** The failures become a targeted correction ("you violated X because Y; keep doing Z"), with the tone set by how many instructions failed.
4. **Iterative refinement (P₁ → R₁ … Pₙ → Rₙ).** Steps 2-4 repeat until everything passes, `max_iterations` is reached, or the latency budget runs out. The best draft seen is returned.

The full original write-up, with the dataset, results and design reasoning, is in [docs/blog.md](docs/blog.md). **Note:** its numbers were measured with AIMon's original feedback model, not this repository's Together judge.

## Install

```bash
git clone https://github.com/ashah-aanya/reprompting-pipeline.git
cd reprompting-pipeline
pip install -e .
export TOGETHER_API_KEY=...   # used by the judge
```

Python 3.10+. The only runtime dependency is the [`together`](https://pypi.org/project/together/) SDK (2.x).

## Quick start

```python
import os
from together import Together
from reprompting_pipeline import RepromptingConfig, run_reprompting_pipeline

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

# Your model, as a function. The pipeline hands you a prompt template to fill in.
def my_llm(template, system_prompt, context, user_query) -> str:
    prompt = template.safe_substitute(system_prompt=system_prompt, context=context, user_query=user_query)
    reply = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # any chat model you like
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0,
    )
    return reply.choices[0].message.content

result = run_reprompting_pipeline(
    llm_fn=my_llm,
    user_query="Tell me about the Orion X1.",
    context="The Orion X1 laptop has a 14-inch display, 16 GB of RAM and a 10-hour battery.",
    user_instructions=["Answer in at most two sentences.", "Do not mention price."],
    reprompting_config=RepromptingConfig(max_iterations=3, return_telemetry=True, return_summary=True),
)

print(result["summary"])        # e.g. "2 iterations, 0 failed instructions remaining"
print(result["best_response"])
```

Write instructions that are **specific and checkable** ("at most two sentences", "start with Yes or No"). Vague or contradictory instructions can't be fixed by re-prompting.

## Configuration

`RepromptingConfig` (all fields optional):

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | `2` | Max LLM calls: 1 initial + re-prompts. Must be ≥ 1. 2-3 is a good range. |
| `latency_limit_ms` | `None` | Stop re-prompting once 75% of this budget is used. |
| `together_api_key` | env `TOGETHER_API_KEY` | Key for the judge. Only read when the judge is created. |
| `judge_model` | see `config.py` | Together model that audits responses. |
| `judge_max_tokens` / `judge_temperature` | `1500` / `0.0` | Judge generation settings. |
| `return_telemetry` | `False` | Add per-iteration details to the result. |
| `return_summary` | `False` | Add a one-line summary to the result. |
| `user_model_max_retries` | `2` | Total attempts (exponential backoff) for each `llm_fn` call. |
| `feedback_model_max_retries` | `2` | Total attempts (exponential backoff) for each judge call. |

`run_reprompting_pipeline` returns `{"best_response": str, "telemetry": [...]?, "summary": str?}`. Full details are in [docs/reprompting.md](docs/reprompting.md).

## Bring your own judge

The judge is one small interface: any object with `evaluate(payload) -> JudgeResult`. Pass it as `judge=` to use another model or provider, or a fake in tests. See [docs/judge.md](docs/judge.md) for the report format, the prompt, and how to measure and tune a judge with the included eval set.

## Status and honest limitations

- **The judge is an LLM, not a calibrated classifier.** Its probabilities are the model's own estimates, and the thresholds (adherence passes at 0.5, toxicity fails above 0.25) were inherited from the pipeline's original detector. Measure your judge model with `evals/run_judge_eval.py` before relying on it.
- Subjective goals (tone, creativity) improve inconsistently.
- Every iteration costs one LLM call plus one judge call, so latency and cost grow with `max_iterations`.
- No benchmark results are published for this version of the judge.

## Development

```bash
pip install -e ".[dev]"
pytest                                  # offline: fake judge, no network (live tests are skipped without a key)
TOGETHER_API_KEY=... pytest -m integration -s
python evals/run_judge_eval.py --dry-run
```

## Origin

I built the first version of this pipeline as the re-prompting API during my internship at AIMon (summer 2025), as part of AIMon's open-source Python SDK, where it called AIMon's hosted detector. This repository is an independent, standalone continuation: it extracts that code and replaces the detector with the LLM judge in `src/reprompting_pipeline/judge.py` (Together AI), so it keeps working on its own.

This project is not affiliated with or endorsed by AIMon.

**About the git history.** The commits from July 2025 are the original development history, extracted from the SDK repository with `git filter-repo` and limited to the re-prompting code. Commit hashes therefore differ from the originals, and the author email was replaced with a GitHub noreply address; dates, authorship and commit contents are preserved (a few messages were lightly reworded to remove names and links). The standalone port to Together AI is the most recent commit, from October 2026.

## License

MIT, see [LICENSE](LICENSE).
