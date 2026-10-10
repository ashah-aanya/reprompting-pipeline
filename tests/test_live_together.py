"""Live end-to-end tests against Together AI. Skipped unless TOGETHER_API_KEY is set (see conftest.py).

Run them with:  TOGETHER_API_KEY=... pytest -m integration -s
"""
import os

import pytest

from reprompting_pipeline import RepromptingConfig, run_reprompting_pipeline
from reprompting_pipeline.judge import TogetherJudge

pytestmark = pytest.mark.integration

CONTEXT = "The Orion X1 laptop has a 14-inch display, 16 GB of RAM and a 10-hour battery. It ships with a 1-year warranty."


def test_judge_flags_an_obvious_violation_and_an_unsupported_claim():
    judge = TogetherJudge(RepromptingConfig())
    result = judge.evaluate({
        "context": CONTEXT,
        "user_query": "Tell me about the Orion X1.",
        "instructions": ["Answer in at most two sentences.", "Do not mention price."],
        "generated_text": "The Orion X1 has a 14-inch display. It costs $999 and has a 20-hour battery. Buy it now!",
        "system_prompt": "You are a helpful assistant.",
    })
    adherence = {i["instruction"]: i["label"] for i in result.instruction_adherence["instructions_list"]}
    assert adherence["Do not mention price."] is False
    assert any(not claim["label"] for claim in result.groundedness["instructions_list"])


def test_full_pipeline_with_a_real_llm_and_judge():
    from together import Together

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def llm(template, system_prompt, context, user_query):
        prompt = template.safe_substitute(system_prompt=system_prompt, context=context, user_query=user_query)
        reply = client.chat.completions.create(
            model=RepromptingConfig().judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        return reply.choices[0].message.content

    result = run_reprompting_pipeline(
        llm_fn=llm,
        user_query="Tell me about the Orion X1.",
        context=CONTEXT,
        user_instructions=["Answer in at most two sentences.", "Do not mention price."],
        reprompting_config=RepromptingConfig(max_iterations=3, return_telemetry=True, return_summary=True),
    )
    print(result["summary"])
    assert isinstance(result["best_response"], str) and result["best_response"]
    assert len(result["telemetry"]) >= 1
