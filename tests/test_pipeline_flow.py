"""Behavioral tests for the re-prompting loop, driven by a scripted judge and a counting LLM (no network)."""
import time
from string import Template

import pytest

from helpers import CountingLLM, ScriptedJudge, make_result
from reprompting_pipeline import RepromptingConfig, StopReasons, run_reprompting_pipeline

A, B = "Use a formal tone.", "Keep it under 50 words."

PASS = make_result(adherence=[(A, 0.95), (B, 0.9)])
FAIL_BOTH = make_result(adherence=[(A, 0.1, "Too casual."), (B, 0.1, "Too long.")])
FAIL_ONE = make_result(adherence=[(A, 0.9), (B, 0.1, "Too long.")])
TOXIC = make_result(adherence=[(A, 0.95), (B, 0.9)], toxicity=[("you idiot", 0.9, "An insult.")])


def run(judge, llm=None, **config):
    llm = llm or CountingLLM()
    config.setdefault("return_telemetry", True)
    config.setdefault("return_summary", True)
    result = run_reprompting_pipeline(
        llm_fn=llm,
        user_query="Explain refunds.",
        system_prompt="You are a support agent.",
        context="Refunds take 5 days.",
        user_instructions=[A, B],
        reprompting_config=RepromptingConfig(**config),
        judge=judge,
    )
    return result, llm


# --- stopping ---------------------------------------------------------------------------

def test_first_response_passes_so_there_is_no_reprompt():
    judge = ScriptedJudge(PASS)
    result, llm = run(judge, max_iterations=3)

    assert result["best_response"] == "response 1"
    assert len(llm.calls) == 1 and judge.calls == 1
    assert result["telemetry"][-1]["stop_reason"] == StopReasons.ALL_INSTRUCTIONS_ADHERED
    assert result["summary"] == "1 iteration, 0 failed instructions remaining"


def test_failure_then_pass_makes_exactly_one_llm_call_per_iteration():
    # Regression: the original code called llm_fn twice per re-prompt (one call without arguments).
    judge = ScriptedJudge(FAIL_BOTH, PASS)
    result, llm = run(judge, max_iterations=3)

    assert len(llm.calls) == 2, "exactly one llm_fn call per iteration"
    assert judge.calls == 2
    assert result["best_response"] == "response 2"
    assert result["summary"] == "2 iterations, 0 failed instructions remaining"
    assert all(call["user_query"] == "Explain refunds." for call in llm.calls)
    assert all(call["context"] == "Refunds take 5 days." for call in llm.calls)


def test_stops_at_max_iterations_and_picks_the_lowest_residual_error():
    # residual error: both fail -> 1.0, one fails -> 0.9. Iteration 2 is the best of the three.
    judge = ScriptedJudge(FAIL_BOTH, FAIL_ONE, FAIL_BOTH)
    result, llm = run(judge, max_iterations=3)

    assert len(llm.calls) == 3
    assert result["telemetry"][-1]["stop_reason"] == StopReasons.MAX_ITERATIONS_REACHED
    assert result["best_response"] == "response 2"
    assert result["summary"] == "3 iterations, 1 failed instructions remaining"


def test_max_iterations_one_never_reprompts():
    judge = ScriptedJudge(FAIL_BOTH)
    result, llm = run(judge, max_iterations=1)
    assert len(llm.calls) == 1 and judge.calls == 1
    assert result["telemetry"][-1]["stop_reason"] == StopReasons.MAX_ITERATIONS_REACHED


def test_toxicity_alone_keeps_the_loop_going():
    judge = ScriptedJudge(TOXIC, PASS)
    result, llm = run(judge, max_iterations=3)

    assert len(llm.calls) == 2
    assert result["telemetry"][0]["stop_reason"] == StopReasons.CONTINUE_TOXICITY
    assert result["best_response"] == "response 2"
    prompt = llm.calls[1]["template"].template
    assert "Your reply contained toxic content" in prompt and "you idiot" in prompt


def test_latency_budget_stops_before_a_reprompt():
    def slow_llm(template, system_prompt, context, user_query):
        time.sleep(0.01)
        return "slow response"

    judge = ScriptedJudge(FAIL_BOTH, PASS)
    result, _ = run(judge, llm=slow_llm, max_iterations=5, latency_limit_ms=1)

    assert judge.calls == 1
    assert result["telemetry"][-1]["stop_reason"] == StopReasons.LATENCY_LIMIT_EXCEEDED


# --- the corrective prompt --------------------------------------------------------------------

def test_corrective_prompt_names_failures_and_reminds_about_passes():
    judge = ScriptedJudge(FAIL_ONE, PASS)
    _, llm = run(judge, max_iterations=2)

    first, second = llm.calls[0]["template"].template, llm.calls[1]["template"].template
    assert first.startswith("System:\n${system_prompt}")
    assert "Revise your previous response" in second
    assert "Previous response:\nresponse 1" in second
    assert 'Violated Instruction: "Keep it under 50 words."' in second
    assert "Too long." in second
    assert "continue to follow these instructions:\n- Use a formal tone." in second
    assert "Almost there" in second  # one failure -> gentle tone


def test_corrective_prompt_tone_scales_with_failure_count():
    judge = ScriptedJudge(FAIL_BOTH, PASS)
    _, llm = run(judge, max_iterations=2)
    assert "Some parts were off" in llm.calls[1]["template"].template


def test_templates_are_safe_to_substitute_with_the_pipeline_arguments():
    judge = ScriptedJudge(FAIL_BOTH, PASS)
    _, llm = run(judge, max_iterations=2)
    call = llm.calls[1]
    assert isinstance(call["template"], Template)
    filled = call["template"].substitute(
        system_prompt=call["system_prompt"], context=call["context"], user_query=call["user_query"]
    )
    assert "You are a support agent." in filled and "Refunds take 5 days." in filled


# --- outputs ---------------------------------------------------------------------------------

def test_telemetry_entries_have_the_documented_keys_and_no_private_metadata():
    judge = ScriptedJudge(FAIL_ONE, PASS)
    result, _ = run(judge, max_iterations=2)

    expected = {
        "iteration", "cumulative_latency_ms", "scores", "response_feedback", "residual_error",
        "failed_instructions_count", "stop_reason", "prompt_template", "response_text",
    }
    assert len(result["telemetry"]) == 2
    for entry in result["telemetry"]:
        assert set(entry) == expected
    assert set(result["telemetry"][0]["scores"]) == {"groundedness", "instruction_adherence", "toxicity"}
    assert result["telemetry"][0]["failed_instructions_count"] == 1
    assert result["telemetry"][0]["stop_reason"] == StopReasons.CONTINUE


def test_telemetry_and_summary_are_omitted_when_disabled():
    result, _ = run(ScriptedJudge(PASS), return_telemetry=False, return_summary=False)
    assert set(result) == {"best_response"}


def test_missing_context_and_system_prompt_get_placeholders():
    llm = CountingLLM()
    run_reprompting_pipeline(
        llm_fn=llm, user_query="hi", reprompting_config=RepromptingConfig(), judge=ScriptedJudge(PASS),
    )
    assert llm.calls[0]["context"] == "[no context provided]"
    assert llm.calls[0]["system_prompt"] == "[no system prompt provided]"


# --- retries and errors -------------------------------------------------------------------------

def test_judge_error_is_retried_then_succeeds():
    judge = ScriptedJudge(RuntimeError("rate limited"), PASS)
    result, llm = run(judge, max_iterations=2, feedback_model_max_retries=2)
    assert judge.calls == 2 and len(llm.calls) == 1
    assert result["best_response"] == "response 1"


def test_judge_that_keeps_failing_reraises_the_last_error():
    judge = ScriptedJudge(RuntimeError("down"))
    with pytest.raises(RuntimeError, match="down"):
        run(judge, feedback_model_max_retries=2)
    assert judge.calls == 2


def test_llm_error_is_retried_then_succeeds():
    attempts = []

    def flaky(template, system_prompt, context, user_query):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("blip")
        return "ok"

    result, _ = run(ScriptedJudge(PASS), llm=flaky, user_model_max_retries=2)
    assert result["best_response"] == "ok" and len(attempts) == 2


def test_max_iterations_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_iterations"):
        RepromptingConfig(max_iterations=0)
