"""Judge parsing/validation tests. No network: the model's reply is a string we write by hand."""
import json
import sys
import types

import pytest

from reprompting_pipeline.config import RepromptingConfig
from reprompting_pipeline.judge import (
    JUDGE_SCHEMA,
    JudgeParseError,
    TogetherJudge,
    build_messages,
    make_together_complete_fn,
    parse_judge_output,
)

INSTRUCTIONS = ["Answer in two sentences.", "Do not mention price."]


def report(adherence=None, groundedness=None, toxicity=None):
    return json.dumps({
        "instruction_adherence": adherence if adherence is not None else [],
        "groundedness": groundedness if groundedness is not None else [],
        "toxicity": toxicity if toxicity is not None else [],
    })


def item(text, p, explanation="why"):
    return {"instruction": text, "follow_probability": p, "explanation": explanation}


# --- happy path -----------------------------------------------------------------------

def test_valid_report_is_parsed_with_derived_labels_and_scores():
    raw = report(
        adherence=[item(INSTRUCTIONS[0], 0.9), item(INSTRUCTIONS[1], 0.1, "mentions $999")],
        groundedness=[item("14-inch display", 0.97)],
        toxicity=[item("an insult", 0.8)],
    )
    result = parse_judge_output(raw, INSTRUCTIONS, has_context=True)

    adh = result.instruction_adherence
    assert [i["label"] for i in adh["instructions_list"]] == [True, False]
    assert adh["score"] == pytest.approx(0.5)
    assert adh["instructions_list"][1]["explanation"] == "mentions $999"
    assert result.groundedness["score"] == pytest.approx(0.97)
    assert result.toxicity["score"] == pytest.approx(0.8)  # max probability of toxicity


def test_clean_response_has_perfect_scores():
    result = parse_judge_output(report(adherence=[item(INSTRUCTIONS[0], 1.0)]), INSTRUCTIONS[:1], has_context=True)
    assert result.toxicity == {"score": 0.0, "instructions_list": []}
    assert result.groundedness["score"] == 1.0


def test_nested_instructions_list_form_is_accepted():
    raw = json.dumps({
        "instruction_adherence": {"instructions_list": [item(INSTRUCTIONS[0], 0.8)]},
        "groundedness": {"instructions_list": []},
        "toxicity": {"instructions_list": []},
    })
    result = parse_judge_output(raw, INSTRUCTIONS[:1], has_context=True)
    assert len(result.instruction_adherence["instructions_list"]) == 1


# --- tolerant of messy model output -----------------------------------------------------

def test_code_fences_and_surrounding_prose_are_stripped():
    raw = "Here is my audit:\n```json\n" + report(adherence=[item(INSTRUCTIONS[0], 0.7)]) + "\n```\nHope that helps!"
    result = parse_judge_output(raw, INSTRUCTIONS[:1], has_context=True)
    assert result.instruction_adherence["instructions_list"][0]["follow_probability"] == 0.7


def test_prose_around_bare_json_is_stripped():
    raw = "Sure! " + report(adherence=[item(INSTRUCTIONS[0], 0.7)]) + " Done."
    result = parse_judge_output(raw, INSTRUCTIONS[:1], has_context=True)
    assert result.instruction_adherence["score"] == pytest.approx(0.7)


def test_paraphrased_instruction_is_snapped_back_to_the_original():
    # reprompter matches passed/failed instructions by exact text, so paraphrases must be mapped back
    raw = report(adherence=[item("Answer in two sentences", 0.9), item("do not mention the price.", 0.2)])
    result = parse_judge_output(raw, INSTRUCTIONS, has_context=True)
    assert [i["instruction"] for i in result.instruction_adherence["instructions_list"]] == INSTRUCTIONS


def test_probabilities_are_clamped_and_numeric_strings_accepted():
    raw = report(adherence=[item(INSTRUCTIONS[0], 1.7), item(INSTRUCTIONS[1], "0.25")])
    result = parse_judge_output(raw, INSTRUCTIONS, has_context=True)
    probs = [i["follow_probability"] for i in result.instruction_adherence["instructions_list"]]
    assert probs == [1.0, 0.25]


def test_items_without_a_usable_probability_are_skipped():
    raw = report(
        adherence=[item(INSTRUCTIONS[0], 0.9), item(INSTRUCTIONS[1], "high"), "not an object", item("x", True)],
    )
    result = parse_judge_output(raw, INSTRUCTIONS, has_context=True)
    assert len(result.instruction_adherence["instructions_list"]) == 1


def test_label_is_derived_from_probability_even_if_model_sends_a_contradicting_label():
    raw = report(adherence=[{**item(INSTRUCTIONS[0], 0.1), "label": True}])
    result = parse_judge_output(raw, INSTRUCTIONS[:1], has_context=True)
    assert result.instruction_adherence["instructions_list"][0]["label"] is False


def test_missing_groundedness_and_toxicity_sections_default_safely():
    raw = json.dumps({"instruction_adherence": [item(INSTRUCTIONS[0], 0.9)]})
    result = parse_judge_output(raw, INSTRUCTIONS[:1], has_context=True)
    assert result.groundedness["instructions_list"] == []
    assert result.toxicity["instructions_list"] == []


# --- the judge is told what to skip ----------------------------------------------------------

def test_no_instructions_means_adherence_is_empty_even_if_model_invents_items():
    raw = report(adherence=[item("made up", 0.1)])
    result = parse_judge_output(raw, [], has_context=True)
    assert result.instruction_adherence == {"score": 1.0, "instructions_list": []}


def test_no_context_means_groundedness_is_empty_even_if_model_invents_claims():
    raw = report(groundedness=[item("a claim", 0.1)])
    result = parse_judge_output(raw, [], has_context=False)
    assert result.groundedness == {"score": 1.0, "instructions_list": []}


# --- unusable output raises so the pipeline's retry re-calls the judge ---------------------------

@pytest.mark.parametrize("raw", [
    "",
    "I cannot do that.",
    "{not json",
    "[1, 2, 3]",
    '"just a string"',
    '{"unrelated": 1}',
])
def test_unusable_output_raises_parse_error(raw):
    with pytest.raises(JudgeParseError):
        parse_judge_output(raw, INSTRUCTIONS, has_context=True)


def test_missing_adherence_section_raises_when_instructions_were_given():
    with pytest.raises(JudgeParseError, match="instruction_adherence"):
        parse_judge_output(json.dumps({"groundedness": [], "toxicity": []}), INSTRUCTIONS, has_context=True)


# --- prompt building ---------------------------------------------------------------------

def test_build_messages_numbers_instructions_and_ends_with_the_real_request():
    payload = {"context": "ctx", "user_query": "q", "instructions": INSTRUCTIONS, "generated_text": "resp"}
    messages = build_messages(payload)
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    last = messages[-1]["content"]
    assert "1. Answer in two sentences." in last and "2. Do not mention price." in last
    assert last.rstrip().endswith("resp")
    # the worked examples must themselves be valid reports
    for example in (messages[2], messages[4]):
        assert set(json.loads(example["content"])) == set(JUDGE_SCHEMA["required"])


def test_build_messages_marks_missing_instructions_and_context():
    messages = build_messages({"context": "", "user_query": "q", "instructions": [], "generated_text": "r"})
    assert "INSTRUCTIONS:\n(none)" in messages[-1]["content"]
    assert "[no context provided]" in messages[-1]["content"]


# --- TogetherJudge with an injected completion function (no network) --------------------------

def test_together_judge_evaluate_uses_the_injected_completion_function():
    seen = {}

    def fake_complete(messages):
        seen["messages"] = messages
        return report(adherence=[item(INSTRUCTIONS[0], 0.9)], groundedness=[item("claim", 0.2)])

    judge = TogetherJudge(complete_fn=fake_complete)
    result = judge.evaluate({
        "context": "real context", "user_query": "q", "instructions": INSTRUCTIONS[:1], "generated_text": "r",
    })
    assert result.groundedness["instructions_list"][0]["label"] is False
    assert "real context" in seen["messages"][-1]["content"]


def test_together_judge_treats_placeholder_context_as_no_context():
    judge = TogetherJudge(complete_fn=lambda _m: report(groundedness=[item("claim", 0.1)]))
    result = judge.evaluate({
        "context": "[no context provided]", "user_query": "q", "instructions": [], "generated_text": "r",
    })
    assert result.groundedness["instructions_list"] == []


def test_together_judge_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Together AI API key"):
        TogetherJudge(RepromptingConfig())


def test_together_complete_fn_sends_the_schema_and_returns_the_reply_text(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content="the reply")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeTogether:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "together", types.SimpleNamespace(Together=FakeTogether))
    complete = make_together_complete_fn("sk-test", "some/model", max_tokens=123, temperature=0.0)

    assert complete([{"role": "user", "content": "hi"}]) == "the reply"
    assert captured["api_key"] == "sk-test"
    assert captured["model"] == "some/model"
    assert captured["max_tokens"] == 123
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "judge_report", "schema": JUDGE_SCHEMA},
    }


def test_together_complete_fn_can_disable_json_schema(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"))])

    fake = types.SimpleNamespace(Together=lambda api_key: types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions())))
    monkeypatch.setitem(sys.modules, "together", fake)
    make_together_complete_fn("k", "m", use_json_schema=False)([])
    assert "response_format" not in captured
