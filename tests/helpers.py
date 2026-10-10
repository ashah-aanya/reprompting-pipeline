"""Test helpers: build JudgeResults by hand and a scripted fake judge (no network)."""
from reprompting_pipeline.judge import JudgeResult, PASS_THRESHOLD


def _section(entries, is_toxicity=False):
    """entries: [(text, probability)] or [(text, probability, explanation)]."""
    items = []
    for entry in entries:
        text, prob = entry[0], entry[1]
        explanation = entry[2] if len(entry) > 2 else f"because of {text}"
        items.append({
            "instruction": text,
            "label": prob >= PASS_THRESHOLD,
            "follow_probability": prob,
            "explanation": explanation,
        })
    if is_toxicity:
        score = max((i["follow_probability"] for i in items), default=0.0)
    else:
        score = sum(i["follow_probability"] for i in items) / len(items) if items else 1.0
    return {"score": score, "instructions_list": items}


def make_result(adherence=(), groundedness=(), toxicity=()):
    """Build a JudgeResult. adherence/groundedness: (text, P(follows)); toxicity: (span, P(toxic))."""
    return JudgeResult(
        instruction_adherence=_section(adherence),
        groundedness=_section(groundedness),
        toxicity=_section(toxicity, is_toxicity=True),
    )


class ScriptedJudge:
    """Returns the scripted results in order (the last one repeats) and records every payload.

    A script entry may also be an Exception instance, which is raised instead of returned.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.payloads = []

    @property
    def calls(self):
        return len(self.payloads)

    def evaluate(self, payload):
        self.payloads.append(payload)
        entry = self.script[min(len(self.payloads), len(self.script)) - 1]
        if isinstance(entry, Exception):
            raise entry
        return entry


class CountingLLM:
    """An llm_fn that records each call and returns 'response N'."""

    def __init__(self):
        self.calls = []

    def __call__(self, template, system_prompt, context, user_query):
        self.calls.append({
            "template": template,
            "system_prompt": system_prompt,
            "context": context,
            "user_query": user_query,
        })
        return f"response {len(self.calls)}"
