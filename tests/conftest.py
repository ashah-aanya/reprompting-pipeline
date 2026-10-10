import types

import pytest

from helpers import ScriptedJudge, make_result


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """The retry decorator sleeps 1s+ between attempts; skip the waiting in tests."""
    # Replace utils' own `time` name, not time.sleep itself: that would be global and also
    # silence the sleeps that tests use to simulate a slow LLM.
    monkeypatch.setattr("reprompting_pipeline.utils.time", types.SimpleNamespace(sleep=lambda _seconds: None))


@pytest.fixture
def judge():
    """Fake judge for the ported scenario tests: fails the first response, passes every later one."""
    failing = make_result(
        adherence=[("Be concise", 0.1, "The response is long."), ("Stay neutral", 0.9)],
        groundedness=[("A claim from the context", 0.95)],
    )
    passing = make_result(
        adherence=[("Be concise", 0.95), ("Stay neutral", 0.9)],
        groundedness=[("A claim from the context", 0.95)],
    )
    return ScriptedJudge(failing, passing)


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.integration tests unless a Together key is available."""
    import os
    if os.environ.get("TOGETHER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="needs TOGETHER_API_KEY (live Together AI call)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
