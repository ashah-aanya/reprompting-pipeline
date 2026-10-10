import pytest
from string import Template

from reprompting_pipeline.config import RepromptingConfig
from reprompting_pipeline.runner import run_reprompting_pipeline

# --- MOCKED LLM FUNCTIONS ---
def my_llm(prompt_template: Template, system_prompt=None, context=None, user_query=None) -> str:
    """Simulates a normal working LLM that returns a string response. Just returns filled_prompt for test"""
    filled_prompt = prompt_template.safe_substitute(
        system_prompt=system_prompt or "",
        context=context or "",
        user_query=user_query or ""
    )
    return filled_prompt

def llm_fn_failure(prompt_template: Template, system_prompt=None, context=None, user_query=None) -> str:
    """Simulates an LLM call that fails every time."""
    raise RuntimeError("LLM call failed intentionally for testing")

def llm_fn_incorrect_return_value(prompt_template: Template, system_prompt=None, context=None, user_query=None):
    """Simulates an LLM that returns an invalid type instead of a string."""
    return 42

# --- MOCKED CONFIG FACTORIES ---
def get_config():
    """Returns a valid base configuration for most tests."""
    return RepromptingConfig(
        return_telemetry=True,
        return_summary=True,
        max_iterations=2,
    )

# --- TESTS EXPECTING FAILURES ---
def test_llm_failure(judge):
    """Should re-raise the LLM function's error when it always fails."""
    config = get_config()
    with pytest.raises(RuntimeError, match="LLM call failed intentionally for testing"):
        run_reprompting_pipeline(
            user_query="Test LLM failure handling",
            context="Context for failure test",
            llm_fn=llm_fn_failure,
            judge=judge,
            reprompting_config=config,
            user_instructions=[]
        )

def test_invalid_llm_fn(judge):
    """Should raise TypeError when LLM function is None."""
    config = get_config()
    with pytest.raises(TypeError):
        run_reprompting_pipeline(
            user_query="Test invalid LLM fn",
            context="Context for failure test",
            llm_fn=None,
            judge=judge,
            reprompting_config=config,
            user_instructions=[]
        )

def test_invalid_return_value(judge):
    """Should raise TypeError when the LLM returns a non-string value."""
    config = get_config()
    with pytest.raises(TypeError, match="LLM returned invalid type int, expected str."):
        run_reprompting_pipeline(
            user_query="Test invalid return type",
            context="Context for type error",
            llm_fn=llm_fn_incorrect_return_value,
            judge=judge,
            reprompting_config=config,
            user_instructions=[]
        )

def test_empty_query(judge):
    """Empty query should raise a ValueError."""
    config = get_config()
    with pytest.raises(ValueError, match="user_query must be a non-empty string"):
        run_reprompting_pipeline(
            user_query="",
            context="",
            llm_fn=my_llm,
            judge=judge,
            reprompting_config=config,
            user_instructions=[]
        )

def test_missing_api_key(monkeypatch):
    """With no injected judge and no Together key anywhere, the default judge must fail clearly (not at import)."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Together AI API key"):
        run_reprompting_pipeline(
            user_query="Testing with no Together API key",
            context="Context for missing key test",
            llm_fn=my_llm,
            reprompting_config=get_config(),
            user_instructions=[]
        )
