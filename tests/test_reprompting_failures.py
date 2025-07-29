from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)

# --- MOCKED LLM FUNCTIONS ---

def my_llm(prompt: str) -> str:
    """Simulates a normal working LLM that returns a string response."""
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0
    )
    return response.choices[0].message.content

def llm_fn_failure(prompt: str) -> str:
    """Simulates an LLM call that fails every time (e.g., API/network error)."""
    raise RuntimeError("LLM call failed intentionally for testing")

def llm_fn_incorrect_return_value(prompt: str):
    """Simulates an LLM that returns an invalid type instead of a string."""
    return 42

# --- MOCKED CONFIG FACTORIES ---

def get_config():
    """Returns a valid base configuration for most tests."""
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=2,
    )

def get_config_with_invalid_aimon_api_key():
    """Returns a config with an intentionally invalid AIMon API key."""
    return RepromptingConfig(
        aimon_api_key="invalid key",
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=3,
    )

# --- TEST CASES ---

def test_llm_failure():
    """Test behavior when the LLM function fails (e.g., network/API error)."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="Test LLM failure handling",
        context="Context for failure test",
        llm_fn=llm_fn_failure,
        reprompting_config=config,
        user_instructions=[]
    )
    
def test_invalid_llm_fn():
    """Test behavior when the LLM function fails (e.g., network/API error)."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="Test LLM failure handling",
        context="Context for failure test",
        llm_fn=None,
        reprompting_config=config,
        user_instructions=[]
    )

def test_invalid_return_value():
    """Test behavior when the LLM function returns a non-string value."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="Test invalid return type",
        context="Context for type error",
        llm_fn=llm_fn_incorrect_return_value,
        reprompting_config=config,
        user_instructions=[]
    )

def test_empty_query():
    """Test pipeline behavior when query, context, and instructions are empty."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="",
        context="",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]
    )

def test_invalid_api_key():
    """Test behavior when using an invalid AIMon API key."""
    config = get_config_with_invalid_aimon_api_key()
    return run_reprompting_pipeline(
        user_query="Testing with invalid AIMon API key",
        context="Context for invalid key test",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]
    )

# --- RUN TESTS MANUALLY ---
if __name__ == "__main__":
    print("\n=== Running Descriptive Tests ===\n")

    print("\n--- test_empty_query ---")
    print(test_empty_query())

    print("\n--- test_invalid_api_key ---")
    print(test_invalid_api_key())

    print("\n--- test_llm_failure ---")
    print(test_llm_failure())
    
    print("\n--- test_invalid_llm ---")
    print(test_invalid_llm_fn())

    print("\n--- test_invalid_return_value ---")
    print(test_invalid_return_value())
