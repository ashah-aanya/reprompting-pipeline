from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)


# --- MOCKED LLM FUNCTIONS ---
def my_llm(prompt: str) -> str:
    """Normal working LLM."""
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0
    )
    return response.choices[0].message.content

def llm_fn_failure(prompt: str) -> str:
    """Simulates API or network failure."""
    raise RuntimeError("LLM call failed intentionally for testing")

def llm_fn_incorrect_return_value(prompt: str):
    """Simulates an incorrect return type (edge case)."""
    return 42

# --- MOCKED CONFIG FUNCTIONS ---

def get_config():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=2,
    )

# --- Test Runs ---

def test_llm_failure():
    config = get_config()
    return run_reprompting_pipeline(
        user_query="Test failure",
        context="Context",
        llm_fn=llm_fn_failure,
        reprompting_config=config,
        user_instructions=[]
    )

def test_invalid_return_value():
    config = get_config()
    return run_reprompting_pipeline(
        user_query="Test invalid return",
        context="Context",
        llm_fn=llm_fn_incorrect_return_value,
        reprompting_config=config,
        user_instructions=[]
    )

def test_empty_context_and_instructions():
    config = get_config()
    return run_reprompting_pipeline(
        user_query="What's the policy?",
        context="",  # Empty context
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]  # No instructions
    )
    
if __name__ == "__main__":
    #test_llm_failure()
    #test_invalid_return_value()
    print(test_empty_context_and_instructions())