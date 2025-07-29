from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)

def my_llm(prompt: str) -> str:
    """Normal working LLM."""
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0
    )
    return response.choices[0].message.content

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

def get_config_with_latency_limit():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=2,
        latency_limit_ms=100
    )
    
def get_config_with_iteration_limit():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=-1,
    )

# --- Test Runs ---

def test_latency_limit():
    config = get_config_with_latency_limit()
    return run_reprompting_pipeline(
        user_query="Test invalid return",
        context="Context",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]
    )

def test_iteration_limit():
    config = get_config_with_iteration_limit()
    return run_reprompting_pipeline(
        user_query="What's the policy?",
        context="hi",  # Empty context
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]  # No instructions
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
    print("\n=== Running Selected Tests ===\n")

    latency_result = test_latency_limit()
    print("\n--- test_latency_limit ---")
    print(latency_result)

    iteration_result = test_iteration_limit()  # assuming test_iteration_limit was meant to call this
    print("\n--- test_max_iterations ---")
    print(iteration_result)
    
    empty_context_and_instructions_result = test_empty_context_and_instructions()  # assuming test_iteration_limit was meant to call this
    print("\n--- test_empty_context_and_instructions ---")
    print(empty_context_and_instructions_result)
