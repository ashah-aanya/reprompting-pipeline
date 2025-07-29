from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)

def my_llm(system_prompt: str, context = None, user_query = None) -> str:
    """Normal working LLM."""
    prompt = system_prompt
    if context:
        prompt += f"\nContext: {context}"
    if user_query:
        prompt += f"\n Query: {user_query}"
    print("printing prompt")
    print(prompt)
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
def get_config_without_telemetry():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=False,
        return_aimon_summary=False,
        application_name="api_test",
        max_iterations=2,
    )

def get_config_with_low_latency_limit():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=2,
        latency_limit_ms=100
    )
    
def get_config_with_latency_limit():
    return RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=False,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
        max_iterations=3,
        latency_limit_ms=5000
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

def test_low_latency_limit():
    config = get_config_with_low_latency_limit()
    return run_reprompting_pipeline(
        user_query="Test latency limit termination return",
        context="Context",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]
    )
    
def test_latency_limit():
    config = get_config_with_latency_limit()
    return run_reprompting_pipeline(
        user_query="What is context?",
        context="Context",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=["Do not use the letter e", "Only use the letter e"]
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

def test_no_telemetry():
    """Test with telemetry and summary disabled."""
    config = get_config_without_telemetry()
    result = run_reprompting_pipeline(
        user_query="What's the policy?",
        context="Context for telemetry disabled run",
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=[]
    )
    # Ensure telemetry and summary are not present
    assert "telemetry" not in result
    assert "summary" not in result
    return result

def test_no_system_prompt():
    """Test with telemetry and summary disabled."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="What's the policy?",
        context="Context for telemetry disabled run",
        llm_fn=my_llm,
        reprompting_config=config,
        # contradictory instructions to test reprompting prompt
        user_instructions=["use the letter e only", "do not use the letter e"]
    )

def test_with_system_prompt():
    """Test with telemetry and summary disabled."""
    config = get_config()
    return run_reprompting_pipeline(
        user_query="What's the policy?",
        context="Context for telemetry disabled run",
        llm_fn=my_llm,
        reprompting_config=config,
        # contradictory instructions to test reprompting prompt
        user_instructions=["use the letter e only", "do not use the letter e"],
        system_prompt= "this is a system prompt"
    )

if __name__ == "__main__":
    print("\n=== Running Selected Tests ===\n")
    print(my_llm("This is a test system prompt.", context="Sample context", user_query="What is the policy?"))


    """low_latency_result = test_low_latency_limit()
    print("\n--- test_low_latency_limit ---")
    print(low_latency_result)
    
    latency_result = test_latency_limit()
    print("\n--- test_latency_limit ---")
    print(latency_result)

    iteration_result = test_iteration_limit()
    print("\n--- test_iteration_limit ---")
    print(iteration_result)
    
    empty_context_and_instructions_result = test_empty_context_and_instructions()
    print("\n--- test_empty_context_and_instructions ---")
    print(empty_context_and_instructions_result)"""
    
    no_telemetry_result = test_no_telemetry()
    print("\n--- test_no_telemetry ---")
    print(no_telemetry_result)
    
    no_system_prompt_result = test_no_system_prompt()
    print("\n--- test_no_system_prompt ---")
    print(no_system_prompt_result)
    
    system_prompt_result = test_with_system_prompt()
    print("\n--- test_with_system_prompt ---")
    print(system_prompt_result)
    
