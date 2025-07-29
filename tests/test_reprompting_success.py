from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

# --- Load API keys from environment variables ---
# These must be set before running this script:
# export TOGETHER_API_KEY="your_together_api_key"
# export AIMON_API_KEY="your_aimon_api_key"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

# --- Initialize Together client for LLM calls ---
client = Together(api_key=TOGETHER_API_KEY)

# --- LLM Function ---
def my_llm(prompt: str) -> str:
    """
    Calls a Together-hosted Mistral-7B model and returns the generated text.
    This is a simple wrapper to make LLM calls look like a standard Python function.
    """
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",  # You can replace with any Together-hosted model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Test Case: Successful Run ---
def test_successful_run():
    """
    This test simulates a realistic pipeline run:
    - A long, complex context (insurance policy excerpt).
    - A query that asks for human-friendly simplification of that policy.
    - Multiple instructions enforcing tone and style.
    - Full telemetry and summary enabled for debugging/inspection.
    """
    # Configure the pipeline for this test
    config = RepromptingConfig(
        aimon_api_key=AIMON_API_KEY,
        publish=True,                  # Publish results to AIMon dashboard (optional)
        return_telemetry=True,         # Include detailed iteration telemetry in the response
        return_aimon_summary=True,     # Include a high-level summary of the run
        application_name="api_test",   # Helps identify where this pipeline run originated
    )
    
    # Query and context: simulating a user asking about insurance drug tiers
    user_query = "what are the drug tiers?"
    context = """[SECTION] 📘 BlueShield Rx Policy Addendum: 2023–2025 ... (truncated for brevity)"""
    
    # Instructions: enforce readability, professionalism, and simplification
    user_instructions = [
        "Avoid overly technical or robotic phrasing; keep the tone human and accessible.",
        "Ensure the response is direct and professional, with minimal informal tone.",
        "Translate or simplify technical details from the context into accurate, user-friendly explanations."
    ]
    
    # Run the pipeline
    result = run_reprompting_pipeline(
        user_query=user_query,
        context=context, 
        llm_fn=my_llm,
        reprompting_config=config,
        user_instructions=user_instructions
    )
    
    # Print the outputs for manual inspection
    print("\n=== BEST RESPONSE ===")
    print(result["best_response"])
    print("\n=== TELEMETRY ===")
    print(result.get("telemetry"))
    print("\n=== SUMMARY ===")
    print(result.get("summary"))
    
# --- Entry Point ---
if __name__ == "__main__":
    # Run the test directly if this file is executed
    test_successful_run()