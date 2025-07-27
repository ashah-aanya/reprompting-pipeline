"""
runner.py — This module provides a high-level function (`run_reprompting_pipeline`) 
for running AIMon's iterative re-prompting workflow.

Typical usage:
    from aimon.reprompting_api.runner import run_reprompting_pipeline
    from aimon.reprompting_api.config import RepromptingConfig
    
    def my_llm(prompt: str) -> str:
        # Call your LLM here (OpenAI, Anthropic, etc.)
        return llm_api.generate(prompt)
    
    response = run_reprompting_pipeline(
        user_query="Summarize this document in 3 bullet points.",
        context=["Some context text...", "Another context section..."],
        llm_fn=my_llm,
        user_instructions=["Be concise.", "Use neutral tone."],
        reprompting_config=RepromptingConfig(max_iterations=3)
    )
    print(response["best_response"])

This function is the primary interface for developers and end-users; contributors 
can extend this behavior by modifying `RepromptingPipeline` or `RepromptingConfig`.
"""

from typing import List, Union
from aimon.reprompting_api.utils import normalize_context
from aimon.reprompting_api.pipeline import RepromptingPipeline
from aimon.reprompting_api.config import RepromptingConfig

def run_reprompting_pipeline(
    user_query: str,
    context: Union[str, List],
    llm_fn,
    user_instructions: List[str] = None,
    reprompting_config: RepromptingConfig = None,
) -> dict:
    """
    High-level wrapper for running the full AIMon re-prompting pipeline.

    This function normalizes the context, initializes the pipeline with configuration 
    and the provided LLM function, and executes the iterative re-prompting process. 

    Args:
        user_query (str): The query to answer.
        context (Union[str, List]): Context (string, list, or retriever).
        llm_fn (Callable): A function to call the LLM.
        user_instructions (List[str], optional): Additional instructions for the LLM.
        reprompting_config (RepromptingConfig, optional): Pipeline configuration.

    Returns:
        dict: A structured dictionary containing:
            - "best_response" (str): The final, best LLM response.
            - "telemetry" (list, optional): Iteration-level telemetry if enabled.
            - "caption" (str, optional): A human-readable pipeline summary if enabled.
    """

    # convert context into a normalized string (handles string or document list inputs)
    context_str = normalize_context(context)
    # Use the provided config or fall back to defaults
    config = reprompting_config or RepromptingConfig()
    # initialize the re-prompting pipeline with the LLM function and configuration
    pipeline = RepromptingPipeline(llm_fn=llm_fn, config=config)
    
    return pipeline.run(
        user_query=user_query,
        context=context_str,
        user_instructions=user_instructions or [] # Default to empty list if none provided
    )