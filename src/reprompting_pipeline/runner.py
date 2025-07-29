"""
runner.py — This module provides a high-level function (`run_reprompting_pipeline`) 
for running AIMon's iterative re-prompting workflow.
This function is the primary interface for developers and end-users; contributors 
can extend this behavior by modifying `RepromptingPipeline` or `RepromptingConfig`.
"""

from typing import List, Optional
from aimon.reprompting_api.pipeline import RepromptingPipeline
from aimon.reprompting_api.config import RepromptingConfig

def run_reprompting_pipeline(
    llm_fn,
    user_query: str,
    context:str = None,
    system_prompt: str = None,
    user_instructions: List[str] = None,
    reprompting_config: RepromptingConfig = None,
) -> dict:
    """
    High-level wrapper for running the full AIMon re-prompting pipeline.

    This function normalizes the context, initializes the pipeline with configuration 
    and the provided LLM function, and executes the iterative re-prompting process. 

    Args:
        user_query (str): The query to answer.
        context (str): Context in str form.
        llm_fn (Callable[[str, Optional[str], Optional[str]],str]): A function to call the LLM that takes in a prompt and optionally
        accepts a user_query and context
        user_instructions (List[str], optional): Instructions to validate adherence to.
        reprompting_config (RepromptingConfig, optional): Pipeline configuration.

    Returns:
        dict: A structured dictionary containing:
            - "best_response" (str): The final, best LLM response.
            - "telemetry" (list, optional): Iteration-level telemetry if enabled.
            - "summary" (str, optional): A human-readable pipeline summary if enabled.
    """

    # Use the provided config or fall back to defaults
    config = reprompting_config or RepromptingConfig()
    
    # validate llm_fn
    if not callable(llm_fn):
        raise TypeError("llm_fn must be a callable that returns a string.")

    if not user_query or not isinstance(user_query, str):
        raise ValueError("user_query must be a non-empty string.")

    # initialize the re-prompting pipeline with the LLM function and configuration
    pipeline = RepromptingPipeline(llm_fn=llm_fn, config=config)
    
    return pipeline.run(
        user_query=user_query,
        context=context,
        user_instructions=user_instructions or [], # Default to empty list if none provided
        system_prompt=system_prompt
    )