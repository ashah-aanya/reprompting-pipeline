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
    Runs the full re-prompting pipeline.

    Args:
        user_query (str): The query to answer.
        context (Union[str, List]): Context (string, list, or retriever).
        llm_fn (Callable): A function to call the LLM.
        user_instructions (List[str], optional): Additional instructions for the LLM.
        reprompting_config (RepromptingConfig, optional): Pipeline configuration.

    Returns:
        dict: A dictionary containing the best response, and optionally telemetry/caption.
    """

    # Normalize context for the pipeline
    context_str = normalize_context(context)
    config = reprompting_config or RepromptingConfig()
    pipeline = RepromptingPipeline(llm_fn=llm_fn, config=config)
    
    return pipeline.run(
        user_query=user_query,
        context=context_str,
        user_instructions=user_instructions or []
    )