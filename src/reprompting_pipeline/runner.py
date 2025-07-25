from typing import List, Union
from pipeline import RepromptingPipeline
from config import RepromptingConfig

def normalize_context(context: Union[str, List]) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in context])
    raise ValueError("Unsupported context type.")

def run_reprompting_pipeline(
    user_query: str,
    context: Union[str, List],
    llm_fn,
    user_instructions: List[str] = None,
    reprompting_config: RepromptingConfig = None
) -> dict:
    config = reprompting_config or RepromptingConfig()
    context_str = normalize_context(context)
    pipeline = RepromptingPipeline(llm_fn=llm_fn, config=config) 
    result = pipeline.run(
        user_query=user_query,
        context=context_str,
        llm_fn=llm_fn,
        user_instructions=user_instructions or []
    )
    return result
