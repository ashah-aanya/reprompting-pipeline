import os
from dataclasses import dataclass
from typing import Optional

# Default Together AI model used as the judge. Small instruct models are cheap and fast; the right
# choice is measured with evals/run_judge_eval.py, so treat this as a starting point, not a verdict.
DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class StopReasons:
    ALL_INSTRUCTIONS_ADHERED = "all_instructions_adhered"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    CONTINUE = "instructions_failed_continue_reprompting"
    CONTINUE_TOXICITY = "toxicity_detect_continue_reprompting"

    ## limits
    LATENCY_LIMIT_EXCEEDED = "latency_limit_exceeded"

    ##errors
    REPROMPTING_FAILED = "reprompting_failed"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RepromptingConfig:
    """
    Configuration for the automated re-prompting pipeline.

    Attributes:
        max_iterations (int): Maximum number of LLM calls (1 initial + N re-prompts). Must be >= 1.
        together_api_key (Optional[str]): Together AI key used by the judge. Defaults to the
            TOGETHER_API_KEY env var, which is only read when the default judge is created, so a
            config can be built (and the package imported) without a key.
        judge_model (str): Together AI model that audits each response.
        judge_max_tokens (int): Max tokens the judge may generate for its JSON report.
        judge_temperature (float): Judge sampling temperature. 0 keeps audits repeatable.
        return_telemetry (bool): Whether to include per-iteration telemetry in the result.
        return_summary (bool): Whether to include a short caption (e.g. "2 iterations, 0 failed instructions remaining").
        latency_limit_ms (Optional[int]): Stop re-prompting once 75% of this budget (ms) is used. None = no limit.
        user_model_max_retries (int): Attempts (with exponential backoff) for each llm_fn call. Defaults to 2.
        feedback_model_max_retries (int): Attempts (with exponential backoff) for each judge call. Defaults to 2.
    """
    max_iterations: int = 2
    together_api_key: Optional[str] = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_max_tokens: int = 1500
    judge_temperature: float = 0.0
    return_telemetry: bool = False
    return_summary: bool = False
    latency_limit_ms: Optional[int] = None
    user_model_max_retries: int = 2
    feedback_model_max_retries: int = 2

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be greater than 0")

    def resolve_api_key(self) -> str:
        """Return the Together key from the config or the environment, or raise if neither is set."""
        key = self.together_api_key or os.getenv("TOGETHER_API_KEY")
        if not key:
            raise ValueError(
                "No Together AI API key found. Pass together_api_key to RepromptingConfig "
                "or set the TOGETHER_API_KEY environment variable."
            )
        return key
