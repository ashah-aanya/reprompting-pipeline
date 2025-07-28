import os
from typing import Optional
from dataclasses import dataclass
import random
import string

def generate_random_string(length):
  """Generates a random string of letters and digits."""
  characters = string.ascii_letters + string.digits
  return ''.join(random.choice(characters) for i in range(length))

class StopReasons:
    ALL_INSTRUCTIONS_ADHERED = "all_instructions_adhered"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    CONTINUE = "instructions_failed_continue_reprompting" 
    CONTINUE_TOXICITY = "toxicity_detect_continue_reprompting"
    
    ## limits
    LATENCY_LIMIT_EXCEEDED = "latency_limit_exceeded"
    
    ##errors
    LLM_CALL_FAILED = "llm_call_failed"
    FEEDBACK_MODEL_FAILED = "feedback_model_failed"
    REPROMPTING_FAILED = "reprompting_failed"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RepromptingConfig:
    """
    Configuration for the automated re-prompting pipeline.

    Attributes:
        publish (bool): Whether to publish results to app.aimon.ai.
        max_iterations (int): Maximum number of re-prompting iterations (1 initial + N retries).
        aimon_api_key (Optional[str]): API key for AIMon integration. Defaults to "AIMON_API_KEY" env var.
        model_name (Optional[str]): Model identifier for telemetry. Defaults to "aimon-react-model-{rand}".
        application_name (Optional[str]): Application identifier for telemetry. Defaults to "aimon-react-application-{rand}".
        return_telemetry (bool): Whether to include per-iteration telemetry in the response.
        return_aimon_summary(bool): Whether to include a human-readable caption summarizing re-prompting. (e.g.: 2 iterations, 0 failed instructions)
        latency_limit_ms (Optional[int]): Maximum cumulative latency (ms) before aborting. None = no limit.
    """
    publish: bool = False
    max_iterations: int = 2
    aimon_api_key: Optional[str] = os.getenv("AIMON_API_KEY")
    model_name: Optional[str] = "aimon-react-model-" + generate_random_string(5)
    application_name: Optional[str] = "aimon-react-application-" + generate_random_string(5)
    return_telemetry: bool = False
    return_aimon_summary: bool = False
    latency_limit_ms: Optional[int] = None
    
    