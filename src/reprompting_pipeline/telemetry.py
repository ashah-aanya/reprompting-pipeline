import json
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path

class TelemetryLogger:
    def __init__(self, log_path="telemetry_log.jsonl"):
        self.session_id = str(uuid.uuid4())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        iteration: int,
        cumulative_latency_ms: float,
        prompt_tokens: int,
        response_tokens: int,
        scores: dict,
        response_feedback: dict,
        residual_error: int,
        failed_instructions_count: int,
        stop_reason: str,
        response_text: str,
        prompt: str = "",
        non_fatal_errors: Optional[List[str]] = None,
    ):
        telemetry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "iteration": iteration,
            "cumulative_latency_ms": cumulative_latency_ms,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "scores": scores,
            "response_feedback": response_feedback,
            "residual_error": residual_error,
            "failed_instructions_count": failed_instructions_count,
            "stop_reason": stop_reason,
            "prompt": prompt,
            "response_text": response_text,
            "non_fatal_errors": non_fatal_errors or []
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(telemetry) + "\n")
