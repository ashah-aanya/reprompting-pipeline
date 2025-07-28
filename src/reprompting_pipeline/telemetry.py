import json
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path

class TelemetryLogger:
    """
    A lightweight logger for recording telemetry events during re-prompting pipeline execution.

    Each telemetry record captures:
      - Timing and token usage for the iteration.
      - Evaluation scores and failed instruction feedback.
      - The LLM's response, prompt, and any non-fatal errors.
      - The stop reason (e.g., max iterations reached, all instructions adhered).
      - A unique session identifier for grouping logs across runs.

    Telemetry is written as JSON Lines (JSONL), with one record per line, to support easy parsing.
    """
    def __init__(self, log_path="telemetry_log.jsonl"):
        """
        Initialize a TelemetryLogger.

        Args:
            log_path (str): Path to the JSONL file where telemetry entries are appended.
                            Directories will be created if they do not exist.
        """
        self.session_id = str(uuid.uuid4())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        iteration: int,
        cumulative_latency_ms: float,
        scores: dict,
        response_feedback: dict,
        residual_error: int,
        failed_instructions_count: int,
        stop_reason: str,
        response_text: str,
        prompt: str = "",
    ):
        """
        Emit a single telemetry entry.

        Args:
            iteration (int): The iteration number of the pipeline (starts at 1).
            cumulative_latency_ms (float): Total latency from pipeline start (ms).
            scores (dict): Evaluation scores (e.g., groundedness, instruction adherence).
            response_feedback (dict): Detailed feedback with follow probability and explanations for failed instructions.
            residual_error (int): Residual error score for the iteration. See utils.py for calculation.
            failed_instructions_count (int): Count of instructions not followed.
            stop_reason (str): Reason for stopping or continuing (e.g., "max_iterations_reached").
            response_text (str): The raw text response from the LLM.
            prompt (str): The prompt text used for this iteration.

        Behavior:
            - Appends the telemetry entry to the configured JSONL file.
            - Adds a UTC timestamp and session ID for traceability.
        """
        telemetry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "iteration": iteration,
            "cumulative_latency_ms": cumulative_latency_ms,
            "scores": scores,
            "response_feedback": response_feedback,
            "residual_error": residual_error,
            "failed_instructions_count": failed_instructions_count,
            "stop_reason": stop_reason,
            "prompt": prompt,
            "response_text": response_text,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(telemetry) + "\n")
