from aimon.reprompting_api.config import RepromptingConfig, StopReasons
from aimon.reprompting_api.telemetry import TelemetryLogger
from aimon.reprompting_api.reprompter import Reprompter
from aimon.reprompting_api.utils import toxicity_check, get_failed_instructions_count, get_failed_instructions, get_residual_error_score
from aimon import Detect
import time
import random

class RepromptingPipeline:
    """
    A pipeline for iterative re-prompting of LLM responses using AIMon evaluation.
    
    This pipeline orchestrates:
      - Initial prompt generation for a given query, context, and user instructions.
      - Interaction with a black-box LLM to generate responses.
      - Evaluation of responses using AIMon detectors (instruction adherence, groundedness, toxicity).
      - Iterative corrective re-prompting until stopping conditions are met.
      - Collection and emission of telemetry for all iterations.
      
    Attributes:
        llm_fn (callable): Function to call the LLM. Must accept a string prompt and return a string response.
        config (RepromptingConfig): Configuration object with API keys and iteration limits.
        reprompter (Reprompter): Utility for generating corrective prompts based on evaluation feedback.
        telemetry (TelemetryLogger): Logger for capturing telemetry data.
        detect (Detect): AIMon detection client for evaluating model responses.
    """
    def __init__(self, llm_fn, config):
        """
        Initialize pipeline with LLM callable and RepromptingConfig.
        
        Args:
            llm_fn (callable): Function to call the LLM. Signature: llm_fn(prompt: str) -> str
            config (RepromptingConfig): Configuration object with API keys and limits.

        """
        self.llm_fn = llm_fn
        self.config = config or RepromptingConfig()
        
        # Utilities for reprompting, telemetry, and scoring
        self.reprompter = Reprompter()
        self.telemetry = TelemetryLogger()

        # Initialize AIMon Detect for response evaluation
        self.detect = Detect(
            values_returned=['user_query', 'instructions', 'generated_text', 'context'],
            config={
                "instruction_adherence": {
                    "detector_name": "default",
                    "explain": True,
                    "extract_from_system": False
                },
                "groundedness": {
                    "detector_name": "default",
                    "explain": True
                },
                "toxicity": {
                    "detector_name": "default",
                    "explain": True
                }
            },
            api_key=self.config.aimon_api_key,
            application_name = self.config.application_name,
            model_name = self.config.model_name,
            publish=self.config.publish
        )
        
    def run(self, user_query, context, user_instructions):
        """
        Execute the full re-prompting pipeline.
        
        Process:
          1. Build an initial prompt with query, context, and instructions.
          2. Call the LLM to generate a response.
          3. Evaluate the response with AIMon detectors.
          4. If violations are found, iteratively generate corrective prompts and re-prompt the LLM.
          5. Stop when all instructions are followed or iteration limits are reached.
          6. Return the best response (lowest residual error) along with telemetry and a summary if configured.
        
        Args:
            user_query (str): The user's query or instruction.
            context (str): Contextual information to include in the prompt.
            user_instructions (list[str]): Instructions the model must follow.

        Returns:
            dict: 
                {
                    "best_response" (str): Best model response from all iterations.
                    "telemetry" (list, optional): Telemetry for all iterations if enabled.
                    "summary" (str, optional): Summary of the process if enabled.
                }
        """
        iteration_outputs = {} # key: iteration number → dict(response_text, residual_error_score, failed_instructions_count)
        pipeline_start = time.time()
        iteration_num = 1
        
        # Build the initial prompt
        curr_prompt = self._build_initial_prompt(context, user_query, user_instructions)

        # First LLM call
        curr_generated_text = self._call_llm(curr_prompt,self.config.user_model_max_retries)
        
        # Evaluate response with AIMon
        curr_payload = self._build_aimon_payload(context, user_query, user_instructions, curr_generated_text)
        curr_result = self._detect_aimon_response(curr_payload, self.config.feedback_model_max_retries)
        
        # Get scores and detailed feedback on failured instructions
        scores, feedback = self.get_response_feedback(curr_result)
        self._record_iteration_output(iteration_outputs, iteration_num, curr_generated_text, curr_result)

        # Iteratively re-prompt until conditions are met or limits reached
        stop_reason = None
        while True:
            should_stop, stop_reason = self._should_stop_reprompting(curr_result, iteration_num, pipeline_start)
            if should_stop:
                break
            
            # Emit telemetry for this iteration
            self._emit_iteration_telemetry(
                iteration_num,
                pipeline_start,
                scores,
                feedback,
                curr_result,
                stop_reason or StopReasons.CONTINUE,
                curr_prompt,
                curr_generated_text,
            )

            # Generate corrective prompt
            curr_prompt = self._build_corrective_prompt(curr_payload, curr_result)
            
            # Retry LLM call with corrective prompt
            curr_generated_text = self._call_llm(curr_prompt,self.config.user_model_max_retries)
            # Re-evaluate the new response
            curr_payload = self._build_aimon_payload(context, user_query, user_instructions, curr_generated_text)
            curr_result = self._detect_aimon_response(curr_payload, self.config.feedback_model_max_retries)

            # Extract updated scores and feedback
            scores, feedback = self.get_response_feedback(curr_result)
            iteration_num += 1
            self._record_iteration_output(iteration_outputs, iteration_num, curr_generated_text, curr_result)

        # Final telemetry after loop exit
        self._emit_iteration_telemetry(
            iteration_num,
            pipeline_start,
            scores,
            feedback,
            curr_result,
            stop_reason or StopReasons.UNKNOWN_ERROR,
            curr_prompt,
            curr_generated_text,
        )

        # Select best response across all iterations
        best_output, best_failed_count = self._select_best_iteration(iteration_outputs)
            
        # Build final response payload
        response = {"best_response": best_output}
        if self.config.return_telemetry:
            response["telemetry"] = self.telemetry.get_all()
        if self.config.return_aimon_summary:
            response["summary"] = self._gen_summary(iteration_num, best_failed_count)

        return response

    def _build_initial_prompt(self, context, user_query, user_instructions):
        """
        Builds initial LLM prompt by clearly delineating context, query, and combined instructions.
        
        Args:
            context (str): Context for the query.
            user_query (str): The user's query.
            user_instructions (list[str]): Instructions for the model.
            
        Returns:
            str: A formatted prompt including context, query, and instructions.
        """
        if not isinstance(user_instructions, list):
            user_instructions = []
        
        # Format the prompt with clear sections
        prompt = (
            f"Context:\n{context}\n\n"
            f"User Query:\n{user_query}\n\n"
            f"Instructions:\n" + "\n".join(user_instructions) + "\n\n"
            "Generate a response that adheres to the instructions and is grounded in the context."
        )
        return prompt
        

    def _build_aimon_payload(self, context, user_query, user_instructions, generated_text):
        """
        Constructs AIMon input payload.

        Args:
            context (str): Context for the LLM.
            user_query (str): The user's query.
            user_instructions (list[str]): Instructions for the model.
            generated_text (str): The model's generated response.

        Returns:
            dict: Payload for AIMon evaluation.
        """
        if not isinstance(user_instructions, list):
            user_instructions = []
        payload = {
            'context': context,
            'user_query': user_query,
            'generated_text': generated_text,
            'instructions': user_instructions,
        }
        return payload

    def _call_llm(self, prompt, max_attempts, base_delay=1):
        """
        Calls the LLM with exponential backoff. Retries if the LLM call fails
        OR returns a non-string value. Raises an exception if all retries fail.

        Args:
            prompt (str): Prompt for the LLM.
            max_attempts (int): Max retry attempts.
            base_delay (float): Initial delay in seconds before backoff.

        Returns:
            str: LLM response text.
        """
        for attempt in range(max_attempts):
            try:
                result = self.llm_fn(prompt)
                if not isinstance(result, str):
                    raise TypeError(f"LLM returned non-string type: {type(result).__name__}, value: {result}")
                return result
            except Exception as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(wait_time)
        raise RuntimeError("LLM call failed or returned invalid type after maximum retries.")
    
    def _detect_aimon_response(self, payload, max_attempts, base_delay=1):
        """
        Calls AIMon Detect with exponential backoff and returns the detection result.

        This method wraps the AIMon evaluation call, retrying if it fails due to transient 
        errors (e.g., network issues or temporary service unavailability). It retries up to 
        `max_attempts` times with exponential backoff before raising a RuntimeError.

        Args:
            payload (dict): A dictionary containing 'context', 'user_query', 
                            'instructions', and 'generated_text' for evaluation.
            max_attempts (int): Maximum number of retry attempts.
            base_delay (float): Initial delay in seconds before exponential backoff.

        Returns:
            object: The AIMon detection result containing evaluation scores and feedback.

        Raises:
            RuntimeError: If AIMon Detect fails after all retry attempts.
        """
        aimon_context = f"{payload['context']}\n\nUser Query:\n{payload['user_query']}"
        aimon_query = f"{payload['user_query']}\n\nInstructions:\n{payload['instructions']}"
        
        @self.detect
        def run_detection(query, instructions, generated_text, context):
            return query, instructions, generated_text, context

        for attempt in range(max_attempts):
            try:
                _, _, _, _, result = run_detection(
                    aimon_query,
                    payload['instructions'],
                    payload['generated_text'],
                    aimon_context
                )
                return result
            except Exception as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(wait_time)
        raise RuntimeError("AIMon detect call failed after maximum retries.")

    def get_response_feedback(self, result):
            """
            Extract groundedness and instruction adherence scores and failed instructions.
            
            Args:
                result (object): AIMon detection result.

            Returns:
                tuple: (scores (dict), failed_instructions (list))
            """
            scores = {
                "groundedness": result.detect_response.groundedness.get("score", 0.0),
                "instruction_adherence": result.detect_response.instruction_adherence.get("score", 0.0)
            }
            feedback = get_failed_instructions(result)
            return scores, feedback
    
    def _build_corrective_prompt(self, payload, result):
        """
        Generate a corrective prompt using AIMon evaluation results.

         Args:
            payload (dict): AIMon input payload.
            result (object): AIMon detection result.

        Returns:
            str: A corrective prompt for re-prompting the LLM.
        """
        return self.reprompter.create_corrective_prompt(result, payload)

    def _should_stop_reprompting(self, result, iteration_num, pipeline_start):
        """
        Determine whether to stop re-prompting. 
        
        Stopping conditions:
        - Max iterations reached.
        - Latency budget 75% depleted
        - All instructions are adhered to.
        - Otherwise, continue if violations or toxicity remain.
        
        Args:
            result (object): AIMon detection result.
            iteration_num (int): Current iteration number.
            
        Returns:
            tuple: 
                (should_stop (bool), stop_reason (str or None))
        """
        # Max iterations reached
        if iteration_num >= self.config.max_iterations:
            return True, StopReasons.MAX_ITERATIONS_REACHED
        
        latency_limit_ms = self.config.latency_limit_ms
        if latency_limit_ms is not None:
            cumulative_latency = self._get_cumulative_latency(pipeline_start)
            if cumulative_latency > ((0.75) * latency_limit_ms):
                return True, StopReasons.LATENCY_LIMIT_EXCEEDED
        
        # Continue if toxicity is detected
        if toxicity_check(result):
            return False, StopReasons.CONTINUE_TOXICITY

        # Continue if there are still failed instructions
        if get_failed_instructions_count(result) > 0:
            return False, StopReasons.CONTINUE

        # All instructions followed
        return True, StopReasons.ALL_INSTRUCTIONS_ADHERED

    def _select_best_iteration(self, iteration_outputs):
        """
        Selects the best iteration based on the lowest residual error score.

        Args:
            iteration_outputs (dict): Mapping of iteration_num -> iteration data.

        Returns:
            tuple: (best_output (str), best_failed_count (int))
        """
        valid_iterations = [
            entry for entry in iteration_outputs.values()
            if isinstance(entry.get("residual_error_score"), (int, float))
        ]
        if not valid_iterations:
            return "[ERROR: No valid response]", None

        best_iteration = min(valid_iterations, key=lambda x: x["residual_error_score"])
        return best_iteration["response_text"], best_iteration["failed_instructions_count"]
    
    def _gen_summary(self, iteration_num, best_failed_count):
        """
        Generate a human-readable summary for the pipeline run.
            e.g.: "2 iterations, 0 failed instructions remaining"

        Args:
            iteration_num (int): Number of iterations performed.
            best_failed_count (int): Number of failed instructions in the best response.

        Returns:
            str: Summary.
        """
        iteration_word = "iteration" if iteration_num == 1 else "iterations"
        summary = f"{iteration_num} {iteration_word}, {best_failed_count} failed instructions remaining"
        return summary
    
    def _build_telemetry_entry(
        self,
        iteration,
        cumulative_latency,
        scores,
        feedback,
        residual_error,
        failed_count,
        stop_reason,
        prompt,
        response_text,
    ):
        """
        Build a structured telemetry entry for an iteration.

        Args:
            iteration (int): Iteration number.
            cumulative_latency (float): Total latency in milliseconds so far.
            scores (dict): Evaluation scores.
            feedback (list): Failed instruction feedback.
            residual_error (float): Residual error score.
            failed_count (int): Number of failed instructions.
            stop_reason (str): Reason for stopping.
            prompt (str): Prompt used for this iteration.
            response_text (str): Model's response.

        Returns:
            dict: Structured telemetry entry.
        """
        return {
            "iteration": iteration,
            "cumulative_latency_ms": cumulative_latency,
            "scores": scores,
            "response_feedback": feedback,
            "residual_error": residual_error,
            "failed_instructions_count": failed_count,
            "stop_reason": stop_reason,
            "prompt": prompt,
            "response_text": response_text,
        }

    def _emit_iteration_telemetry(
        self,
        iteration_num,
        pipeline_start,
        scores,
        feedback,
        curr_result,
        stop_reason,
        curr_prompt,
        curr_generated_text,
    ):
        """
        Build and emit telemetry for an iteration. Calculates cumulative latency.

        Args:
            iteration_num (int): Current iteration number.
            pipeline_start (float): Start time of the pipeline (epoch).
            scores (dict): Evaluation scores.
            feedback (list): Failed instruction feedback.
            curr_result (object): AIMon detection result.
            stop_reason (str): Reason for stopping or continuing.
            curr_prompt (str): Prompt used.
            curr_generated_text (str): Model response text.

        Returns:
            dict: The telemetry entry.
        """
        cumulative_latency_ms = self._get_cumulative_latency(pipeline_start)
        
        residual_error = get_residual_error_score(curr_result) if curr_result else None
        failed_count = get_failed_instructions_count(curr_result) if curr_result else None
        
        entry = self._build_telemetry_entry(
            iteration_num,
            cumulative_latency_ms,
            scores,
            feedback,
            residual_error,
            failed_count,
            stop_reason,
            curr_prompt,
            curr_generated_text,
        )
        try:
            self.telemetry.emit(**entry)
        except Exception as e:
            # Log error silently without breaking the pipeline
            print(f"[Warning] Telemetry emission failed: {e}")
        return entry
    
    def _get_cumulative_latency(self, pipeline_start):
        """
        Calculate cumulative latency since pipeline start.

        Args:
            pipeline_start (float): Start time of the pipeline (epoch).

        Returns:
            float: Cumulative latency in milliseconds.
        """
        return (time.time() - pipeline_start) * 1000
    
    def _record_iteration_output(self, iteration_outputs, iteration_num, generated_text, result):
        """
        Record iteration outputs for later selection of the best response.

        Args:
            iteration_outputs (dict): Stores outputs per iteration.
            iteration_num (int): Current iteration number.
            generated_text (str): Model's generated response.
            result (object): AIMon detection result.

        Returns:
            None
        """
        iteration_outputs[iteration_num] = {
            "response_text": generated_text,
            "residual_error_score": get_residual_error_score(result),
            "failed_instructions_count": get_failed_instructions_count(result)
        }