from aimon.reprompting_api.config import RepromptingConfig, StopReasons
from aimon.reprompting_api.telemetry import TelemetryLogger
from aimon.reprompting_api.reprompter import Reprompter
from aimon.reprompting_api.utils import toxicity_check, get_failed_instructions_count, get_failed_instructions, get_residual_error_score
from aimon import Detect
import time
import logging
import os
from transformers import AutoTokenizer
import json

logger = logging.getLogger(__name__)

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
        tokenizer (AutoTokenizer): Tokenizer used for estimating prompt/response token counts.
        tokenizer_fn (callable): Function to count tokens in a string. Defaults to HuggingFace tokenizer.
    """
    def __init__(self, llm_fn, config, tokenizer_fn=None):
        """
        Initialize pipeline with LLM callable and RepromptingConfig.
        
        Args:
            llm_fn (callable): Function to call the LLM. Signature: llm_fn(prompt: str) -> str
            config (RepromptingConfig): Configuration object with API keys and limits.
            tokenizer_fn (callable, optional): Custom token counting function. Defaults to internal tokenizer.

        """
        self.llm_fn = llm_fn
        self.config = config or RepromptingConfig()
        
        # Utilities for reprompting, telemetry, and scoring
        self.reprompter = Reprompter()
        self.telemetry = TelemetryLogger()
        
        # Default tokenizer (Mistral model). Used if no custom tokenizer is provided.
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Select token counting function (fallback to default if none provided)
        if tokenizer_fn and callable(tokenizer_fn):
            self.tokenizer_fn = tokenizer_fn
        elif getattr(self.config, "tokenizer_fn", None) and callable(self.config.tokenizer_fn):
            self.tokenizer_fn = self.config.tokenizer_fn
        else:
            logger.warning("No valid tokenizer_fn provided; falling back to default.")
            self.tokenizer_fn = self._default_count_tokens

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
          6. Return the best response (lowest residual error) along with telemetry and a caption if configured.
        
        Args:
            user_query (str): The user's query or instruction.
            context (str): Contextual information to include in the prompt.
            user_instructions (list[str]): Instructions the model must follow.

        Returns:
            dict: 
                {
                    "best_response" (str): Best model response from all iterations.
                    "telemetry" (list, optional): Telemetry for all iterations if enabled.
                    "caption" (str, optional): Summary of the process if enabled.
                }
        """
        iteration_outputs = {} # key: iteration number → dict(response_text, residual_error_score, failed_instructions_count)
        telemetry_data = [] # stores per-iteration telemetry logs
        non_fatal_errors = [] # collect non-breaking errors (e.g., tokenizer failures)
        pipeline_start = time.time()
        iteration_num = 1
        
        # Build the initial prompt
        curr_prompt = self._build_initial_prompt(context, user_query, user_instructions)

        # First LLM call
        try:
            curr_generated_text = self.llm_fn(curr_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._terminate_with_failure(StopReasons.LLM_CALL_FAILED, iteration_num, curr_prompt, telemetry_data, iteration_outputs, pipeline_start)

        # Evaluate response with AIMon
        curr_payload = self._build_aimon_payload(context, user_query, user_instructions, curr_generated_text)
        try:
            curr_result = self._detect_aimon_response(curr_payload)
        except Exception as e:
            logger.error(f"AIMon detect failed: {e}")
            return self._terminate_with_failure(StopReasons.FEEDBACK_MODEL_FAILED, iteration_num, curr_prompt, telemetry_data, iteration_outputs, pipeline_start, generated_text=curr_generated_text)

        # Get scores and detailed feedback on failured instructions
        scores, feedback = self.get_response_feedback(curr_result)
        self._record_iteration_output(iteration_outputs, iteration_num, curr_generated_text, curr_result)

        # Iteratively re-prompt until conditions are met or limits reached
        stop_reason = None
        while True:
            should_stop, stop_reason = self._should_stop_reprompting(curr_result, iteration_num)
            if should_stop:
                break
            
            # Emit telemetry for this iteration
            telemetry_entry = self._emit_iteration_telemetry(
                iteration_num,
                pipeline_start,
                scores,
                feedback,
                curr_result,
                stop_reason or StopReasons.CONTINUE,
                curr_prompt,
                curr_generated_text,
                non_fatal_errors,
            )
            telemetry_data.append(telemetry_entry)

            # Generate corrective prompt
            curr_prompt = self._build_corrective_prompt(curr_payload, curr_result)
            
            # Retry LLM call with corrective prompt
            try:
                curr_generated_text = self.llm_fn(curr_prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                iteration_num += 1
                return self._terminate_with_failure(StopReasons.LLM_CALL_FAILED, iteration_num, curr_prompt, telemetry_data, iteration_outputs, pipeline_start)

            # Re-evaluate the new response
            curr_payload = self._build_aimon_payload(context, user_query, user_instructions, curr_generated_text)
            try:
                curr_result = self._detect_aimon_response(curr_payload)
            except Exception as e:
                logger.error(f"AIMon detect failed: {e}")
                iteration_num += 1
                return self._terminate_with_failure(StopReasons.FEEDBACK_MODEL_FAILED, iteration_num, curr_prompt, telemetry_data, iteration_outputs, pipeline_start, generated_text=curr_generated_text)
            
            # Extract updated scores and feedback
            scores, feedback = self.get_response_feedback(curr_result)
            iteration_num += 1
            self._record_iteration_output(iteration_outputs, iteration_num, curr_generated_text, curr_result)

        # Final telemetry after loop exit
        final_entry = self._emit_iteration_telemetry(
            iteration_num,
            pipeline_start,
            scores,
            feedback,
            curr_result,
            stop_reason or StopReasons.UNKNOWN_ERROR,
            curr_prompt,
            curr_generated_text,
            non_fatal_errors,
        )
        telemetry_data.append(final_entry)

        # Select best response across all iterations
        best_output, best_failed_count = self._select_best_iteration(iteration_outputs)
            
        # Build final response payload
        response = {"best_response": best_output}
        if self.config.return_telemetry:
            response["telemetry"] = telemetry_data
        if self.config.return_aimon_caption:
            response["caption"] = self._gen_summary_caption(iteration_num, best_failed_count)

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
        
        # Append groundedness safeguards to user instructions
        groundedness_instructions = [
            "Ensure that the response does not contain minor or major factual inaccuracies or completely made-up sentences when compared to the context.",
            "Ensure that the response correctly infers implicit information without introducing unsupported facts.",
            "Ensure that the response does not arbitrarily choose between contradictory pieces of information without clarification.",
            "Ensure that the response does not misinterpret or fabricate implicit information.",
            "Ensure that the response does not include hallucinated content due to ambiguity."
        ]

        all_instructions = user_instructions + groundedness_instructions

        # Format multi-section prompt
        return (
            "### CONTEXT ###\n"
            f"{context.strip()}\n\n"
            "### USER QUERY ###\n"
            f"{user_query.strip()}\n\n"
            "### INSTRUCTIONS ###\n"
            + "\n".join(f"- {inst}" for inst in all_instructions)
        )

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
            logger.warning("user_instructions is not a list. Defaulting to empty list.")
            user_instructions = []
        payload = {
            'context': context,
            'user_query': user_query,
            'generated_text': generated_text,
            'instructions': user_instructions,
        }
        return payload

    def _detect_aimon_response(self, payload):
        """
        Evaluate a response using AIMon detectors. Retries up to one additional time if failed instruction explanations are empty.
        Args:
            payload (dict): AIMon input payload.

        Returns:
            object: AIMon detection result object.
        """
        aimon_context = f"{payload['context']}\n\nUser Query:\n{payload['user_query']}"
        aimon_query = f"{payload['user_query']}\n\nInstructions:\n{payload['instructions']}"

        @self.detect
        def run_detection(query, instructions, generated_text, context):
            return query, instructions, generated_text, context

        max_iterations = 2

        for attempt in range(max_iterations):
            _, _, _, _, result = run_detection(
                aimon_query,
                payload['instructions'],
                payload['generated_text'],
                aimon_context
            )

            # Check if explanations for failed instructions are empty
            failed_explanations = []
            try:
                for inst in result.detect_response.instruction_adherence.get("instructions_list", []):
                    if not inst.get("label", True):  # Only failed
                        failed_explanations.append(inst.get("explanation", "").strip())
                for inst in result.detect_response.groundedness.get("instructions_list", []):
                    if not inst.get("label", True):  # Only failed
                        failed_explanations.append(inst.get("explanation", "").strip())
            except Exception as e:
                print(f"⚠️ Error checking failed explanations: {e}")

            if all(failed_explanations) and all(e != "" for e in failed_explanations):
                return result

            if attempt < max_iterations - 1:
                print(f"⚠️ Empty explanation(s) for failed instructions detected. Retrying... (Attempt {attempt+1}/{max_iterations})")
                
        return result

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

    def _should_stop_reprompting(self, result, iteration_num):
        """
        Determine whether to stop re-prompting. 
        
        Stopping conditions:
        - Max iterations reached.
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
            logger.info(f"Reached maximum number of iterations: {iteration_num}")
            return True, StopReasons.MAX_ITERATIONS_REACHED
        
        # Continue if toxicity is detected
        if toxicity_check(result):
            logger.info("toxicity detected. Continuing reprompting.")
            return False, StopReasons.CONTINUE_TOXICITY

        # Continue if there are still failed instructions
        if get_failed_instructions_count(result) > 0:
            logger.info("Detected adherence/groundedness violations. Continuing reprompting.")
            return False, StopReasons.CONTINUE

        # All instructions followed
        logger.info("All instructions followed; stopping reprompting.")
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
    
    def _gen_summary_caption(self, iteration_num, best_failed_count):
        """
        Generate a human-readable summary caption for the pipeline run.

        Args:
            iteration_num (int): Number of iterations performed.
            best_failed_count (int): Number of failed instructions in the best response.

        Returns:
            str: Summary caption.
        """
        caption = (
                f"Re-prompting completed in {iteration_num} iterations "
                f"with {best_failed_count} failed instructions remaining."
            )
        return caption
    
    def _default_count_tokens(self, text, non_fatal_errors):
        """
        Count tokens in a string using the default tokenizer.

        Args:
            text (str): Text to tokenize.

        Returns:
            int: Token count.
        """
        if not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(text))
    
    def count_tokens(self, text, non_fatal_errors):
        """
        Count tokens using the configured tokenizer function.

        Args:
            text (str): Text to tokenize.
            non_fatal_errors (list): List to log tokenizer failures 

        Returns:
            int: Token count.
        """
        try:
            return int(self.tokenizer_fn(text))
        except Exception as e:
            error_msg = f"tokenizer_failed: {e}"
            if error_msg not in non_fatal_errors:
                non_fatal_errors.append(error_msg)
            return 0
    
    def _build_telemetry_entry(
        self,
        iteration,
        cumulative_latency,
        prompt_tokens,
        response_tokens,
        scores,
        feedback,
        residual_error,
        failed_count,
        stop_reason,
        prompt,
        response_text,
        non_fatal_errors,
    ):
        """
        Build a structured telemetry entry for an iteration.

        Args:
            iteration (int): Iteration number.
            cumulative_latency (float): Total latency in milliseconds so far.
            prompt_tokens (int): Token count for the prompt.
            response_tokens (int): Token count for the response.
            scores (dict): Evaluation scores.
            feedback (list): Failed instruction feedback.
            residual_error (float): Residual error score.
            failed_count (int): Number of failed instructions.
            stop_reason (str): Reason for stopping.
            prompt (str): Prompt used for this iteration.
            response_text (str): Model's response.
            non_fatal_errors (list): Non-fatal errors encountered.

        Returns:
            dict: Structured telemetry entry.
        """
        return {
            "iteration": iteration,
            "cumulative_latency_ms": cumulative_latency,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "scores": scores,
            "response_feedback": feedback,
            "residual_error": residual_error,
            "failed_instructions_count": failed_count,
            "stop_reason": stop_reason,
            "prompt": prompt,
            "response_text": response_text,
            "non_fatal_errors": non_fatal_errors or [],
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
        non_fatal_errors
    ):
        """
        Build and emit telemetry for an iteration. Calculates cumulative latency and token counts.

        Args:
            iteration_num (int): Current iteration number.
            pipeline_start (float): Start time of the pipeline (epoch).
            scores (dict): Evaluation scores.
            feedback (list): Failed instruction feedback.
            curr_result (object): AIMon detection result.
            stop_reason (str): Reason for stopping or continuing.
            curr_prompt (str): Prompt used.
            curr_generated_text (str): Model response text.
            non_fatal_errors (list): Collected non-fatal errors.

        Returns:
            dict: The telemetry entry.
        """
        cumulative_latency_ms = (time.time() - pipeline_start) * 1000
        prompt_tokens = self.count_tokens(curr_prompt, non_fatal_errors)
        response_tokens = self.count_tokens(curr_generated_text, non_fatal_errors)
        
        residual_error = get_residual_error_score(curr_result) if curr_result else None
        failed_count = get_failed_instructions_count(curr_result) if curr_result else None
        
        entry = self._build_telemetry_entry(
            iteration_num,
            cumulative_latency_ms,
            prompt_tokens,
            response_tokens,
            scores,
            feedback,
            residual_error,
            failed_count,
            stop_reason,
            curr_prompt,
            curr_generated_text,
            non_fatal_errors,
        )
        try:
            self.telemetry.emit(**entry)
        except Exception as e:
            error_msg = f"telemetry_emit_failed: {e}"
            if error_msg not in entry["non_fatal_errors"]:
                entry["non_fatal_errors"].append(error_msg)
            logger.warning(error_msg)
        return entry
    
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

    def _terminate_with_failure(self, stop_reason, iteration_num, prompt, telemetry_data, iteration_outputs, pipeline_start, generated_text=None):
        """
        Handle a fatal error by recording telemetry and returning a failure response.

        Args:
            stop_reason (str): Reason for failure.
            iteration_num (int): Current iteration.
            prompt (str): Prompt used at failure.
            telemetry_data (list): Collected telemetry entries.
            iteration_outputs (dict): Outputs from completed iterations.
            pipeline_start (float): time of start.
            generated_text (str, optional): Last generated text, if available.

        Returns:
            dict: Failure response with best available output, telemetry, and caption.
        """
        entry = self._emit_iteration_telemetry(
            iteration_num=iteration_num,
            pipeline_start=pipeline_start,
            scores={},
            feedback=[],
            curr_result=None,
            stop_reason=stop_reason,
            curr_prompt=prompt,
            curr_generated_text=generated_text or f"[ERROR: {stop_reason}]",
            non_fatal_errors=[]
        )
        telemetry_data.append(entry)

        # Determine best available output
        best_output = "[ERROR: No valid response]"
        valid = [
            entry for entry in iteration_outputs.values()
            if isinstance(entry.get("residual_error_score"), (int, float))
        ]
        if valid:
            # Prefer the response with the lowest residual error
            best_iter = min(valid, key=lambda x: x["residual_error_score"])
            best_output = best_iter["response_text"]
        elif generated_text:
            # Fall back to the last generated text if no valid previous iterations exist
            best_output = generated_text

        return {
            "best_response": best_output,
            "telemetry": telemetry_data if self.config.return_telemetry else None,
            "caption": (
                f"Re-prompting terminated early due to {stop_reason}."
                if self.config.return_aimon_caption else None
            )
        }