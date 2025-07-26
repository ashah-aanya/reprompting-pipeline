from aimon.reprompting_api.config import RepromptingConfig, StopReasons
from aimon.reprompting_api.telemetry import TelemetryLogger
from aimon.reprompting_api.reprompter import Reprompter
from aimon.reprompting_api.utils import toxicity_check, get_failed_instructions_count, get_failed_instructions, get_residual_error_score
from aimon import Detect
import time
import logging
import os
from transformers import AutoTokenizer

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
        
        self.reprompter = Reprompter()
        self.telemetry = TelemetryLogger()
        
        self.config.return_telemetry = getattr(self.config, 'return_telemetry', False)
        self.config.return_aimon_caption = getattr(self.config, 'return_aimon_caption', False)

        # AIMon Detect configuration
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
    def run(self, user_query, context, user_instructions):
        """
        Execute the full re-prompting pipeline.
        
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
        # key: iteration number, value: dict with output and error
        iteration_outputs = {}
        telemetry_data = []
        non_fatal_errors = []
        pipeline_start = time.time()
        cumulative_latency_ms = 0.0
        iteration_num = 1
        
        curr_prompt = self.gen_prompt(context, user_query, user_instructions)
        
        try:
            curr_generated_text = self.llm_fn(curr_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            cumulative_latency_ms = (time.time() - pipeline_start) * 1000
            return self.terminate_with_failure(
                StopReasons.LLM_CALL_FAILED,
                iteration_num,
                curr_prompt,
                telemetry_data,
                iteration_outputs,
                cumulative_latency_ms,
            )
    
        logger.info("=== Prompt ===")
        logger.info(curr_prompt)
        logger.info("\n=== Generated Response ===")
        logger.info(curr_generated_text)

        # Create AIMon payload and detect response
        curr_payload = self.create_payload(context, user_query, user_instructions, curr_generated_text)
        ## evaluate response on hallucination/groundedness, instruction adherence, and toxicity
        try:
            curr_result = self.detect_aimon_response(curr_payload)
        except Exception as e:
            logger.error(f"AIMon detect failed: {e}")
            cumulative_latency_ms = (time.time() - pipeline_start) * 1000
            return self.terminate_with_failure(
                StopReasons.FEEDBACK_MODEL_FAILED,
                iteration_num,
                curr_prompt,
                telemetry_data,
                iteration_outputs,
                cumulative_latency_ms,
            )
        # Placeholder for token counts (update if available)
        prompt_tokens = 0
        response_tokens = 0
        #prompt_tokens = self.count_tokens(curr_prompt)
        #response_tokens = self.count_tokens(curr_generated_text)
        
        # Scores and errors for telemetry
        scores, feedback = self.get_response_feedback(curr_result)

        ## update map with current iteration
        iteration_outputs[iteration_num] = {
            "response_text": curr_generated_text,
            "residual_error_score": get_residual_error_score(curr_result),
            "failed_instructions_count": get_failed_instructions_count(curr_result)
        }
        
        stop_reason = None
        
        while True:
            
            should_stop, stop_reason = self.end_reprom(curr_result, iteration_num)
            if should_stop:
                break
            
            cumulative_latency_ms = (time.time() - pipeline_start) * 1000

            telemetry_entry = self.build_telemetry_entry(
                iteration_num,
                cumulative_latency_ms,
                prompt_tokens,
                response_tokens,
                scores,
                feedback,
                get_residual_error_score(curr_result),
                get_failed_instructions_count(curr_result),
                StopReasons.CONTINUE,
                curr_prompt,
                curr_generated_text,
                non_fatal_errors,
            )
            telemetry_data.append(telemetry_entry)
            self.telemetry.emit(**telemetry_entry)
            
            logger.info(f"Iteration Num = {iteration_num}")

            # Generate corrective prompt
            curr_prompt = self.gen_reprompt(curr_payload, curr_result)
            logger.info("=== Corrective Prompt ===")
            logger.info(curr_prompt)
            # retrieve new response from LLM to the corrective prompt
            try:
                curr_generated_text = self.llm_fn(curr_prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                cumulative_latency_ms = (time.time() - pipeline_start) * 1000
                return self.terminate_with_failure(
                    StopReasons.LLM_CALL_FAILED,
                    iteration_num,
                    curr_prompt,
                    telemetry_data,
                    iteration_outputs,
                    cumulative_latency_ms,
                )
            logger.info("\n=== Generated Response ===")
            logger.info(curr_generated_text)
            # Create new AIMon payload with the updated prompt and response
            # Note: This assumes the context remains the same throughout iterations
            curr_payload = self.create_payload(context, user_query, user_instructions, curr_generated_text)

            ## evaluates the new response on hallucination, instruction adherence, and toxicity
            try:
                curr_result = self.detect_aimon_response(curr_payload)
            except Exception as e:
                logger.error(f"AIMon detect failed: {e}")
                cumulative_latency_ms = (time.time() - pipeline_start) * 1000
                return self.terminate_with_failure(
                    StopReasons.FEEDBACK_MODEL_FAILED,
                    iteration_num,
                    curr_prompt,
                    telemetry_data,
                    iteration_outputs,
                    cumulative_latency_ms,
                )   
            logger.info("=== AIMon Result ===")
            logger.info(curr_result)

            # Real token counts if available
            prompt_tokens = 0
            response_tokens = 0
            #prompt_tokens = self.count_tokens(curr_prompt)
            #response_tokens = self.count_tokens(curr_generated_text)

            # creates score and error vectors for the current result to store in telemetry
            scores, feedback = self.get_response_feedback(curr_result)

            iteration_num += 1
            iteration_outputs[iteration_num] = {
                "response_text": curr_generated_text,
                "residual_error_score": get_residual_error_score(curr_result),
                "failed_instructions_count": get_failed_instructions_count(curr_result)
            }
            
        cumulative_latency_ms = (time.time() - pipeline_start) * 1000
        final_entry = self.build_telemetry_entry(
            iteration_num,
            cumulative_latency_ms,
            prompt_tokens,
            response_tokens,
            scores,
            feedback,
            get_residual_error_score(curr_result),
            get_failed_instructions_count(curr_result),
            stop_reason or StopReasons.UNKNOWN_ERROR,
            curr_prompt,
            curr_generated_text,
            non_fatal_errors,
        )
        telemetry_data.append(final_entry)
        self.telemetry.emit(**final_entry)

        valid_iterations = [
            entry for entry in iteration_outputs.values() if isinstance(entry.get("residual_error_score"), (int, float))
        ]
        if valid_iterations:
            best_iteration = min(valid_iterations, key=lambda x: x["residual_error_score"])
            best_output = best_iteration["response_text"]
            best_failed_count = best_iteration["failed_instructions_count"]
        else:
            best_output = "[ERROR: No valid response]"
            best_failed_count = None
            
        ## building reponse
        response = {"best_response": best_output}
        if self.config.return_telemetry:
            response["telemetry"] = telemetry_data
        if self.config.return_aimon_caption:
            response["caption"] = self.gen_caption(iteration_num, best_failed_count)

        return response

    def gen_prompt(self, context, user_query, user_instructions):
        """
        Builds initial LLM prompt by clearly delineating context, query, and combined instructions.
        
        Args:
            context (str): Context for the query.
            user_query (str): The user's query.
            user_instructions (list[str]): Instructions for the model.
            
        Returns:
            str: A formatted prompt including context, query, and instructions.
        """
        
        ## improve this to not be hardcoded
        groundedness_instructions = [
            "Ensure that the response does not contain minor or major factual inaccuracies or completely made-up sentences when compared to the context.",
            "Ensure that the response correctly infers implicit information without introducing unsupported facts.",
            "Ensure that the response does not arbitrarily choose between contradictory pieces of information without clarification.",
            "Ensure that the response does not misinterpret or fabricate implicit information.",
            "Ensure that the response does not include hallucinated content due to ambiguity."
        ]

        all_instructions = user_instructions + groundedness_instructions

        return (
            "### CONTEXT ###\n"
            f"{context.strip()}\n\n"
            "### USER QUERY ###\n"
            f"{user_query.strip()}\n\n"
            "### INSTRUCTIONS ###\n"
            + "\n".join(f"- {inst}" for inst in all_instructions)
        )

    def create_payload(self, context, user_query, user_instructions, generated_text):
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

    def detect_aimon_response(self, payload):
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

            # Only check failed instructions
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

            # If all failed explanations are non-empty, we're done
            if all(failed_explanations) and all(e != "" for e in failed_explanations):
                return result

            if attempt < max_iterations - 1:
                print(f"⚠️ Empty explanation(s) for failed instructions detected. Retrying... (Attempt {attempt+1}/{max_iterations})")

        return result  # Return result from last attempt

    def gen_reprompt(self, payload, result):
        """
        Generate a corrective prompt using AIMon evaluation results.

         Args:
            payload (dict): AIMon input payload.
            result (object): AIMon detection result.

        Returns:
            str: A corrective prompt for re-prompting the LLM.
        """
        return self.reprompter.create_corrective_prompt(result, payload)

    ## latency time-out should be added
    def end_reprom(self, result, iteration_num):
        """
        Determine whether to stop re-prompting. 
        
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

        # Continue if there are still failed instructions
        if get_failed_instructions_count(result) > 0:
            logger.info("Detected adherence/groundedness violations. Continuing reprompting.")
            return False, None

        # Continue if toxicity is detected
        if toxicity_check(result):
            logger.info("toxicity detected. Continuing reprompting.")
            return False, None

        # All instructions followed
        logger.info("All instructions followed; stopping reprompting.")
        return True, StopReasons.ALL_INSTRUCTIONS_ADHERED
    
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
    
    def gen_caption(self, iteration_num, best_failed_count):
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
    
    def count_tokens(self, text):
        """
        Count tokens in a string using the configured tokenizer.

        Args:
            text (str): Text to tokenize.

        Returns:
            int: Token count.
        """
        return len(self.tokenizer.encode(text))
    
    def build_telemetry_entry(
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
    
    def terminate_with_failure(self, stop_reason, iteration, prompt, telemetry_data, iteration_outputs, cumulative_latency_ms):
        """
        Handle a fatal error by recording telemetry and returning a failure response.

        Args:
            stop_reason (str): Reason for failure.
            iteration (int): Current iteration.
            prompt (str): Prompt used at failure.
            telemetry_data (list): Collected telemetry entries.
            iteration_outputs (dict): Outputs from completed iterations.
            cumulative_latency_ms (float): Total pipeline latency.

        Returns:
            dict: Failure response with best available output, telemetry, and caption.
        """
        entry = self.build_telemetry_entry(
            iteration,
            cumulative_latency_ms,
            0,  # prompt_tokens
            0,  # response_tokens
            {},  # scores
            [],  # response_feedback
            None,  # residual_error
            None,  # failed_instructions_count
            stop_reason,
            prompt,
            f"[ERROR: {stop_reason}]",
            []  # non_fatal_errors
        )
        telemetry_data.append(entry)
        self.telemetry.emit(**entry)

        # Pick best previous response if available
        best_output = "[ERROR: No valid response]"
        if iteration_outputs:
            valid = [entry for entry in iteration_outputs.values() if isinstance(entry.get("residual_error_score"), (int, float))]
            if valid:
                best_iter = min(valid, key=lambda x: x["residual_error_score"])
                best_output = best_iter["response_text"]

        return {
            "best_response": best_output,
            "telemetry": telemetry_data if self.config.return_telemetry else None,
            "caption": f"Re-prompting terminated early due to {stop_reason}." if self.config.return_aimon_caption else None
        }

