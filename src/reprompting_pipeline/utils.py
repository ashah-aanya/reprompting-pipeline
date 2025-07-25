from typing import List, Union, Any  

def normalize_context(context: Union[str, List[Any]]) -> str:
    if isinstance(context, str):
        return context  
    if isinstance(context, list):
        return "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in context])
    raise ValueError(f"Unsupported context type: {type(context)}")

def toxicity_check(result):
    """
    Verifies output toxicity.

    Returns:
    bool: True if above threshold, else False.
    """
    toxicity = result.detect_response.toxicity["score"]
    if toxicity > 0.25:
        return True
    return False
    
def get_failed_instructions(result, aimon_payload):
    """
    Extracts failed instructions from instruction_adherence and groundedness sections
    of the AIMon detect_response. Includes any toxicity-related instructions with
    follow_probability > 0.25, even if they aren't labeled as failed.

    Returns:
        List[Dict]: Sorted list of failed instructions by increasing follow_probability.
        Toxicity-related failures appear first.
    """
    failed = []

    # Extract from instruction_adherence
    for inst in result.detect_response.instruction_adherence.get("instructions_list", []):
        is_failed = not inst.get("label", True)
        score = inst.get("follow_probability", 0.0)
        text = inst.get("instruction", "")
        explanation = inst.get("explanation", "")
        if is_failed:               
            failed.append({
                "type": "instruction_failure",
                "source": "instruction_adherence",
                "instruction": text,
                "score": score,
                "explanation": explanation
            })
        elif "toxicity" in text.lower() and score > 0.25:
            failed.append({
                "type": "instruction_failure",
                "source": "instruction_adherence",
                "instruction": text,
                "score": score,
                "explanation": explanation
            })

        # Extract from groundedness
    for inst in result.detect_response.groundedness.get("instructions_list", []):
        is_failed = not inst.get("label", True)
        score = inst.get("follow_probability", 0.0)
        text = inst.get("instruction", "")
        explanation = inst.get("explanation", "")
        if is_failed:
            failed.append({
                "type": "instruction_failure",
                "source": "groundedness",
                "instruction": text,
                "score": score,
                "explanation": explanation
            })
        elif "toxicity" in text.lower() and score > 0.25:
            failed.append({
                "type": "instruction_failure",
                "source": "groundedness",
                "instruction": text,
                "score": score,
                "explanation": explanation
            })

    # Sort by follow_probability
    failed.sort(key=lambda x: x["score"])

    # Move high-score toxicity issues to top
    toxicity_first = [f for f in failed if "toxicity" in f["instruction"].lower()]
    others = [f for f in failed if "toxicity" not in f["instruction"].lower()]
    return toxicity_first + others
    
def get_failed_instructions_count(result) -> int:
    """
    Counts the number of failed instructions (including toxicity > 0.25).
    Optimized to skip building detailed failure objects.
    """
    count = 0
    # Instruction adherence
    for inst in result.detect_response.instruction_adherence.get("instructions_list", []):
        if not inst.get("label", True) or ("toxicity" in inst.get("instruction", "").lower() and inst.get("follow_probability", 0.0) > 0.25):
            count += 1
    # Groundedness
    for inst in result.detect_response.groundedness.get("instructions_list", []):
        if not inst.get("label", True) or ("toxicity" in inst.get("instruction", "").lower() and inst.get("follow_probability", 0.0) > 0.25):
            count += 1
    return count