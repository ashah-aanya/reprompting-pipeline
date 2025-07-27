from typing import List, Union, Any  

TOXICITY_THRESHOLD = 0.25

def normalize_context(context: Union[str, List[Any]]) -> str:
    if isinstance(context, str):
        return context  
    if isinstance(context, list):
        return "\n\n".join([getattr(doc, "page_content", str(doc)) for doc in context])
    raise ValueError(f"Unsupported context type: {type(context)}")

def _count_toxicity_failures(result) -> int:
    """
    Counts how many toxicity instructions exceed the threshold.

    Returns:
        int: Number of failed toxicity instructions.
    """
    return sum(
        1
        for inst in result.detect_response.toxicity.get("instructions_list", [])
        if inst.get("follow_probability", 0.0) > TOXICITY_THRESHOLD
    )


def toxicity_check(result) -> bool:
    """
    Verifies if there are any toxicity failures.

    Returns:
        bool: True if at least one toxicity instruction exceeds the threshold, else False.
    """
    return _count_toxicity_failures(result) > 0


def get_failed_toxicity_instructions(result):
    failed = []
    for inst in result.detect_response.toxicity.get("instructions_list", []):
        if inst.get("follow_probability", 0.0) > TOXICITY_THRESHOLD:
            failed.append({
                "type": "toxicity_failure",
                "source": "toxicity",
                "instruction": inst.get("instruction", ""),
                "score": inst.get("follow_probability", 0.0),
                "explanation": inst.get("explanation", "")
            })
    return failed
    
def get_failed_instructions(result):
    """
    Extracts failed instructions:
    - Adherence & groundedness: label=False
    - Toxicity: score > threshold
    """
    failed = []
    # Adherence & groundedness: label-based
    for source in ["instruction_adherence", "groundedness"]:
        for inst in getattr(result.detect_response, source, {}).get("instructions_list", []):
            if not inst.get("label", True):
                failed.append({
                    "type": f"{source}_failure",
                    "source": source,
                    "instruction": inst.get("instruction", ""),
                    "score": inst.get("follow_probability", 0.0),
                    "explanation": inst.get("explanation", "")
                })
    # Toxicity: score-based
    failed.extend(get_failed_toxicity_instructions(result))
    # Sort by score (most confident first)
    failed.sort(key=lambda x: x["score"], reverse=True)
    return failed

def get_failed_instructions_count(result) -> int:
    """
    Counts failed instructions across instruction_adherence, groundedness, and toxicity.
    - For adherence & groundedness: uses label=False
    - For toxicity: uses score > threshold
    """
    count = 0
    # Instruction adherence
    for inst in result.detect_response.instruction_adherence.get("instructions_list", []):
        if not inst.get("label", True):
            count += 1
    # Groundedness
    for inst in result.detect_response.groundedness.get("instructions_list", []):
        if not inst.get("label", True):
            count += 1
    count += _count_toxicity_failures(result)  # Toxicity
    return count

def get_residual_error_score(result):
    """
    Calculates a residual error score based on penalized follow probabilities.
    Heavily penalizes values < 0.5.
    A higher score = worse response.
    """
    grounded_list = result.detect_response.groundedness.get("instructions_list", [])
    grounded_probs = [item["follow_probability"] for item in grounded_list] if grounded_list else []
    avg_grounded_penalty = penalized_average(grounded_probs) if grounded_probs else 0.0

    adherence_list = result.detect_response.instruction_adherence.get("instructions_list", [])
    adherence_probs = [item["follow_probability"] for item in adherence_list] if adherence_list else []
    avg_adherence_penalty = penalized_average(adherence_probs) if adherence_probs else 0.0

    residual_error_score = (avg_grounded_penalty + avg_adherence_penalty) / 2
    return round(residual_error_score, 2)

def penalized_average(probs):
    """
    Penalize follow probabilities below 0.5 more heavily.
    For each instruction:
    - If follow_probability >= 0.5 → use as is        
    - If follow_probability < 0.5 → double the penalty (1 - prob) * 2
    """
    penalties = []
    for p in probs:
        if p >= 0.5:
            penalty = 1 - p
        else:
            penalty = (1 - p) * 2  # heavier penalty
        penalties.append(penalty)
    return sum(penalties) / len(penalties)