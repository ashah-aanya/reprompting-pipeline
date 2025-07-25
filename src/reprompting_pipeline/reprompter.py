from aimon.reprompting_api.utils import toxicity_check, get_failed_instructions_count, get_failed_instructions
class Reprompter:                                                            

    def get_passed_instructions(self, result, aimon_payload):
        """
        Returns an ordered list of instructions that passed
        instruction adherence and groundedness checks.

        Instructions are considered passed if they are not present in the failed list.
        """
        # Get all original instructions
        all_instructions = aimon_payload.get("instructions", [])

        # Get failed instructions from unified failure extractor
        failed_instructions = {item["instruction"] for item in get_failed_instructions(result)}

        # Filter to only passed instructions
        passed = [instr for instr in all_instructions if instr not in failed_instructions]

        return passed

    ## should this resurface instructions that were followed? Should this resurface context?
    def create_corrective_prompt(self, result, aimon_payload):
        failed_instructions = get_failed_instructions(result)
        failed_count = get_failed_instructions_count(result)

        if toxicity_check(result):
            tone = "Your reply contained toxic content. You must revise it to remove any harmful language."   
        # Tone based on residual error score
        if failed_count >= 3:
            tone = "Your reply had major issues. Fix all points below."
        elif failed_count >= 2:
            tone = "Some parts were off. Improve using the notes below."
        else:
            tone = "Almost there. Just a few small fixes needed."

        ## relays failed instructions in order of confidence
        ## should we resurface instructions that were followed but had a follow probability < 0.85?
        issue_lines = []
        if failed_instructions:
            issue_lines.append("Fix the following:")
            for i, error in enumerate(failed_instructions, start=1):
                msg = self.get_reprompt_per_instruction(error)
                issue_lines.append(f"{i}. {msg}")
        else:
            issue_lines.append("No major issues.")

        issue_text = "\n\n".join(issue_lines)
        passed = self.get_passed_instructions(result, aimon_payload)
        if passed:
            passed_instructions = (
                "Continue to follow these instructions:\n" +
                "\n".join(f"- {instr}" for instr in passed) + "\n\n"
            )
        else:
            passed_instructions = ""

        # corrective prompt
        prompt = (
            f"Revise your previous response to this query:\n"
            f"{aimon_payload['user_query']}\n\n"
            ## resurfaces context
            f"Context:\n"
            f"{aimon_payload['context']}\n\n"
            f"Previous response:\n"
            f"{aimon_payload['generated_text']}\n\n"
            f"{tone}\n\n"
            f"{issue_text}\n\n"
            "Preserve correct content. Return only the revised output with no extra explanation.\n"
            ## resurfaces user instructions that were followed as a reminder to continue
            f"{passed_instructions}\n\n"

        )
        return prompt

    ## based on the IA confidence in the failure of the instruction, dictate the tone of the reprompt
    def get_reprompt_per_instruction(self, failed_instruction):
        """
            Generate a targeted corrective instruction for a single failed instruction.
            This function should be optimized to provide clear and actionable feedback
            based on the confidence level of the failure.
        """        
        confidence = (1.0 - failed_instruction["score"])*100  # Higher means we are more confident they failed
        confidence_str = f"{confidence:.2f}%"
        
        return (
            f" We are {confidence_str} confident that the following instruction was not followed:\n"
            f"→ Violated Instruction: \"{failed_instruction['instruction']}\"\n"
            f"→ Explanation: {failed_instruction['explanation']}\n"
        )

