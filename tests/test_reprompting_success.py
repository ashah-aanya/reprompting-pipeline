from aimon.reprompting_api.config import RepromptingConfig
from aimon.reprompting_api.runner import run_reprompting_pipeline
import os
from together import Together

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AIMON_API_KEY = os.environ.get("AIMON_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)

# Define llm_fn
def my_llm(prompt: str) -> str:
   """Calls a Mistral model and returns the generated text."""
   response = client.chat.completions.create(
       model="mistralai/Mistral-7B-Instruct-v0.2",  # Any Together‑hosted model
       messages=[{"role": "user", "content": prompt}]
   )
   return response.choices[0].message.content

def test_successful_run():
    config = RepromptingConfig(
        aimon_api_key = AIMON_API_KEY,
        publish=True,
        return_telemetry=True,
        return_aimon_summary=True,
        application_name="api_test",
    )
    
    user_query = "what are the drug tiers?"
    context = """[SECTION] 📘 BlueShield Rx Policy Addendum: 2023–2025 Commercial & Employer-Sponsored Plans [SECTION] Confidential – Not for external dissemination without compliance review. [SECTION] 🔹 Section 2.1.7 – Drug Coverage Eligibility Matrix [SECTION] Prescription drug eligibility is governed by a tiered, multi-variant benefit design informed by annual P&T Committee decisions, manufacturer rebates, CMS Part D benchmarking (when applicable), and employer-specific customizations. The following formulary tiers apply unless superseded by a group rider or conditional override: [SECTION] - **Tier 1 (Generic Core):** Includes FDA-approved AB-rated generics; requires no PA or ST, unless the member is flagged under the Risk Management Tier Hold (RMTH) protocol due to prior misuse. [SECTION] - **Tier 2 (Preferred Brand & Enhanced Generics):** Coverage dependent on documented trial/failure of Tier 1 alternatives unless contraindicated. Members in the Legacy Bridge plan must obtain both prescriber attestation and pharmacy alignment verification. [SECTION] - **Tier 3 (Non-Preferred & Specialty Entry):** May require dual-layer review if member has not met chronic condition enrollment criteria (CCE) in the last benefit year. Tier migration possible mid-cycle based on new formulary rules. [SECTION] - **Tier 4 (Specialty Injectables, Biologics, and Condition-Limited Agents):** Includes drugs subject to clinical pathway alignment; claims must be adjudicated through the PBM’s split-fulfillment logic unless the prescribing entity is credentialed as Tier 4-A. [SECTION] 🚫 Exception: Certain biosimilars classified under Tier 4 in national formularies may be covered at Tier 2 if dispensed under limited-distribution contracts, provided the prescribing facility participates in the 340B program **and** the member is flagged under Enhanced Affordability Priority (EAP). [SECTION] 🔁 **Prior Authorization (PA) Layering Logic** [SECTION] Drugs requiring PA are subject to a three-stage filter: [SECTION] 1. **Therapeutic Criteria Review (TCR)** – Clinical alignment with diagnosis and formulary path. [SECTION] 2. **Coverage Policy Sync (CPS)** – Matches requested use with plan sponsor coverage schema. [SECTION] 3. **Utilization Watch Flag (UWF)** – If triggered, a third-party medical director review is initiated (adds 2–4 business days). [SECTION] 💡 Exemplar: *Trulicity* (GLP-1 receptor agonist) [SECTION] - **Base Tier:** Tier 3 across most commercial plans [SECTION] - **Override Possibility:** Auto-lifts to Tier 2 under Metabolic Risk Bundling if member is concurrently enrolled in cardiac risk management AND insulin titration modules. [SECTION] - **Caveat:** Auto-injector version may still trigger UWF if prescribed without 90-day adherence documentation to metformin or contraindication to semaglutide. [SECTION] 🗂️ **Adjudication Complexity Notes** [SECTION] - Fill attempts at non-network or out-of-state pharmacies may default to full retail pricing, even if coverage is active. [SECTION] - Certain maintenance tier drugs can only be filled at 90-day intervals after two successful 30-day fills unless dispensed via SmartSync (auto-align refill system). [SECTION] - Claims using discount cards (e.g., manufacturer copay assistance) will not count toward deductible or out-of-pocket limits unless the pharmacy submits a Coordinated Adjudication Adjustment Request (CAAR). [SECTION] ⚠️ **Denials & Appeals** [SECTION] - If PA is denied, appeals must cite new clinical rationale. Re-submission of identical documentation will be auto-denied. [SECTION] - Members in Tier Restructuring Delay (TRD) periods due to employer override cannot file external appeals unless the drug is life-sustaining and not replaceable under Tier 1/2. [SECTION] - Denials on non-formulary drugs are not eligible for Tier Transition Program (TTP) unless covered during prior plan year with no lapse in coverage >30 days. [SECTION] 📊 **Plan Differences** [SECTION] - Standard, Enhanced, Platinum, and Concierge tiers each have different deductible-accumulation thresholds and copay structures. [SECTION] - For Platinum+ plans, Tier 3 copay is waived on first-time fills initiated post-discharge from an inpatient episode if coded using post-acute NDCs. [SECTION] 📣 Misc. Clarifications [SECTION] - The “Healthy Living Rewards” program, mentioned in new member packets, does not affect coverage or drug tier placement. It is a wellness initiative only. [SECTION] - Benefit year resets on Jan 1, but tier realignment occurs quarterly and may retroactively affect claims filled in the trailing 45-day buffer period. [SECTION] 🔒 REMINDER: Member Services guidance may reflect outdated tier assignments if formulary refreshes are in progress. Online lookup tools update in real time and take precedence during adjudication disputes."""
    user_instructions = [
        "Avoid overly technical or robotic phrasing; keep the tone human and accessible.",
        "Ensure the response is direct and professional, with minimal informal tone.",
        "Translate or simplify technical details from the context into accurate, user-friendly explanations."
    ]
    
    result = run_reprompting_pipeline(
            user_query=user_query,
            context=context, 
            llm_fn=my_llm,
            reprompting_config=config,
            user_instructions=user_instructions
        )
    
    print("\n=== BEST RESPONSE ===")
    print(result["best_response"])
    print("\n=== TELEMETRY ===")
    print(result.get("telemetry"))
    print("\n=== SUMMARY ===")
    print(result.get("summary"))
    
if __name__ == "__main__":
    test_successful_run()