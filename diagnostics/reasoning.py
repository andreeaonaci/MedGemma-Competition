import json
from typing import Dict

# Lista exhaustiva a diagnosticelor oftalmologice (ordonata alfabetic intern)
OPHTHALMOLOGICAL_DIAGNOSES = sorted([
    "ACANTHAMOEBA KERATITIS", "AGE-RELATED MACULAR DEGENERATION (AMD)", 
    "ALLERGIC CONJUNCTIVITIS", "AMBLYOPIA", "ANGLE-CLOSURE GLAUCOMA", 
    "ASTIGMATISM", "BACTERIAL CONJUNCTIVITIS", "BLEPHARITIS", 
    "BRANCH RETINAL ARTERY OCCLUSION (BRAO)", "BRANCH RETINAL VEIN OCCLUSION (BRVO)", 
    "CATARACT", "CENTRAL RETINAL ARTERY OCCLUSION (CRAO)", 
    "CENTRAL RETINAL VEIN OCCLUSION (CRVO)", "CENTRAL SEROUS RETINOPATHY", 
    "CHALAZION", "COLOR BLINDNESS", "CONJUNCTIVITIS", "CORNEAL ABRASION", 
    "CORNEAL DYSTROPHY", "CORNEAL ULCER", "CYTOMEGALOVIRUS (CMV) RETINITIS", 
    "DACRYOCYSTITIS", "DIABETIC MACULAR EDEMA (DME)", "DIABETIC RETINOPATHY", 
    "DRY EYE SYNDROME", "ECTROPION", "ENDOPHTHALMITIS", "ENTROPION", 
    "EPIRETINAL MEMBRANE", "ESOTROPIA", "EXOTROPIA", "FUCHS DYSTROPHY", 
    "GLAUCOMA", "HERPES KERATITIS", "HORDEOLUM (STYE)", "HYPEROPIA", 
    "HYPERTENSIVE RETINOPATHY", "HYPHEMA", "IRITIS", "ISCHEMIC OPTIC NEUROPATHY", 
    "KERATOCONUS", "MACULAR EDEMA", "MACULAR HOLE", "MYOPIA", "NYSTAGMUS", 
    "OCULAR HYPERTENSION", "OPTIC NEURITIS", "OPTIC NEUROPATHY", "PAPILLEDEMA", 
    "PINGUECULA", "PRESBYOPIA", "PTERYGIUM", "PTOSIS", "RETINAL DETACHMENT", 
    "RETINAL TEAR", "RETINITIS PIGMENTOSA", "RETINOBLASTOMA", 
    "RETINOPATHY OF PREMATURITY (ROP)", "SCLERITIS", "STRABISMUS", 
    "SUBCONJUNCTIVAL HEMORRHAGE", "THYROID EYE DISEASE", "TRACHOMA", 
    "TRICHIASIS", "UVEITIS", "VIRAL CONJUNCTIVITIS", "VITREOUS DETACHMENT", 
    "VITREOUS HEMORRHAGE"
])

# Construim lista finala cu HEALTHY la inceput si OTHER la final
ALLOWED_CONDITIONS = ["HEALTHY"] + OPHTHALMOLOGICAL_DIAGNOSES + ["OTHER"]

REQUIRED_KEYS = {
    "condition",
    "confidence",
    "findings",
    "severity",
    "recommendations",
}

def build_diagnosis_prompt(clinical_context: str) -> str:
    """
    Construct a strict ophthalmology diagnostic prompt.

    Constraints:
    - Limit scope to predefined retinal conditions only.
    - Force structured reasoning.
    - Force strictly valid JSON output.
    - No additional commentary allowed.
    """
    allowed = ", ".join(ALLOWED_CONDITIONS)

    prompt = f"""
You are an ophthalmology diagnostic assistant.

You are strictly limited to diagnosing ONLY the following conditions:
{allowed}

You must:
1. Analyze retinal fundus image features.
2. Reason step by step internally.
3. Select the most likely condition from the allowed list ONLY. If the condition is missing, output "OTHER".
4. Assign severity: mild, moderate, severe, or N/A.
5. Provide structured findings.
6. Provide clinical recommendations.
7. Provide confidence score between 0 and 1.

You MUST return ONLY valid JSON.
No markdown.
No explanation.
No extra text.
No commentary.

JSON schema:
{{
    "condition": "<one of allowed conditions>",
    "severity": "<mild|moderate|severe|N/A>",
    "confidence": <float 0-1>,
    "findings": ["finding1", "finding2"],
    "recommendations": ["rec1", "rec2"]
}}

Clinical Context:
{clinical_context}
"""
    return prompt.strip()

def enforce_json_schema(output_text: str) -> Dict:
    """
    Objectively validate JSON structure without restricting AI diagnostic capabilities.
    """
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError("Model output is not valid JSON.") from e

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object.")

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    return parsed