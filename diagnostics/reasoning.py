import json
from typing import Dict

# Lista exhaustiva a diagnosticelor oftalmologice (ordonata alfabetic intern)
OPHTHALMOLOGICAL_DIAGNOSES = sorted([
    "ACANTHAMOEBA KERATITIS", "AGE-RELATED MACULAR DEGENERATION (DRY)", 
    "AGE-RELATED MACULAR DEGENERATION (WET)", "ALLERGIC CONJUNCTIVITIS", 
    "AMBLYOPIA", "ANTERIOR ISCHEMIC OPTIC NEUROPATHY (AION)", 
    "ANTERIOR UVEITIS (IRITIS)", "ASTIGMATISM", "BACTERIAL CONJUNCTIVITIS", 
    "BACTERIAL CORNEAL ULCER", "BEST DISEASE", "BLEPHARITIS", 
    "BRANCH RETINAL ARTERY OCCLUSION (BRAO)", "BRANCH RETINAL VEIN OCCLUSION (BRVO)", 
    "BULLOUS KERATOPATHY", "CATARACT", "CENTRAL RETINAL ARTERY OCCLUSION (CRAO)", 
    "CENTRAL RETINAL VEIN OCCLUSION (CRVO)", "CENTRAL SEROUS CHORIORETINOPATHY (CSCR)", 
    "CHALAZION", "CHOROIDAL MELANOMA", "CHOROIDAL NEOVASCULARIZATION (CNV)", 
    "COLOR BLINDNESS", "CONGENITAL GLAUCOMA", "CONJUNCTIVITIS", 
    "CORNEAL ABRASION", "CORNEAL DYSTROPHY", "CYSTOID MACULAR EDEMA (CME)", 
    "CYTOMEGALOVIRUS (CMV) RETINITIS", "DACRYOCYSTITIS", 
    "DIABETIC MACULAR EDEMA (DME)", "DIABETIC RETINOPATHY (NON-PROLIFERATIVE)", 
    "DIABETIC RETINOPATHY (PROLIFERATIVE)", "DRY EYE SYNDROME", 
    "ECTROPION", "ENDOPHTHALMITIS", "ENTROPION", "EPIRETINAL MEMBRANE (ERM)", 
    "ESOTROPIA", "EXOTROPIA", "FUCHS ENDOTHELIAL DYSTROPHY", "FUNGAL KERATITIS", 
    "GLAUCOMA (ANGLE-CLOSURE)", "GLAUCOMA (NEOVASCULAR)", "GLAUCOMA (NORMAL TENSION)", 
    "GLAUCOMA (PIGMENTARY)", "GLAUCOMA (PRIMARY OPEN-ANGLE)", 
    "GLAUCOMA (PSEUDOEXFOLIATION)", "HERPES SIMPLEX KERATITIS", 
    "HERPES ZOSTER OPHTHALMICUS", "HORDEOLUM (STYE)", "HYPEROPIA", 
    "HYPERTENSIVE RETINOPATHY", "HYPHEMA", "INTERMEDIATE UVEITIS", 
    "ISCHEMIC OPTIC NEUROPATHY", "KERATOCONUS", "MACULAR EDEMA", "MACULAR HOLE", 
    "MICROANEURYSMS", "MYOPIA", "MYOPIC MACULOPATHY", 
    "NON-ARTERITIC ANTERIOR ISCHEMIC OPTIC NEUROPATHY (NAION)", "NYSTAGMUS", 
    "OCULAR HYPERTENSION", "OCULAR ISCHEMIC SYNDROME", "OPTIC NEURITIS", 
    "OPTIC NEUROPATHY", "PANUVEITIS", "PAPILLEDEMA", "PINGUECULA", 
    "POSTERIOR UVEITIS", "PRESBYOPIA", "PSEUDOPHAKIA", "PTERYGIUM", "PTOSIS", 
    "PURTSCHER RETINOPATHY", "RETINAL DETACHMENT (EXUDATIVE)", 
    "RETINAL DETACHMENT (RHEGMATOGENOUS)", "RETINAL DETACHMENT (TRACTIONAL)", 
    "RETINAL TEAR", "RETINITIS PIGMENTOSA", "RETINOBLASTOMA", 
    "RETINOPATHY OF PREMATURITY (ROP)", "SCLERITIS", "SOLAR RETINOPATHY", 
    "STARGARDT DISEASE", "STRABISMUS", "SUBCONJUNCTIVAL HEMORRHAGE", 
    "SYMPATHETIC OPHTHALMIA", "THYROID RETINOPATHY", 
    "THYROID-RELATED OPHTHALMOPATHY (TRO)", "TOXIC OPTIC NEUROPATHY", 
    "TOXOPLASMOSIS RETINOCHOROIDITIS", "TRACHOMA", "TRICHIASIS", 
    "VALSALVA RETINOPATHY", "VIRAL CONJUNCTIVITIS", "VITREOMACULAR TRACTION (VMT)", 
    "VITREOUS DETACHMENT", "VITREOUS HEMORRHAGE"
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