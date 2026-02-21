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
    Construct a strict ophthalmology diagnostic prompt using Directed Chain-of-Thought.
    """
    allowed = ", ".join(ALLOWED_CONDITIONS)

    prompt = f"""
You are an expert ophthalmology diagnostic AI. 

CRITICAL INSTRUCTION: You MUST assume this retinal image contains subtle pathology (e.g., microaneurysms, drusen, abnormal cupping, hemorrhages) unless you can definitively prove otherwise. Do NOT default to "HEALTHY". 

You are strictly limited to diagnosing ONLY from the following conditions:
{allowed}

MANDATORY PROTOCOL:
1. SCAN the macula for drusen or exudates.
2. SCAN the vascular network for dot-blot hemorrhages or microaneurysms.
3. SCAN the optic disc for abnormal cup-to-disc ratio or pallor.
4. If ANY of the above are found, select the exact specific condition from the list.
5. You may ONLY output "HEALTHY" if you have rigorously scanned all three areas and found zero abnormalities. If you are unsure, default to "OTHER" or the closest matching pathology, NOT "HEALTHY".

JSON schema:
{{
    "condition": "<exact string from allowed list>",
    "severity": "<mild|moderate|severe|N/A>",
    "confidence": <float 0.0-1.0>,
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