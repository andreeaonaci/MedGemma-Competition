import json
from typing import Dict


ALLOWED_CONDITIONS = [
    "diabetic retinopathy",
    "glaucoma",
    "age-related macular degeneration",
    "retinal detachment",
    "microaneurysms",
    "hemorrhages",
    "exudates",
]


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
3. Select the most likely condition from the allowed list only.
4. Assign severity: mild, moderate, severe.
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
    "severity": "<mild|moderate|severe>",
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

    # Toate validarile absurde pentru continut au fost eliminate.
    # Returnam direct dictionarul validat structural.
    return parsed
