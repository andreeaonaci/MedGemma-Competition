import logging
from typing import Dict, Any

import torch

from diagnostics.reasoning import (
    build_diagnosis_prompt,
    enforce_json_schema,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MedGemmaWrapper:
    """
    Local MedGemma wrapper with:

    - Automatic CUDA detection
    - Safe CPU fallback
    - Structured prompting
    - Strict JSON enforcement
    - torch.no_grad() inference
    """

    def __init__(self, device: str = "cuda"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA detected. Using GPU.")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available. Falling back to CPU.")

        self.model = None

    def load(self):
        """
        Load MedGemma model locally and move to correct device.
        Actual loading implementation not included.
        """
        self.model = None

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def generate_diagnosis(self, image, clinical_context: str) -> Dict[str, Any]:
        """
        Full diagnosis pipeline:

        1. Build strict prompt
        2. Run model inference
        3. Enforce strict JSON schema
        4. Return validated dictionary
        """

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        image = self._move_to_device(image)

        prompt = build_diagnosis_prompt(clinical_context)

        with torch.no_grad():
            # Placeholder inference call
            # Expected to return raw string output
            raw_output = ""

        validated_output = enforce_json_schema(raw_output)

        return validated_output

    def generate_comparison(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Comparison not structured yet.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        with torch.no_grad():
            raw_output = ""

        return {"result": raw_output}


def load_model(device: str = "cuda"):
    wrapper = MedGemmaWrapper(device=device)
    wrapper.load()
    return wrapper


def generate_diagnosis(image, clinical_context: str) -> dict:
    model = load_model()
    return model.generate_diagnosis(image, clinical_context)


def generate_comparison(text_a: str, text_b: str) -> dict:
    model = load_model()
    return model.generate_comparison(text_a, text_b)
