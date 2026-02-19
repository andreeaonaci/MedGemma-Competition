import logging
from typing import Dict, Any
from PIL import Image
import re

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    
    BitsAndBytesConfig,
)

from diagnostics.reasoning import (
    
    enforce_json_schema,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_ID = "google/medgemma-4b-it" 


class MedGemmaWrapper:
    """
    Optimized for NVIDIA Quadro RTX 3000 (6GB VRAM)

    - 4-bit quantization
    - Automatic CUDA detection
    - device_map="auto"
    - Strict JSON enforcement
    """

    def __init__(self, device: str = "cuda"):
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA detected. Using GPU.")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available. Falling back to CPU.")

        self.model = None
        self.processor = None

    def load(self):
        logger.info(f"Loading model: {MODEL_ID}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )

        if self.device == "cuda":

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            self.model.to("cpu")

        self.model.eval()

        logger.info("Model loaded successfully.")


    def generate_diagnosis(self, image: Image.Image, clinical_context: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # 1. Provide an exact JSON template in the system prompt
        system_prompt = """You are an expert ophthalmologist.
You MUST output your response STRICTLY as a raw JSON object matching this exact format, with no additional text, markdown, or explanation:
{
  "findings": "...",
  "condition": "...",
  "severity": "...",
  "recommendations": "...",
  "confidence": "high/medium/low"
}"""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": clinical_context},
                                        {"type": "image", "image": image}]}
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.float16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False
            )
            generation = generation[0][input_len:]

        raw_output = self.processor.decode(generation, skip_special_tokens=True)

        # 2. Force clean extraction of the JSON block using Regex
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            clean_json_string = json_match.group(0)
        else:
            # Fallback in case the model failed completely
            clean_json_string = raw_output 

        return enforce_json_schema(clean_json_string)


    def generate_comparison(self, text_a: str, text_b: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        prompt = f"""
Compare two ophthalmology retinal reports.

Report A:
{text_a}

Report B:
{text_b}

Return strict JSON:
{{
  "comparison_result": "...",
  "reasoning": "...",
  "confidence_level": "low/medium/high"
}}
"""

        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0
            )

        raw_output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return enforce_json_schema(raw_output)


def load_model(device: str = "cuda"):
    wrapper = MedGemmaWrapper(device=device)
    wrapper.load()
    return wrapper


if __name__ == "__main__":
    # Simple test
    model = load_model()

    dummy_image = torch.zeros((3, 224, 224))  # Dummy image tensor
    clinical_context = "Patient with blurred vision and floaters."
    diagnosis = model.generate_diagnosis(dummy_image, clinical_context)
    print(diagnosis)