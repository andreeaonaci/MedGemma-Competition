import logging
import base64
from io import BytesIO
from PIL import Image
from contextlib import asynccontextmanager
import cv2
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import torch

from langchain_core.prompts import PromptTemplate

from diagnostics.reasoning import build_diagnosis_prompt
from diagnostics.reasoning import build_diagnosis_prompt
from models.medgemma_wrapper import load_model
from utils.attention_rollout import AttentionRollout as VisionAttentionRollout
import os

os.environ["PYTORCH_SDP_ATTENTION"] = "eager"



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

medgemma_model = None

# --- SCHEME DE DATE ---

class VisualComparisonRequest(BaseModel):
    image1_base64: str
    image2_base64: str
    question: str

class ChatMessage(BaseModel):
    role: str 
    content: str

class DiagnosisChatRequest(BaseModel):
    image_base64: str
    message: str
    history: List[ChatMessage] = []
    clinical_context: Optional[str] = "You are an expert ophthalmology AI. Answer the user's questions about the provided retinal image."

class DiagnosisChatResponse(BaseModel):
    reply: str
    

class AuditDiagnosisRequest(BaseModel):
    image_base64: str
    clinical_context: str

# --- UTILITARE ---

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decodifica string-ul base64 intr-un obiect PIL Image pentru model."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def build_langchain_prompt(request: DiagnosisChatRequest) -> str:
    """Foloseste Langchain pentru a formata istoricul si intrebarea curenta."""
    
    template = """{system_context}

Conversation History:
{history_text}

Human: {current_message}
AI Ophthalmic Assistant:"""

    prompt = PromptTemplate(
        input_variables=["system_context", "history_text", "current_message"],
        template=template
    )
    
    # Formatam istoricul din lista in text
    history_str = ""
    for msg in request.history:
        prefix = "Human: " if msg.role == "user" else "AI: "
        history_str += f"{prefix}{msg.content}\n"
        
    if not history_str:
        history_str = "No previous conversation."

    return prompt.format(
        system_context=request.clinical_context,
        history_text=history_str.strip(),
        current_message=request.message
    )



@asynccontextmanager
async def lifespan(app: FastAPI):
    global medgemma_model
    logger.info("Starting up FastAPI server. Loading MedGemma model into VRAM...")
    try:
        medgemma_model = load_model(device="cuda")
        logger.info("Model loaded successfully into API context.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield 
    
    logger.info("Shutting down FastAPI server. Clearing VRAM...")
    if medgemma_model is not None:
        del medgemma_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(title="MedGemma Diagnostic API", version="1.0", lifespan=lifespan)

###Endpointuri

@app.get("/health")
async def health_check():
    status = "ready" if medgemma_model is not None else "model_not_loaded"
    return {"status": status}

@app.post("/chat/diagnosis", response_model=DiagnosisChatResponse)
async def chat_diagnosis(request: DiagnosisChatRequest):
    if medgemma_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        pil_image = decode_base64_to_image(request.image_base64)
        
        final_prompt = build_langchain_prompt(request)
        
        ai_response = medgemma_model.generate_chat_response(pil_image, final_prompt)
        
        return DiagnosisChatResponse(reply=ai_response)
        
    except Exception as e:
        logger.error(f"Error during API inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit_diagnosis")
async def audit_diagnosis(request: AuditDiagnosisRequest):
    if medgemma_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        pil_image = decode_base64_to_image(request.image_base64)
        
        result_dict = medgemma_model.generate_diagnosis(pil_image, request.clinical_context)
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error during API audit diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visual_comparison")
async def visual_comparison(request: VisualComparisonRequest):
    if medgemma_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        
        img1 = decode_base64_to_image(request.image1_base64)
        img2 = decode_base64_to_image(request.image2_base64)
        
        result_text = medgemma_model.generate_visual_comparison(img1, img2, request.question)
        
        return {"result": result_text}
        
    except Exception as e:
        logger.error(f"Error during API visual comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/attention-heatmap")
async def attention_heatmap_endpoint(
    file: UploadFile = UploadFile(...), 
    context: str = Form(...)
):
    if medgemma_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Load image
        pil_image = Image.open(BytesIO(await file.read())).convert("RGB")

        # Build the prompt like in reasoning.py
        full_prompt = build_diagnosis_prompt(context)

        # Construct messages exactly as generate_diagnosis does
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image", "image": pil_image}
                ]
            }
        ]

        # Tokenize for the model
        inputs = medgemma_model.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(medgemma_model.device, dtype=torch.float16)

        # Generate attention rollout
        rollout = VisionAttentionRollout(medgemma_model.model)  # SigLIP-based
        cam_tensor = rollout.generate(inputs)  # torch tensor [H, W] normalized 0-1

        # Convert torch tensor -> PIL image
        cam_tensor = (cam_tensor.clamp(0, 1) * 255).to(torch.uint8)
        cam_pil = Image.fromarray(cam_tensor.cpu().numpy()).convert("RGB")

        # Return as PNG
        buffer = BytesIO()
        cam_pil.save(buffer, format="PNG")
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Error generating attention heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)