import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, SiglipVisionModel, SiglipProcessor
from tensorflow.image import resize as tf_resize
import torch
import glob
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    return processor, model

def load_separate_models():
    vision_model = SiglipVisionModel.from_pretrained("google/medsiglip-448")
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")

def resize(image):
    return Image.fromarray(
        tf_resize(
            images=image, size=[448, 448], method='bilinear', antialias=False
        ).numpy().astype(np.uint8)
    )

def preprocess_image(image_path):

        image = Image.open(image_path).convert('RGB')
        image = resize(image)#target_size, Image.LANCZOS)

        return image


def generate_embeddings_with_text(texts, images, model, processor, zero_shot_classification=False):
    resized_imgs = [resize(img) for img in images]
    print(texts)
    inputs = processor(text=texts, images=resized_imgs, padding="max_length", truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    img_embeddings = [np.array(emb) for emb in outputs.image_embeds.detach().cpu().numpy()]
    text_embeddings = [np.array(emb) for emb in outputs.text_embeds.detach().cpu().numpy()]
    probs = None
    if zero_shot_classification:
        logits_per_image = outputs.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)

    return {"img_embed": img_embeddings,
            "text_embed": text_embeddings,
            "zero_shot": probs}

def generate_embedding_only_text(texts, model, processor):
    inputs = processor(text=texts, padding = "max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    #print(text_embeds)
    text_embeds = text_embeds["pooler_output"] / text_embeds["pooler_output"].norm(p=2, dim=-1, keepdim=True)
    return text_embeds.detach().cpu().numpy()

def generate_embedding_only_image(images, model, processor):
    inputs = processor(images = images, padding = "max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        img_embeds = model.get_image_features(**inputs)
    img_embeds = img_embeds["pooler_output"] / img_embeds["pooler_output"].norm(p=2, dim=-1, keepdim=True)
    #img_embeds = np.array(image_embeds.cpu().numpy())
    return img_embeds.detach().cpu().numpy()

def generate_embeddings_only_image(images, model, processor):
    resized_imgs = [resize(img) for img in images]

    inputs = processor(images=resized_imgs, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return {"img_embed": outputs.image_embeds}



def save_embeddings(
    image_embeddings,
    description_embeddings,
    image_paths,
    output_path
):

    #output_path = Path(output_path)

    # Save as npz with metadata
    np.savez(
        output_path,
        image_embeddings=image_embeddings,
        description_embeddings = description_embeddings,
        image_paths=[str(p) for p in image_paths]
    )
    print(f"Embeddings saved to {output_path}")

def load_embeddings(embeddings_path) :

    data = np.load(embeddings_path, allow_pickle=True)
    image_embeddings = data['image_embeddings']
    description_embeddings = data['description_embeddings']
    image_paths = data['image_paths'].tolist()
    return image_embeddings, description_embeddings, image_paths


#load_model()