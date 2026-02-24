import embeddings
import json
import os

import numpy as np

images_dir = "data/case_images"
#IMAGE_DIR = Path(images_dir)
#image_paths = list(IMAGE_DIR.glob("*.png"))


json_data = "data/cases.json"
processor, model = embeddings.load_model()

def extract_img_paths_description(data_path):
    texts, oct_paths = [], []
    with open(data_path, 'r') as file:
        data = json.load(file)
        for e in data:
            texts.append(e["Interpretation"])
            oct_paths.append(os.path.join(images_dir, e["raw_image"]))
    return texts, oct_paths

def create_cases_embeddings(texts, oct_paths, store=True):
    result = embeddings.generate_embeddings_with_text(texts,
                                                      [embeddings.preprocess_image(img) for img in oct_paths],
                                                      model, processor,
                                                      zero_shot_classification=False)
    print(result)
    if store:
        embeddings.save_embeddings(result["img_embed"], result["text_embed"], oct_paths, "cases.npz" )
    return result




def vector_search(query_vector , vectors, k=5):
    scores = np.dot(vectors, query_vector.T).squeeze()

    top_k_idx = np.argsort(scores)[-k:][::-1]
    return top_k_idx, scores[top_k_idx]



def ensemble(query_text, query_image,
             image_embeddings,
             description_embeddings,
             model,
             processor,
             weight=0.5, image_is_path=False):
    #search by text embedding
    query_emb = embeddings.generate_embedding_only_text([query_text], model, processor)
    text_top_k_idx, text_scores = vector_search(query_emb[0], description_embeddings, k=20 )
    print(text_top_k_idx, text_scores)
    text_dict = dict(zip(text_top_k_idx,text_scores) )

    #search by image embeddings
    if image_is_path:
        query_image = embeddings.preprocess_image(query_image)
    else:
        query_image = embeddings.resize(query_image)
    query_emb = embeddings.generate_embedding_only_image([query_image], model, processor)
    image_top_k_idx, image_scores = vector_search(query_emb[0], image_embeddings, k=30)
    print("aici", image_top_k_idx, image_scores)
    img_dict = dict(zip(image_top_k_idx, image_scores))
    print("dict", img_dict)
    idx = np.unique(np.concatenate((text_top_k_idx, image_top_k_idx)))
    print()
    print("indecsi ", idx, text_top_k_idx, image_top_k_idx)
    print()
    scores = [(weight*(text_dict[i] if i in text_dict else 0) + (1-weight)*(img_dict[i] if i in img_dict else 0),i)
              for i in idx]
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print(scores)
    return [e[1] for e in scores][:5], [e[0] for e in scores][:5]

if __name__ == "__main__":
    texts, oct_paths = extract_img_paths_description(json_data)
    print(texts[:1], oct_paths[:1])
    create_cases_embeddings(texts[:20], oct_paths[:20])
    image_embeddings, description_embeddings, image_paths = embeddings.load_embeddings("cases.npz")

    top_k_idx, scores = vector_search(image_embeddings[0], image_embeddings)
    print(top_k_idx, scores)

    query_text  = "drusen"
    query_emb = embeddings.generate_embedding_only_text([query_text], model, processor)
    top_k_idx, scores = vector_search(query_emb[0], description_embeddings)
    print(top_k_idx, scores)

    query_image = image_paths[0]
    top_k_idx, scores = ensemble(query_text, query_image,  image_embeddings, description_embeddings, model, processor, image_is_path=True)
    print(top_k_idx, scores)




