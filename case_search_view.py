import os
import json
import numpy as np
from PIL import Image
import streamlit as st

import embeddings as emb_module
import hybrid_search

CASES_JSON = "data/cases.json"
CASES_NPZ = "cases.npz"
IMAGES_DIR = "data/case_images"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return emb_module.load_model()


@st.cache_resource(show_spinner="Loading embeddings...")
def load_cases_embeddings():
    return emb_module.load_embeddings(CASES_NPZ)


@st.cache_data
def load_cases():
    with open(CASES_JSON) as f:
        data = json.load(f)
    lookup = {case["raw_image"]: case for case in data}
    return data, lookup


def get_case(idx, image_paths, cases_lookup):
    filename = os.path.basename(image_paths[idx])
    return cases_lookup.get(filename)


def render_result(rank, idx, score, image_paths, cases_lookup):
    case = get_case(idx, image_paths, cases_lookup)
    if case is None:
        st.warning(f"Result #{rank}: case not found for index {idx}")
        return

    with st.container(border=True):
        header_col, score_col = st.columns([4, 1])
        with header_col:
            st.markdown(f"**#{rank} — Case {case.get('Case', idx + 1)}**")
        with score_col:
            st.metric("Score", f"{score:.4f}")

        images_col, info_col = st.columns([2, 2])

        with images_col:
            raw_col, fundus_col = st.columns([3, 1])
            with raw_col:
                raw_path = os.path.join(IMAGES_DIR, case["raw_image"])
                if os.path.exists(raw_path):
                    st.image(raw_path, caption="OCT", use_container_width=True)
            with fundus_col:
                fundus_path = os.path.join(IMAGES_DIR, case.get("fundus_image", ""))
                if os.path.exists(fundus_path):
                    st.image(fundus_path, caption="Fundus", use_container_width=True)

            ann_path = os.path.join(IMAGES_DIR, case.get("annotated_image", ""))
            if os.path.exists(ann_path):
                st.image(ann_path, caption="Annotated", use_container_width=True)

        with info_col:
            age = case.get("Age", "N/A")
            location = case.get("Location", "N/A")
            quality = case.get("Scan Quality", "N/A")

            st.markdown(
                f"**Age:** {age}  \n"
                f"**Location:** {location}  \n"
                f"**Scan quality:** {quality}"
            )

            st.markdown("**Interpretation:**")
            st.write(case.get("Interpretation", ""))

            findings = case.get("findings", [])
            if findings:
                st.markdown("**Findings:**")
                for finding in findings:
                    area = finding.get("area", "")
                    st.markdown(f"*{area}*")
                    for content in finding.get("content", []):
                        alt = content.get("alteration", "")
                        desc = content.get("description", "")
                        st.markdown(f"- **{alt}**: {desc}")


# =========================================================
# MAIN RENDER FUNCTION (THIS IS WHAT YOU IMPORT AND CALL)
# =========================================================

def render_case_search_page():

    st.header("Medical Case Ensemble Search")
    st.caption("Search OCT cases by text description, uploaded image, or both.")

    processor, model = load_model()
    image_embeddings, description_embeddings, image_paths = load_cases_embeddings()
    cases_list, cases_lookup = load_cases()

    # Sidebar settings (scoped safely)
    st.sidebar.header("Search settings")
    weight = st.sidebar.slider(
        "Ensemble weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = image only · 1 = text only · 0.5 = balanced",
    )
    st.sidebar.caption("Weight is used only when both text and image are provided.")

    text_col, img_col = st.columns(2)

    with text_col:
        query_text = st.text_area(
            "Text query",
            placeholder="e.g. drusen, AMD, EZ disruption, subretinal fluid…",
            height=120,
        )

    with img_col:
        uploaded_file = st.file_uploader(
            "Query image (OCT)",
            type=["png", "jpg", "jpeg"]
        )
        query_image = None
        if uploaded_file:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Query image", use_container_width=True)

    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked:
        has_text = bool(query_text.strip())
        has_image = query_image is not None

        if not has_text and not has_image:
            st.warning("Please provide at least a text query or an image.")
            return

        mode_label = (
            "ensemble (text + image)"
            if has_text and has_image
            else ("text" if has_text else "image")
        )

        with st.spinner(f"Searching by {mode_label}…"):

            if has_text and has_image:
                top_idx, scores = hybrid_search.ensemble(
                    query_text.strip(),
                    query_image,
                    image_embeddings,
                    description_embeddings,
                    model,
                    processor,
                    weight=weight,
                    image_is_path=False,
                )

            elif has_text:
                query_emb = emb_module.generate_embedding_only_text(
                    [query_text.strip()], model, processor
                )
                top_idx, scores = hybrid_search.vector_search(
                    query_emb[0], description_embeddings
                )

            else:
                resized = emb_module.resize(query_image)
                query_emb = emb_module.generate_embedding_only_image(
                    [resized], model, processor
                )
                top_idx, scores = hybrid_search.vector_search(
                    query_emb[0], image_embeddings
                )

        st.divider()
        st.subheader(f"Top {len(top_idx)} results — {mode_label}")

        for rank, (idx, score) in enumerate(zip(top_idx, scores), start=1):
            render_result(rank, int(idx), float(score), image_paths, cases_lookup)