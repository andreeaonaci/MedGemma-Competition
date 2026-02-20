import streamlit as st
import numpy as np
import cv2
from PIL import Image

from models.medgemma_wrapper import load_model
from comparison import image_comparator
from similarity import metrics
from image_processing import operations
from utils import helpers
from ai_comparison_view import render_comparison_page
from research_audit_view import render_research_audit_page

# 1. Configurare obligatorie pe prima linie
st.set_page_config(page_title="MedGemma Local Ophthalmology Assistant", layout="wide")

# --- LOGICA DE NAVIGARE SI TRANSFER (Integrată) ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Diagnosis"

if 'transferred_image' not in st.session_state:
    st.session_state.transferred_image = None

# Buffer pentru a păstra imaginea procesată vizibilă și după procesare
if 'processed_buffer' not in st.session_state:
    st.session_state.processed_buffer = None

st.title("MedGemma Ophthalmology Diagnostic Assistant (Local)")
st.info(
    "Experimental application\n"
    "Not a certified medical device\n"
    "Does not replace clinical judgment"
)

# --- Load model once ---
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = get_model()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("📍 Navigation")
    page_options = ["Diagnosis", "Comparison", "Image Processing", "AI Comparison between 2 images", "Research Audit"]
    
    selection = st.radio(
        "Go to:", 
        page_options, 
        index=page_options.index(st.session_state.current_page)
    )
    
    if selection != st.session_state.current_page:
        st.session_state.current_page = selection
        st.rerun()

# ----------------- Pagina: Diagnosis -----------------
if st.session_state.current_page == "Diagnosis":
    st.header("Diagnosis")
    uploaded_image = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg"])
    clinical_context = st.text_area("Clinical Context")
    run_button = st.button("Generate Diagnosis")

    if run_button:
        if uploaded_image is None or not clinical_context.strip():
            st.warning("Please provide image and clinical context.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            try:
                img = helpers.load_image(uploaded_image)
                result = model.generate_diagnosis(img, clinical_context)
                st.subheader("Diagnosis Results")
                condition = result.get("condition", "Unspecified").upper()
                severity = str(result.get("severity", "N/A")).title()
                confidence = str(result.get("confidence", "N/A")).upper()
                
                findings = result.get("findings", "No detailed findings available.")
                if isinstance(findings, list):
                    findings = "\n".join([f"- {f}" for f in findings])
                
                recommendations = result.get("recommendations", "No specific recommendations.")
                if isinstance(recommendations, list):
                    recommendations = "\n".join([f"- {r}" for r in recommendations])

                st.success(f"**Detected Condition:** {condition}")
                col1, col2 = st.columns(2)
                col1.metric("Severity Level", severity)
                col2.metric("AI Confidence", confidence)

                st.markdown("### Clinical Findings")
                st.info(findings)
                st.markdown("### Recommendations")
                st.warning(recommendations)
            except Exception as e:
                st.error(f"Error during diagnosis: {e}")

# ----------------- Pagina: Comparison -----------------
elif st.session_state.current_page == "Comparison":
    st.header("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        image_a_file = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"], key="comp_a")
    with col2:
        image_b_file = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"], key="comp_b")

    sim_metric = st.selectbox("Select Similarity Metric", options=list(metrics._METRIC_REGISTRY.keys()))
    run_comparison = st.button("Compute Comparison")

    if run_comparison:
        if image_a_file is None or image_b_file is None:
            st.warning("Please upload both images.")
        else:
            try:
                img_a = helpers.load_image(image_a_file)
                img_b = helpers.load_image(image_b_file)
                np_a = np.array(img_a)
                np_b = np.array(img_b)
                aligned_a, aligned_b = image_comparator.align_images(np_a, np_b)
                diff_map = image_comparator.compute_difference_map(aligned_a, aligned_b)
                heatmap = image_comparator.generate_heatmap_overlay(aligned_a, diff_map)
                score = metrics.compute_similarity(aligned_a, aligned_b, sim_metric)

                st.subheader("Similarity Score")
                st.write(f"{sim_metric}: {score:.4f}")
                col1, col2 = st.columns(2)
                col1.image(aligned_a, caption="Aligned Image A", use_container_width=True)
                col2.image(aligned_b, caption="Aligned Image B", use_container_width=True)
                st.image(heatmap, caption="Difference Heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Error during comparison: {e}")

# ----------------- Pagina: Image Processing -----------------
elif st.session_state.current_page == "Image Processing":
    st.header("Image Processing")
    col1, col2 = st.columns(2)
    with col1:
        img_file1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"], key="proc1")
    with col2:
        img_file2 = st.file_uploader("Upload Image 2 (optional)", type=["png", "jpg", "jpeg"], key="proc2")

    sync_mode = st.checkbox("Synchronized Mode")
    
    ops = {
        "Otsu Threshold": operations.otsu_threshold,
        "Multi Threshold": operations.multi_threshold,
        "Canny Edges": operations.canny_edges,
        "Contours": operations.detect_contours,
        "Histogram Equalization": operations.histogram_equalization,
        "CLAHE": operations.clahe_equalization,
        "Brightness": operations.adjust_brightness,
        "Contrast": operations.adjust_contrast,
    }

    selected_ops = [name for name in ops.keys() if st.checkbox(name)]
    multi_thresh_values = st.text_input("Multi Thresholds", value="50,100,150")
    canny_low = st.slider("Canny Low", 0, 255, 50)
    canny_high = st.slider("Canny High", 0, 255, 150)
    brightness_val = st.slider("Brightness", -100, 100, 0)
    contrast_val = st.slider("Contrast", 0.1, 3.0, 1.0)

    run_processing = st.button("Apply Operations")

    # Logica de procesare
    if run_processing:
        if img_file1 is None:
            st.warning("Please upload an image.")
        else:
            try:
                img1 = np.array(helpers.load_image(img_file1))
                img2 = np.array(helpers.load_image(img_file2)) if img_file2 else None
                
                # --- Capturăm detaliile operațiilor pentru a le oferi ca și context AI-ului ---
                st.session_state.original_buffer = img1  # Păstrăm imaginea originală pentru referință
                ops_details = []

                def apply_ops(img):
                    for op in selected_ops:
                        if op == "Multi Threshold":
                            thresholds = [int(x.strip()) for x in multi_thresh_values.split(",") if x.strip()]
                            img = ops[op](img, thresholds)
                            ops_details.append(f"Multi Threshold({multi_thresh_values})")
                        elif op == "Canny Edges":
                            img = ops[op](img, canny_low, canny_high)
                            ops_details.append(f"Canny Edges(Low:{canny_low}, High:{canny_high})")
                        elif op == "Brightness":
                            img = ops[op](img, brightness_val)
                            ops_details.append(f"Brightness({brightness_val})")
                        elif op == "Contrast":
                            img = ops[op](img, contrast_val)
                            ops_details.append(f"Contrast({contrast_val})")
                        else:
                            img = ops[op](img)
                            ops_details.append(op)
                    return img

                st.session_state.processed_buffer = apply_ops(img1)
                
                # Salvăm string-ul de context in buffer
                st.session_state.processed_buffer_ops = ", ".join(ops_details) if ops_details else "No operations applied."
                
                if img2 is not None:
                    st.session_state.processed_buffer_img2 = apply_ops(img2) if sync_mode else img2
            except Exception as e:
                st.error(f"Error: {e}")

    # AFIȘARE REZULTATE ȘI BUTON TRANSFER (Unic)
    if st.session_state.processed_buffer is not None:
        st.subheader("Processed Image 1")
        st.image(st.session_state.processed_buffer, use_container_width=True)
        
        # Arătăm utilizatorului ce metadate vor fi transmise
        st.caption(f"🔧 **Metadata to transfer:** {st.session_state.get('processed_buffer_ops', 'None')}")

        # --- BUTONUL DE REDIRECȚIONARE ---
        if st.button("🚀 Send to Research Audit & Analyze", type="primary", key="transfer_btn_audit_1"):
            st.session_state.transferred_image = st.session_state.processed_buffer
            # Transferăm și metadatele
            st.session_state.transferred_ops = st.session_state.get('processed_buffer_ops', '')
            st.session_state.transferred_original=st.session_state.get('original_buffer', None)

            st.session_state.current_page = "Research Audit"
            st.rerun()

        if img_file2 and 'processed_buffer_img2' in st.session_state:
            st.subheader("Image 2")
            st.image(st.session_state.processed_buffer_img2, use_container_width=True)

# ----------------- Pagina: AI Comparison -----------------
elif st.session_state.current_page == "AI Comparison between 2 images":
    render_comparison_page(model)

# ----------------- Pagina: Research Audit -----------------
elif st.session_state.current_page == "Research Audit":
    render_research_audit_page(model)