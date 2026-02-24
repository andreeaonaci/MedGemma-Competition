import os
import json
from PIL import Image
import requests
import streamlit as st
import numpy as np
import cv2
import embeddings as emb_module
import hybrid_search

# IMPORT NOU: Aducem pagina de chat
from diagnostic_chat_view import render_diagnostic_chat_page

# Importurile vechi raman pentru celelalte functionalitati
from comparison import image_comparator
from similarity import metrics
from image_processing import operations
from utils import helpers
from ai_comparison_view import render_comparison_page
from research_audit_view import render_research_audit_page
from case_search_view import render_case_search_page

CASES_JSON = "data/cases.json"
CASES_NPZ = "cases.npz"
IMAGES_DIR = "data/case_images"


# 1. Configurare obligatorie pe prima linie
st.set_page_config(page_title="MedGemma Local Ophthalmology Assistant", layout="wide")

# --- LOGICA DE NAVIGARE SI TRANSFER ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Diagnosis"

if 'transferred_image' not in st.session_state:
    st.session_state.transferred_image = None

if 'processed_buffer' not in st.session_state:
    st.session_state.processed_buffer = None

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 MedGemma Ophthalmology Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Disclaimer:</strong><br>
        • Experimental research application<br>
        • Not a certified medical device<br>
        • Does not replace clinical judgment
    </div>
""", unsafe_allow_html=True)

# --- ELIMINAREA MODELULUI LOCAL ---
# Modelul este acum rulat de serverul FastAPI in fundal.
# Setam model=None pentru a nu "sparge" tab-urile vechi care il asteptau ca parametru.
model = None 

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Define pages with icons
    page_options = {
        "🩺 Diagnosis": "Diagnosis",
        "🔬 Comparison": "Comparison",
        "🖼️ Image Processing": "Image Processing",
        "🤖 AI Comparison": "AI Comparison between 2 images",
        "📊 Research Audit": "Research Audit",
    }
    
    st.write("---")
    
    for display_name, page_key in page_options.items():
        btn_type = "primary" if st.session_state.current_page == page_key else "secondary"
        
        if st.button(display_name, type=btn_type, use_container_width=True):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()
    
    st.write("---")
    st.caption("💡 Powered by MedGemma 4B")

# ----------------- Pagina: Diagnosis (ACUM ESTE CHAT INTERACTIV) -----------------
if st.session_state.current_page == "Diagnosis":
    # Aici apelam noua noastra functie care comunica cu API-ul
    render_diagnostic_chat_page()

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
                col1.image(aligned_a, caption="Aligned Image A", width='stretch')
                col2.image(aligned_b, caption="Aligned Image B", width='stretch')
                st.image(heatmap, caption="Difference Heatmap", width='stretch')
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

    if run_processing:
        if img_file1 is None:
            st.warning("Please upload an image.")
        else:
            try:
                img1 = np.array(helpers.load_image(img_file1))
                img2 = np.array(helpers.load_image(img_file2)) if img_file2 else None
                
                st.session_state.original_buffer = img1  
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
                st.session_state.processed_buffer_ops = ", ".join(ops_details) if ops_details else "No operations applied."
                
                if img2 is not None:
                    st.session_state.processed_buffer_img2 = apply_ops(img2) if sync_mode else img2
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.processed_buffer is not None:
        st.subheader("Processed Image 1")
        st.image(st.session_state.processed_buffer, width='stretch')
        st.caption(f"Metadata to transfer: {st.session_state.get('processed_buffer_ops', 'None')}")

        if st.button("Send to Research Audit & Analyze", type="primary", key="transfer_btn_audit_1"):
            st.session_state.transferred_image = st.session_state.processed_buffer
            st.session_state.transferred_ops = st.session_state.get('processed_buffer_ops', '')
            st.session_state.transferred_original=st.session_state.get('original_buffer', None)

            st.session_state.current_page = "Research Audit"
            st.rerun()

        if img_file2 and 'processed_buffer_img2' in st.session_state:
            st.subheader("Image 2")
            st.image(st.session_state.processed_buffer_img2, width='stretch')

# ----------------- Pagina: AI Comparison -----------------
elif st.session_state.current_page == "AI Comparison between 2 images":
    render_comparison_page(model)

# ----------------- Pagina: Research Audit -----------------
elif st.session_state.current_page == "Research Audit":
    render_research_audit_page(model)

# elif st.session_state.current_page == "Explainability":
#     st.header("🔍 Attention Rollout Heatmap")
#     st.markdown("Visualize which regions of the retinal image the AI model focuses on during analysis.")
#     st.write("")
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         uploaded_file = st.file_uploader(
#             "📤 Upload Retinal Image",
#             type=["png", "jpg", "jpeg"],
#             key="heatmap_upload",
#             help="Upload an OCT or fundus image for attention analysis"
#         )
        
#         if uploaded_file:
#             st.image(uploaded_file, caption="Original Image", use_container_width=True)
    
#     with col2:
#         clinical_context = st.text_area(
#             "📝 Clinical Context (Optional)",
#             key="heatmap_context",
#             placeholder="Enter any relevant clinical information...",
#             height=150
#         )
        
#         st.write("")
#         generate_btn = st.button(
#             "🎨 Generate Attention Heatmap",
#             type="primary",
#             use_container_width=True
#         )
    
#     if generate_btn:
#         if uploaded_file is None:
#             st.warning("⚠️ Please upload an image first.")
#         else:
#             with st.spinner("🔄 Generating attention heatmap..."):
#                 try:
#                     response = requests.post(
#                         "http://localhost:8000/attention-heatmap",
#                         files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
#                         data={"context": clinical_context}
#                     )
                    
#                     if response.status_code == 200:
#                         st.success("✅ Heatmap generated successfully!")
#                         st.divider()
#                         st.subheader("Attention Heatmap Result")
#                         st.image(response.content, caption="Model Attention Heatmap", use_container_width=True)
#                         st.info("💡 Warmer colors (red/yellow) indicate regions the model focused on most.")
#                     else:
#                         error_detail = response.json().get("detail", "Unknown error")
#                         st.error(f"❌ Failed to generate heatmap: {error_detail}")
#                 except Exception as e:
#                     st.error(f"❌ Error connecting to backend: {str(e)}")

# elif st.session_state.current_page == "Case Search":
#     render_case_search_page()

