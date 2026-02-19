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



st.set_page_config(page_title="MedGemma Local Ophthalmology Assistant", layout="wide")

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


# --- Tabs ---
tabs = st.tabs(["Diagnosis", "Comparison", "Image Processing", "AI Comparison between 2 images","Research Audit"])

# ----------------- Diagnosis Tab -----------------
with tabs[0]:
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
                #tensor = helpers.convert_to_tensor(img)
                result = model.generate_diagnosis(img,clinical_context)
                st.subheader("Diagnosis Results")
               # Safe extraction of dictionary data
                condition = result.get("condition", "Unspecified").upper()
                severity = str(result.get("severity", "N/A")).title()
                confidence = str(result.get("confidence", "N/A")).upper()
                
                # Handling findings (checking if the model returned a list or string)
                findings = result.get("findings", "No detailed findings available.")
                if isinstance(findings, list):
                    findings = "\n".join([f"- {f}" for f in findings])
                    
                # Handling recommendations (checking if the model returned a list or string)
                recommendations = result.get("recommendations", "No specific recommendations.")
                if isinstance(recommendations, list):
                    recommendations = "\n".join([f"- {r}" for r in recommendations])

                # Visual display of the primary diagnosis
                st.success(f"**Detected Condition:** {condition}")

                # Using metrics for short parameters
                col1, col2 = st.columns(2)
                col1.metric("Severity Level", severity)
                col2.metric("AI Confidence", confidence)

                # Displaying textual details in colored panels
                st.markdown("### Clinical Findings")
                st.info(findings)

                st.markdown("### Recommendations")
                st.warning(recommendations)
            except Exception as e:
                st.error(f"Error during diagnosis: {e}")


# ----------------- Comparison Tab -----------------
with tabs[1]:
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

                # Align images
                aligned_a, aligned_b = image_comparator.align_images(np_a, np_b)
                # Difference map & heatmap
                diff_map = image_comparator.compute_difference_map(aligned_a, aligned_b)
                heatmap = image_comparator.generate_heatmap_overlay(aligned_a, diff_map)
                # Similarity
                score = metrics.compute_similarity(aligned_a, aligned_b, sim_metric)

                st.subheader("Similarity Score")
                st.write(f"{sim_metric}: {score:.4f}")

                st.subheader("Aligned Images")
                col1, col2 = st.columns(2)
                col1.image(aligned_a, caption="Aligned Image A", use_column_width=True)
                col2.image(aligned_b, caption="Aligned Image B", use_column_width=True)

                st.subheader("Difference Heatmap")
                st.image(heatmap, caption="Heatmap Overlay", use_column_width=True)
            except Exception as e:
                st.error(f"Error during comparison: {e}")


# ----------------- Image Processing Tab -----------------
with tabs[2]:
    st.header("Image Processing")
    col1, col2 = st.columns(2)
    with col1:
        img_file1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"], key="proc1")
    with col2:
        img_file2 = st.file_uploader("Upload Image 2 (optional)", type=["png", "jpg", "jpeg"], key="proc2")

    sync_mode = st.checkbox("Synchronized Mode (apply operations to both images)")

    # Operation selection
    ops = {
        "Otsu Threshold": operations.otsu_threshold,
        "Multi Threshold": operations.multi_threshold,
        "Canny Edges": operations.canny_edges,
        "Contours": operations.detect_contours,
        "Histogram Equalization": operations.histogram_equalization,
        "CLAHE": operations.clahe_equalization,
        "Brightness": operations.adjust_brightness,
        "Contrast": operations.adjust_contrast,
        # "Rotate": operations.rotate_image,
        # "Gaussian Blur": operations.gaussian_blur,
    }

    selected_ops = []
    for op_name in ops.keys():
        if st.checkbox(op_name):
            selected_ops.append(op_name)

    # Parameter sliders
    multi_thresh_values = st.text_input("Multi Thresholds (comma-separated)", value="50,100,150")
    canny_low = st.slider("Canny Low Threshold", 0, 255, 50)
    canny_high = st.slider("Canny High Threshold", 0, 255, 150)
    brightness_val = st.slider("Brightness Adjustment", -100, 100, 0)
    contrast_val = st.slider("Contrast Adjustment", 0.1, 3.0, 1.0)
    rotate_angle = st.slider("Rotation Angle", -180, 180, 0)
    
    gaussian_ksize = st.slider("Gaussian Kernel Size", 1, 31, 3, step=2)

    run_processing = st.button("Apply Operations")

    if run_processing:
        if img_file1 is None:
            st.warning("Please upload at least one image.")
        else:
            try:
                img1 = np.array(helpers.load_image(img_file1))
                img2 = np.array(helpers.load_image(img_file2)) if img_file2 else None

                def apply_ops(img):
                    for op in selected_ops:
                        if op == "Multi Threshold":
                            thresholds = [int(x.strip()) for x in multi_thresh_values.split(",") if x.strip()]
                            img = ops[op](img, thresholds)
                        elif op == "Canny Edges":
                            img = ops[op](img, canny_low, canny_high)
                        elif op == "Brightness":
                            img = ops[op](img, brightness_val)
                        elif op == "Contrast":
                            img = ops[op](img, contrast_val)
                        elif op == "Rotate":
                            img = ops[op](img, rotate_angle)
                        elif op == "Gaussian Blur":
                            img = ops[op](img, gaussian_ksize)
                        else:
                            img = ops[op](img)
                    return img

                proc_img1 = apply_ops(img1)
                st.subheader("Processed Image 1")
                st.image(proc_img1, use_column_width=True)

                if img2 is not None:
                    if sync_mode:
                        proc_img2 = apply_ops(img2)
                    else:
                        proc_img2 = img2
                    st.subheader("Image 2")
                    st.image(proc_img2, use_column_width=True)

            except Exception as e:
                st.error(f"Error during image processing: {e}")


###Pentru pagina de comparatie vizuala intre 2 imagini (Tab 3) - combinare spatiala si analiza directa a diferentei intre doua imagini
with tabs[3]:
    render_comparison_page(model)

    # ----------------- Research Audit Tab -----------------
with tabs[4]:
    render_research_audit_page(model)