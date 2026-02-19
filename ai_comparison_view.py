import streamlit as st
from utils import helpers

def render_comparison_page(model):
    st.header("AI Visual Comparison")
    st.info("Upload two retinal images to identify pathological differences.")

    col1, col2 = st.columns(2)
    with col1:
        image_a_file = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"], key="ai_comp_a")
    with col2:
        image_b_file = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"], key="ai_comp_b")
    comparison_question = st.text_area(
        "Clinical Query", 
        value="Compare these two retinal images. What pathological features are visible in Image A but missing in Image B?"
    )
    
    run_ai_comparison = st.button("Generate AI Comparison")

    if run_ai_comparison:
        if image_a_file is None or image_b_file is None:
            st.warning("Please upload both images.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            try:
                img_a = helpers.load_image(image_a_file)
                img_b = helpers.load_image(image_b_file)
                
                with st.spinner("Concatenating spatial data and running inference..."):
                    comparison_result = model.generate_visual_comparison(img_a, img_b, comparison_question)
                
                st.subheader("Comparative Analysis Report")
                st.write(comparison_result)
                
            except Exception as e:
                st.error(f"Error during visual comparison: {e}")