import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
from utils import helpers

API_COMPARE_URL = "http://localhost:8000/api/visual_comparison"

def image_to_base64(image: Image.Image) -> str:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def render_comparison_page(model=None): # Pastram argumentul pentru compatibilitate cu app.py
    st.header("AI Visual Comparison (Side-by-Side)")
    st.info("Upload two retinal images to ask MedGemma to compare them for pathological differences.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Image A")
        img1_file = st.file_uploader("Upload First Image", type=["png", "jpg", "jpeg"], key="ai_comp_1")
        img1 = None
        if img1_file is not None:
            img1 = helpers.load_image(img1_file)
            st.image(img1, caption="Image A", width='stretch')

    with col2:
        st.markdown("### Image B")
        img2_file = st.file_uploader("Upload Second Image", type=["png", "jpg", "jpeg"], key="ai_comp_2")
        img2 = None
        if img2_file is not None:
            img2 = helpers.load_image(img2_file)
            st.image(img2, caption="Image B", width='stretch')

    st.divider()
    
    question = st.text_input(
        "Ask a clinical question about these two images:", 
        value="Compare these two retinal images carefully. What are the key pathological differences between them?"
    )
    
    if st.button("Generate Comparison Report", type="primary", width='stretch'):
        if img1 is None or img2 is None:
            st.warning("Please upload both Image A and Image B before generating the comparison.")
        else:
            with st.spinner("MedGemma is analyzing and comparing the images via API..."):
                try:
                    payload = {
                        "image1_base64": image_to_base64(img1),
                        "image2_base64": image_to_base64(img2),
                        "question": question
                    }
                    
                    response = requests.post(API_COMPARE_URL, json=payload)
                    response.raise_for_status()
                    
                    result_text = response.json().get("result", "No result returned.")
                    
                    st.success("Comparison Analysis Complete")
                    
                    # Afisam rezultatul intr-un container stilizat
                    with st.container(border=True):
                        st.markdown("###AI Clinical Report")
                        st.write(result_text)
                    
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Please ensure `python api.py` is running.")
                except Exception as e:
                    st.error(f"Error during comparison: {e}")