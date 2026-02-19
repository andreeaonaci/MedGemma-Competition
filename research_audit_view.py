import streamlit as st
import csv
import os
from datetime import datetime
from utils import helpers

def render_research_audit_page(model):
    st.header("Clinical Research & Audit")
    st.info("Log human diagnosis and compare it with AI inference for statistical benchmarking.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"], key="audit_img")
    
    with col2:
        st.markdown("### 👨‍⚕️ Clinical Ground Truth")
        doctor_diagnosis = st.selectbox(
            "Select human diagnosis (Ground Truth):",
            ["Diabetic Retinopathy", "Glaucoma", "Age-Related Macular Degeneration", "Healthy", "Other"],
            key="audit_doc_diag"
        )
        
    clinical_context = st.text_input("Clinical Context (Optional)", value="Patient routine checkup.", key="audit_ctx")
    run_audit = st.button("Generate & Log AI Diagnosis")

    if run_audit:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            try:
                # 1. Incarcam imaginea
                image = helpers.load_image(uploaded_file)
                
                # 2. Rulam inferenta AI
                with st.spinner("Running AI inference..."):
                    result = model.generate_diagnosis(image, clinical_context)
                
                # 3. Extragem datele
                condition = result.get("condition", "Unspecified").upper()
                severity = str(result.get("severity", "N/A")).title()
                confidence = str(result.get("confidence", "N/A")).upper()
                
                # 4. Afisam rezultatul rapid pe ecran
                st.success(f"**AI Diagnosis:** {condition} | **Severity:** {severity} | **Confidence:** {confidence}")
                
                # 5. Salvarea silentionasa in CSV
                log_file = "clinical_audit_log.csv"
                file_exists = os.path.isfile(log_file)
                img_name = uploaded_file.name
                
                with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # Cream capul de tabel la prima rulare
                        writer.writerow(["Timestamp", "Image_Name", "Doctor_Diagnosis", "AI_Condition", "AI_Severity", "AI_Confidence"])
                    
                    # Scriem randul cu rezultatele
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        img_name,
                        doctor_diagnosis,
                        condition,
                        severity,
                        confidence
                    ])
                st.info(f"💾 Data successfully logged to `{log_file}`")
                
            except Exception as e:
                st.error(f"Error during audit generation: {e}")