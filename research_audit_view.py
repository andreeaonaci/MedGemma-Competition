import streamlit as st
import csv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import string
from datetime import datetime
from PIL import Image
import numpy as np
from utils import helpers

# Funcție utilitară pentru a lipi două imagini stânga-dreapta
def combine_side_by_side(img1, img2):
    i1 = Image.fromarray(img1) if isinstance(img1, np.ndarray) else img1
    i2 = Image.fromarray(img2) if isinstance(img2, np.ndarray) else img2
    
    if i1.mode != 'RGB': i1 = i1.convert('RGB')
    if i2.mode != 'RGB': i2 = i2.convert('RGB')
    
    # Redimensionare proporțională pentru a avea aceeași înălțime
    new_h = max(i1.height, i2.height)
    i1 = i1.resize((int(i1.width * new_h / i1.height), new_h))
    i2 = i2.resize((int(i2.width * new_h / i2.height), new_h))
    
    # Creare canvas nou și lipirea imaginilor
    dst = Image.new('RGB', (i1.width + i2.width, new_h))
    dst.paste(i1, (0, 0))
    dst.paste(i2, (i1.width, 0))
    return dst

def render_research_audit_page(model):
    st.header("Clinical Research & Audit")
    
    if 'audit_limit' not in st.session_state:
        st.session_state.audit_limit = 10
    
    sub_tabs = st.tabs(["🩺 Data Collection", "📊 Performance Dashboard"])
    
    # --- SUB-TAB 1: Data Collection ---
    with sub_tabs[0]:
        st.info("Log human diagnosis and compare it with AI inference for statistical benchmarking.")

        col1, col2 = st.columns(2)
        
        image_to_analyze = None
        img_name = "unknown_image.png"
        processing_metadata = "" 
        is_composite_image = False # Steag pentru a ști cum să setăm promptul

        with col1:
            st.markdown("### 🖼️ Diagnostic Imagery")
            
            # Verificăm dacă avem date transferate
            transferred_img = st.session_state.get('transferred_image', None)
            transferred_orig = st.session_state.get('transferred_original', None)
            
            # CAZUL 1: Avem imagini din Lab-ul de procesare
            if transferred_img is not None and transferred_orig is not None:
                st.success("✅ Original & Processed images loaded from Lab.")
                processing_metadata = st.session_state.get('transferred_ops', '')
                
                if processing_metadata:
                    st.info(f"**Applied Filters:** {processing_metadata}")
                
                # Afișăm ambele imagini medicului
                prev_col1, prev_col2 = st.columns(2)
                with prev_col1:
                    st.image(transferred_orig, caption="Original", use_container_width=True)
                with prev_col2:
                    st.image(transferred_img, caption="Processed", use_container_width=True)
                
                # Lipim imaginile în fundal pentru AI
                image_to_analyze = combine_side_by_side(transferred_orig, transferred_img)
                img_name = "composite_transfer.png"
                is_composite_image = True
                
                if st.button("🗑️ Discard images & upload new"):
                    st.session_state['transferred_image'] = None
                    st.session_state['transferred_original'] = None
                    st.session_state['transferred_ops'] = None
                    st.rerun()
            
            # CAZUL 2: Încărcare manuală clasică (fără procesare)
            else:
                uploaded_file = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"], key="audit_img")
                if uploaded_file is not None:
                    image_to_analyze = helpers.load_image(uploaded_file)
                    img_name = uploaded_file.name
                    st.image(image_to_analyze, caption="Manual Upload Preview", use_container_width=True)
        
        with col2:
            st.markdown("### 👨‍⚕️ Clinical Ground Truth")
            doctor_diagnosis = st.selectbox(
                "Select human diagnosis (Ground Truth):",
                ["DIABETIC RETINOPATHY", "GLAUCOMA", "AGE-RELATED MACULAR DEGENERATION", "HEALTHY", "OTHER"],
                key="audit_doc_diag"
            )
            
            clinical_context = st.text_input("Clinical Context (Optional)", value="Patient routine checkup.", key="audit_ctx")
            
            st.divider()
            run_audit = st.button("⚡ Generate & Log AI Diagnosis", use_container_width=True)

        if run_audit:
            if image_to_analyze is None:
                st.warning("Please provide an image first (upload or transfer from Processing).")
            elif model is None:
                st.error("Model not loaded.")
            else:
                try:
                    with st.spinner("MedGemma is analyzing the data..."):
                        
                        final_ai_context = clinical_context
                        
                        # --- INJECTAREA CONTEXTULUI PANORAMIC PENTRU AI ---
                        if is_composite_image and processing_metadata:
                            final_ai_context = f"{clinical_context} | [CRITICAL INSTRUCTION FOR AI: The provided visual input is a single side-by-side composite image containing two views of the same eye. The LEFT half is the original unmodified image. The RIGHT half is the same image digitally processed using the following techniques: {processing_metadata}. Please analyze both sides holistically to form your diagnosis, using the right side for enhanced features but relying on the left side to confirm they are not artificial filter artifacts.]"
                        
                        result = model.generate_diagnosis(image_to_analyze, final_ai_context)
                    
                    condition = result.get("condition", "Unspecified").upper()
                    severity = str(result.get("severity", "N/A")).title()
                    confidence = str(result.get("confidence", "N/A")).upper()
                    
                    st.success(f"**AI Diagnosis:** {condition} | **Severity:** {severity} | **Confidence:** {confidence}")
                    
                    # Salvare în CSV
                    log_file = "clinical_audit_log.csv"
                    file_exists = os.path.isfile(log_file)
                    
                    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Timestamp", "Image_Name", "Doctor_Diagnosis", "AI_Condition", "AI_Severity", "AI_Confidence"])
                        
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            img_name,
                            doctor_diagnosis,
                            condition,
                            severity,
                            confidence
                        ])
                    st.info(f"💾 Entry successfully added to `{log_file}`")
                    
                except Exception as e:
                    st.error(f"Error during audit generation: {e}")

    # --- SUB-TAB 2: Performance Dashboard ---
    with sub_tabs[1]:
        st.subheader("Statistical Benchmarking")
        log_file = "clinical_audit_log.csv"
        
        if os.path.isfile(log_file):
            try:
                df = pd.read_csv(log_file)
                if not df.empty:
                    df['Doctor_Diagnosis'] = df['Doctor_Diagnosis'].astype(str).str.strip().str.upper()
                    df['AI_Condition'] = df['AI_Condition'].astype(str).str.strip().str.upper()
                    df['Match'] = df['Doctor_Diagnosis'] == df['AI_Condition']
                    
                    total_cases = len(df)
                    accuracy = df['Match'].mean() * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Total Cases Analyzed", total_cases)
                    col_m2.metric("Overall AI Accuracy", f"{accuracy:.1f}%")
                    col_m3.metric("Total Discrepancies", total_cases - df['Match'].sum())
                    
                    st.divider()
                    st.markdown("### Diagnostic Distribution (Drift)")
                    st.caption("Bar = Doctor's Distribution | Line = AI's Distribution")
                    
                    doc_dist = (df['Doctor_Diagnosis'].value_counts() / total_cases) * 100
                    ai_dist = (df['AI_Condition'].value_counts() / total_cases) * 100
                    all_conditions = sorted(list(set(doc_dist.index).union(set(ai_dist.index))))
                    
                    alphabet = list(string.ascii_uppercase)
                    start_letter, end_letter = st.select_slider("🔍 Filter Conditions Alphabetically:", options=alphabet, value=('A', 'Z'))
                    
                    filtered_conditions = [c for c in all_conditions if start_letter <= c[0].upper() <= end_letter]
                    
                    if not filtered_conditions:
                        st.info("No conditions in this alphabetical range.")
                    else:
                        displayed = filtered_conditions[:st.session_state.audit_limit]
                        doc_f = doc_dist.reindex(displayed, fill_value=0)
                        ai_f = ai_dist.reindex(displayed, fill_value=0)
                        
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(x=doc_f.values, y=doc_f.index, orientation='h', width=0.5, name="Doctor (Ground Truth)", marker_color='#3b82f6'))
                        fig1.add_trace(go.Scatter(x=ai_f.values, y=ai_f.index, mode='markers', cliponaxis=False, marker=dict(symbol='line-ns', size=24, line=dict(color='black', width=2)), name="AI Output"))
                        
                        annotations = [dict(xref='paper', yref='y', x=0, y=c, text=f"<b>{c}</b>", showarrow=False, xanchor='left', yanchor='bottom', yshift=16, font=dict(size=13, color="#1f2937")) for c in displayed]
                        max_x = max(max(doc_dist.values, default=0), max(ai_dist.values, default=0)) + 5
                        
                        fig1.update_layout(xaxis_title="Percentage of Total Cases (%)", yaxis=dict(showticklabels=False, autorange="reversed"), xaxis=dict(range=[-2, max_x], zeroline=False, showgrid=True), annotations=annotations, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0), barmode='overlay', margin=dict(l=0, r=0, t=70, b=10), height=100 + (len(displayed) * 60), transition=dict(duration=400, easing="cubic-in-out"))
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        if len(filtered_conditions) > 10:
                            c1, c2, _ = st.columns([1, 1, 4])
                            if st.session_state.audit_limit < len(filtered_conditions):
                                if c1.button("🔽 Show More (+10)"): st.session_state.audit_limit += 10; st.rerun()
                            if st.session_state.audit_limit > 10:
                                if c2.button("🔼 Collapse All"): st.session_state.audit_limit = 10; st.rerun()
                    
                    st.divider()
                    st.markdown("### ⚠️ AI Deviation Analysis")
                    deviations = df[~df['Match']]
                    if not deviations.empty:
                        st.dataframe(deviations[['Timestamp', 'Image_Name', 'Doctor_Diagnosis', 'AI_Condition', 'AI_Confidence']], use_container_width=True)
                    else:
                        st.success("No deviations found. AI and Clinical Truth are in 100% agreement.")
            except Exception as e:
                st.error(f"Error reading performance data: {e}")
        else:
            st.info("No audit data found. Generate some diagnoses in the 'Data Collection' tab first.")