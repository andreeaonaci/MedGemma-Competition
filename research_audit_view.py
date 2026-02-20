import streamlit as st
import csv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import string
from datetime import datetime
from utils import helpers

def render_research_audit_page(model):
    st.header("Clinical Research & Audit")
    
    if 'audit_limit' not in st.session_state:
        st.session_state.audit_limit = 10
    
    sub_tabs = st.tabs(["🩺 Data Collection", "📊 Performance Dashboard"])
    
    # --- SUB-TAB 1: Data Collection ---
    with sub_tabs[0]:
        st.info("Log human diagnosis and compare it with AI inference for statistical benchmarking.")

        col1, col2 = st.columns(2)
        
        # Variabile suport pentru logica de analiza
        image_to_analyze = None
        img_name = "unknown_image.png"

        with col1:
            st.markdown("### 🖼️ Input Image")
            
            # 1. Prioritate: Verificăm dacă există o imagine transferată din Lab-ul de Procesare
            transferred_img = st.session_state.get('transferred_image', None)
            
            if transferred_img is not None:
                st.success("✅ Processed image loaded automatically from Lab.")
                image_to_analyze = transferred_img
                img_name = "processed_transfer.png" # Nume generic pentru logare CSV
                
                # Afisare imagine transferata
                st.image(image_to_analyze, caption="Ready for AI Analysis", use_container_width=True)
                
                # Opțiune de a anula transferul și a reveni la încărcarea manuală
                if st.button("🗑️ Discard processed image & upload new"):
                    st.session_state['transferred_image'] = None
                    st.rerun()
            else:
                # 2. Dacă nu există transfer, afișăm uploader-ul standard
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
                    with st.spinner("MedGemma is analyzing the image..."):
                        # Analizăm imaginea (fie ea transferată sau încărcată manual)
                        result = model.generate_diagnosis(image_to_analyze, clinical_context)
                    
                    condition = result.get("condition", "Unspecified").upper()
                    severity = str(result.get("severity", "N/A")).title()
                    confidence = str(result.get("confidence", "N/A")).upper()
                    
                    st.success(f"**AI Diagnosis:** {condition} | **Severity:** {severity} | **Confidence:** {confidence}")
                    
                    # --- Salvare Date în CSV ---
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
                        
                        # Setările cerute: width=0.5 pentru barele albastre
                        fig1.add_trace(go.Bar(
                            x=doc_f.values, 
                            y=doc_f.index, 
                            orientation='h', 
                            width=0.5, 
                            name="Doctor (Ground Truth)", 
                            marker_color='#3b82f6'
                        ))
                        
                        # Setările cerute: line-ns, size 24, width 2 pentru liniile negre
                        fig1.add_trace(go.Scatter(
                            x=ai_f.values, 
                            y=ai_f.index, 
                            mode='markers', 
                            cliponaxis=False, 
                            marker=dict(symbol='line-ns', size=24, line=dict(color='black', width=2)), 
                            name="AI Output"
                        ))
                        
                        # Adnotările aliniate corect deasupra barelor
                        annotations = [dict(
                            xref='paper', 
                            yref='y', 
                            x=0, 
                            y=c, 
                            text=f"<b>{c}</b>", 
                            showarrow=False, 
                            xanchor='left', 
                            yanchor='bottom', 
                            yshift=16, 
                            font=dict(size=13, color="#1f2937")
                        ) for c in displayed]
                        
                        max_x = max(max(doc_dist.values, default=0), max(ai_dist.values, default=0)) + 5
                        
                        fig1.update_layout(
                            xaxis_title="Percentage of Total Cases (%)", 
                            yaxis=dict(showticklabels=False, autorange="reversed"), 
                            xaxis=dict(range=[-2, max_x], zeroline=False, showgrid=True), 
                            annotations=annotations,
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                            barmode='overlay',
                            margin=dict(l=0, r=0, t=70, b=10),
                            height=100 + (len(displayed) * 60),
                            transition=dict(duration=400, easing="cubic-in-out")
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Logica butoanelor Show More / Collapse
                        if len(filtered_conditions) > 10:
                            c1, c2, _ = st.columns([1, 1, 4])
                            if st.session_state.audit_limit < len(filtered_conditions):
                                if c1.button("🔽 Show More (+10)"): 
                                    st.session_state.audit_limit += 10
                                    st.rerun()
                            if st.session_state.audit_limit > 10:
                                if c2.button("🔼 Collapse All"): 
                                    st.session_state.audit_limit = 10
                                    st.rerun()
                    
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