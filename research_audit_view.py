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
        with col1:
            uploaded_file = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"], key="audit_img")
        
        with col2:
            st.markdown("### 👨‍⚕️ Clinical Ground Truth")
            doctor_diagnosis = st.selectbox(
                "Select human diagnosis (Ground Truth):",
                ["DIABETIC RETINOPATHY", "GLAUCOMA", "AGE-RELATED MACULAR DEGENERATION", "HEALTHY", "OTHER"],
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
                    image = helpers.load_image(uploaded_file)
                    
                    with st.spinner("Running AI inference..."):
                        result = model.generate_diagnosis(image, clinical_context)
                    
                    condition = result.get("condition", "Unspecified").upper()
                    severity = str(result.get("severity", "N/A")).title()
                    confidence = str(result.get("confidence", "N/A")).upper()
                    
                    st.success(f"**AI Diagnosis:** {condition} | **Severity:** {severity} | **Confidence:** {confidence}")
                    
                    log_file = "clinical_audit_log.csv"
                    file_exists = os.path.isfile(log_file)
                    img_name = uploaded_file.name
                    
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
                    st.info(f"💾 Data successfully logged to `{log_file}`")
                    
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
                    start_letter, end_letter = st.select_slider(
                        "🔍 Filter Conditions Alphabetically:",
                        options=alphabet,
                        value=('A', 'Z')
                    )
                    
                    filtered_conditions = [
                        cond for cond in all_conditions 
                        if start_letter <= cond[0].upper() <= end_letter
                    ]
                    
                    total_filtered = len(filtered_conditions)
                    
                    if total_filtered == 0:
                        st.info(f"No conditions found starting with letters between '{start_letter}' and '{end_letter}'.")
                    else:
                        displayed_conditions = filtered_conditions[:st.session_state.audit_limit]
                        
                        doc_dist_filtered = doc_dist.reindex(displayed_conditions, fill_value=0)
                        ai_dist_filtered = ai_dist.reindex(displayed_conditions, fill_value=0)
                        
                        fig1 = go.Figure()
                        
                        # Ream pus grosimea optima de 0.5
                        fig1.add_trace(go.Bar(
                            x=doc_dist_filtered.values,   
                            y=doc_dist_filtered.index,    
                            orientation='h',     
                            width=0.5, 
                            name="Doctor (Ground Truth)",
                            marker_color='#3b82f6'
                        ))
                        
                        # Liniile negre din nou subtiri (width=2) si ne-taiate (cliponaxis=False)
                        fig1.add_trace(go.Scatter(
                            x=ai_dist_filtered.values, 
                            y=ai_dist_filtered.index, 
                            name="AI Output",
                            mode='markers',
                            cliponaxis=False, 
                            marker=dict(
                                symbol='line-ns',
                                size=24,
                                line=dict(color='black', width=2) 
                            ),
                            hoverinfo='x+name'
                        ))
                        
                        annotations = []
                        for cond in displayed_conditions:
                            annotations.append(dict(
                                xref='paper', 
                                yref='y',     
                                x=0,          
                                y=cond,
                                text=f"<b>{cond}</b>", 
                                showarrow=False,
                                xanchor='left',
                                yanchor='bottom',
                                yshift=16, # Inaltime optima deasupra barelor
                                font=dict(size=13, color="#1f2937") 
                            ))
                        
                        dynamic_height = 100 + (len(displayed_conditions) * 60)
                        max_global_x = max(max(doc_dist.values, default=0), max(ai_dist.values, default=0)) + 5
                        
                        fig1.update_layout(
                            xaxis_title="Percentage of Total Cases (%)", 
                            yaxis_title="", 
                            yaxis=dict(
                                showticklabels=False, 
                                autorange="reversed"  
                            ),
                            xaxis=dict(
                                range=[-2, max_global_x], # Asigura vizibilitatea liniilor de la 0%
                                zeroline=False, 
                                showgrid=True
                            ),
                            annotations=annotations,
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                            barmode='overlay',
                            margin=dict(l=0, r=0, t=70, b=10), 
                            height=dynamic_height,
                            transition=dict(duration=400, easing="cubic-in-out")
                        )
                        
                        # Am sters orice fortare artificiala de iframe. Se va folosi strict latimea recipientului.
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        if total_filtered > 10:
                            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
                            with btn_col1:
                                if st.session_state.audit_limit < total_filtered:
                                    if st.button("🔽 Show More (+10)"):
                                        st.session_state.audit_limit += 10
                                        st.rerun()
                            with btn_col2:
                                if st.session_state.audit_limit > 10:
                                    if st.button("🔼 Collapse All"):
                                        st.session_state.audit_limit = 10
                                        st.rerun()
                    
                    st.divider()
                    
                    st.markdown("### ⚠️ AI Deviation Analysis")
                    st.caption("Detailed view of cases where MedGemma deviated from the Ground Truth.")
                    deviations = df[~df['Match']]
                    
                    if not deviations.empty:
                        st.dataframe(
                            deviations[['Timestamp', 'Image_Name', 'Doctor_Diagnosis', 'AI_Condition', 'AI_Confidence']], 
                            use_container_width=True
                        )
                    else:
                        st.success("No deviations found yet. The AI has 100% agreement with the human doctor.")
                        
                else:
                    st.warning("Audit log is empty. Generate some diagnoses first.")
            except Exception as e:
                st.error(f"Error reading audit log: {e}")
        else:
            st.info("No audit data found. Please log a diagnosis in the Data Collection tab.")