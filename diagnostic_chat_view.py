import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
from utils import helpers

API_URL = "http://localhost:8000/chat/diagnosis"

def image_to_base64(image: Image.Image) -> str:
    """Converteste o imagine PIL in string Base64 pentru API."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def render_diagnostic_chat_page():
    st.header("Medgemma Diagnostic Chat")
    st.info("Upload a retinal image and start an interactive clinical consultation with MedGemma.")

    # 1. Initializarea memoriei persistente
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_image_base64" not in st.session_state:
        st.session_state.current_image_base64 = None
    if "current_image_display" not in st.session_state:
        st.session_state.current_image_display = None

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Contextual Image")
        uploaded_file = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"], key="chat_uploader")
        
        # 2. Actualizam cache-ul DOAR daca se incarca un fisier nou
        if uploaded_file is not None:
            image = helpers.load_image(uploaded_file)
            st.session_state.current_image_display = image
            st.session_state.current_image_base64 = image_to_base64(image)
            
        # 3. Randam imaginea intotdeauna din cache, independent de uploader
        if st.session_state.current_image_display is not None:
            st.image(st.session_state.current_image_display, caption="Active Image Context", width='stretch')
            
        # Am actualizat butonul pentru a curata atat chat-ul cat si imaginea salvata
        if st.button("Clear Conversation & Image", width='stretch'):
            st.session_state.chat_messages = []
            st.session_state.current_image_display = None
            st.session_state.current_image_base64 = None
            st.rerun()

    with col2:
        st.markdown("### Clinical Consultation")
        
        chat_container = st.container(height=600, border=True)
        
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                
        user_query = st.chat_input("Ask a clinical question about the image...")
        
        if user_query:
            if not st.session_state.current_image_base64:
                st.warning("Please upload an image first to establish the clinical context.")
            else:
                st.session_state.chat_messages.append({"role": "user", "content": user_query})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    
                payload = {
                    "image_base64": st.session_state.current_image_base64,
                    "message": user_query,
                    "history": st.session_state.chat_messages[:-1], 
                    "clinical_context": "You are a highly skilled ophthalmologist AI. Analyze the image and respond directly, professionally, and accurately to the user's queries."
                }
                
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("MedGemma is analyzing..."):
                            try:
                                response = requests.post(API_URL, json=payload)
                                response.raise_for_status() 
                                
                                ai_reply = response.json().get("reply", "No response received.")
                                st.markdown(ai_reply)
                                
                                st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
                                
                            except requests.exceptions.ConnectionError:
                                st.error("Cannot connect to backend. Please ensure `python api.py` is running in a separate terminal.")
                            except Exception as e:
                                st.error(f"An error occurred: {e}")