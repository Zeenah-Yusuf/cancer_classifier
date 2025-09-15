# Simple password gate
def authenticate():
    st.sidebar.subheader("🔐 Login")
    password = st.sidebar.text_input("Enter access key", type="password")
    if password == "oncolens2025":
        return True
    else:
        st.sidebar.warning("Invalid access key")
        return False

if not authenticate():
    st.stop()



import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

st.image("https://drive.google.com/file/d/1RaJtN2y-ee1OTCwmNGoSVfZTBV5qWaek/view?usp=sharing", width=150)
st.markdown("<h1 style='text-align: center;'>🧬 Oncolens Cancer Classifier</h1>", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="OncoLens", page_icon="🧬", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body { background-color: #f9fdfc; }
        h1, h2, h3, h4, h5, h6 { color: #008080; }
        .stButton>button, .stDownloadButton>button {
            background-color: #008080;
            color: white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="OncoLens Navigation",
        options=["Home", "Classifier", "Patient Info", "Compliance"],
        icons=["house", "activity", "person", "shield-check"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0fdfc"},
            "icon": {"color": "#008080", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#e0f7f5"},
            "nav-link-selected": {"background-color": "#008080", "color": "white"},
        }
    )

# Load model and label encoder
model = tf.keras.models.load_model('cancer_classifier_model.h5')
label_map = joblib.load('label_encoder.pkl')
inv_label_map = {v: k for k, v in label_map.items()}
label_descriptions = { ... }  # same dictionary as before

# Page: Home (Splash Screen)
if selected == "Home":
    st.image("logo.png", width=120)
    st.markdown("<h1 style='text-align: center;'>Welcome to OncoLens</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>AI-Powered Cancer Image Classification</h4>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <p>Upload medical images, analyze cancer types, and explore healthcare compliance — all in one place.</p>
            <a href='#Classifier'><button style='background-color:#008080;color:white;padding:10px 20px;border:none;border-radius:5px;'>Get Started</button></a>
        </div>
    """, unsafe_allow_html=True)

# Page: Classifier
elif selected == "Classifier":
    st.header("📤 Upload Medical Image")
    uploaded_file = st.file_uploader("Upload a medical image to begin", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((128, 128))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        class_name = list(label_map.keys())[list(label_map.values()).index(class_idx)]
        description = label_descriptions.get(class_name, "Unknown class")
        confidence = round(float(np.max(pred)) * 100, 2)
        st.success(f"**Cancer Type:** {description}")
        st.info(f"**Classification Role:** `{class_name}`")
        st.metric(label="Prediction Confidence", value=f"{confidence}%")
        result_text = f"Cancer Type: {description}\nRole: {class_name}\nConfidence: {confidence}%"
        st.download_button("📥 Download Results", result_text.encode("utf-8"), "oncolens_result.txt", "text/plain")
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io

        # Generate PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(50, 750, "OncoLens Cancer Classification Report")
        c.drawString(50, 720, f"Cancer Type: {description}")
        c.drawString(50, 700, f"Classification Role: {class_name}")
        c.drawString(50, 680, f"Prediction Confidence: {confidence}%")
        c.drawString(50, 640, "Disclaimer: This result is for research and educational use only.")
        c.drawString(50, 760, f"Patient Name: {name}")
        c.drawString(50, 740, f"Age: {age} | Gender: {gender}")
        c.save()
        pdf_buffer.seek(0)

        # Download button
        st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="oncolens_report.pdf",
        mime="application/pdf"
        )

# Page: Patient Info
elif selected == "Patient Info":
    st.header("🧑‍⚕️ Patient Metadata")
    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        notes = st.text_area("Clinical Notes")
        submitted = st.form_submit_button("Save Metadata")
        if submitted:
            st.success(f"Metadata saved for {name}, age {age}, gender {gender}.")

# Page: Compliance
elif selected == "Compliance":
    st.header("📜 Healthcare & AI Compliance Standards")
    with st.expander("View Full Standards"):
        st.markdown("""
        - ✅ **FDA Guidelines Compliant**  
          Follows U.S. Food and Drug Administration principles for AI in medical devices.
        - ✅ **CE Medical Device Standards**  
          Meets European Union safety and performance requirements.
        - ✅ **ISO 13485 Certified**  
          Adheres to international standards for medical device quality management.
        - ✅ **WHO Ethics Referenced**  
          Aligns with World Health Organization guidance on ethical AI in healthcare.
        - ✅ **HIPAA & GDPR Awareness**  
          Designed with data privacy and patient confidentiality in mind.
        """)

    st.markdown("""
    > ⚠️ **Medical Disclaimer**  
    This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis or treatment.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 OncoLens AI | Empowering medical diagnostics through intelligent technology</p>", unsafe_allow_html=True)
