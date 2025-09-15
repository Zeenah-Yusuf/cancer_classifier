import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Page config
st.set_page_config(page_title="OncoLens", page_icon="🧬", layout="wide")

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

# st.image("https://imgur.com/a/tZSjx0K", width=150)
# st.markdown("<h1 style='text-align: center;'>🧬 Oncolens Cancer Classifier</h1>", unsafe_allow_html=True)


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
st.markdown("""
    <style>
        .floating-button {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background-color: #008080;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 30px;
            font-size: 14px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            z-index: 100;
        }
        .floating-button:hover {
            background-color: #006666;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Dropdown styling */
        .stSelectbox label {
            font-weight: bold;
            color: #008080;
        }
        .stSelectbox div[data-baseweb="select"] {
            border: 2px solid #008080;
            border-radius: 6px;
        }

        /* Go button styling */
        .go-button > button {
            background-color: #008080;
            color: white;
            padding: 8px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            margin-top: 10px;
        }
        .go-button > button:hover {
            background-color: #006666;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <img src="https://i.imgur.com/UJTEe8w.png" width="100" style="border: 2px solid #008080; border-radius: 8px;">
    </div>
""", unsafe_allow_html=True)
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
    # Override navigation if triggered from Home page
if "selected_page" in st.session_state:
    selected = st.session_state["selected_page"]
    del st.session_state["selected_page"]

# Load model and label encoder
model = tf.keras.models.load_model('cancer_classifier_model.h5')
label_map = joblib.load('label_encoder.pkl')
inv_label_map = {v: k for k, v in label_map.items()}
label_descriptions = {
     "all_benign": "Benign blood cells (non-cancerous)",
    "all_early": "Early-stage acute lymphoblastic leukemia",
    "all_pre": "Pre-B cell subtype of leukemia",
    "all_pro": "Pro-B cell subtype of leukemia",
    "brain_glioma": "Glioma (tumor from glial cells)",
    "brain_menin": "Meningioma (tumor from meninges)",
    "brain_tumor": "General brain tumor",
    "breast_benign": "Benign breast tissue",
    "breast_malignant": "Malignant breast tissue (cancerous)",
    "cervix_dyk": "Dyskeratotic cells (abnormal keratinization)",
    "cervix_koc": "Koilocytotic cells (HPV-related changes)",
    "cervix_mep": "Metaplastic epithelial cells",
    "cervix_pab": "Parabasal cells (immature squamous cells)",
    "cervix_sfi": "Superficial squamous cells (normal)",
    "colon_aca": "Colon adenocarcinoma (colon cancer)",
    "colon_bnt": "Benign colon tissue",
    "kidney_normal": "Healthy kidney tissue",
    "kidney_tumor": "Kidney tumor (cancerous)",
    "lung_aca": "Lung adenocarcinoma",
    "lung_bnt": "Benign lung tissue",
    "lung_scc": "Lung squamous cell carcinoma",
    "lymph_cll": "Chronic lymphocytic leukemia",
    "lymph_fl": "Follicular lymphoma",
    "lymph_mcl": "Mantle cell lymphoma",
    "oral_normal": "Healthy oral tissue",
    "oral_scc": "Oral Squamous Cell Carcinoma"
} 

# Page: Home
if selected == "Home":
    st.image("https://i.imgur.com/UJTEe8w.png", width=150)
    st.markdown("<h1 style='text-align: center;'>Welcome to OncoLens</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>AI-Powered Cancer Image Classification</h4>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <p>Upload medical images, analyze cancer types, and explore healthcare compliance — all in one place.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Ready to explore?")
    destination = st.selectbox("Choose where to go:", ["Classifier", "Patient Info", "Compliance"])
    if st.button("Go", key="go_button"):
        st.session_state["selected_page"] = destination

# Page: Classifier
elif selected == "Classifier":
    st.header("📤 Upload Medical Image")
    uploaded_file = st.file_uploader("Upload a medical image to begin", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).resize((128, 128))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict", key="predict_button"):
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
            format_choice = st.radio("Choose download format:", ["Text (.txt)", "PDF (.pdf)"])

            if format_choice == "Text (.txt)":
                st.download_button(
                    label="📥 Download Results",
                    data=result_text.encode("utf-8"),
                    file_name="oncolens_result.txt",
                    mime="text/plain"
                )

            elif format_choice == "PDF (.pdf)":
                patient_name = st.session_state.get("name", "N/A")
                patient_age = st.session_state.get("age", "N/A")
                patient_gender = st.session_state.get("gender", "N/A")

                pdf_buffer = io.BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=letter)
                c.setFont("Helvetica", 12)
                c.drawString(50, 750, "OncoLens Cancer Classification Report")
                c.drawString(50, 720, f"Patient Name: {patient_name}")
                c.drawString(50, 700, f"Age: {patient_age} | Gender: {patient_gender}")
                c.drawString(50, 680, f"Cancer Type: {description}")
                c.drawString(50, 660, f"Classification Role: {class_name}")
                c.drawString(50, 640, f"Prediction Confidence: {confidence}%")
                c.drawString(50, 600, "Disclaimer: This result is for research and educational use only.")
                c.save()
                pdf_buffer.seek(0)

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name="oncolens_report.pdf",
                    mime="application/pdf"
                )

    # Floating button (styled and placed outside prediction logic)
    st.markdown("""
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 100;">
            <style>
                .stButton>button {
                    background-color: #008080;
                    color: white;
                    border-radius: 30px;
                    padding: 10px 20px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                }
                .stButton>button:hover {
                    background-color: #006666;
                }
            </style>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔙 Go Back to Home", key="floating_classifier"):
        st.session_state["selected_page"] = "Home"
        
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
            st.session_state["name"] = name
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.success(f"Metadata saved for {name}, age {age}, gender {gender}.")
    st.markdown("""
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 100;">
            <style>
                .stButton>button {
                    background-color: #008080;
                    color: white;
                    border-radius: 30px;
                    padding: 10px 20px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                }
                .stButton>button:hover {
                    background-color: #006666;
                }
            </style>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔙 Go Back to Home", key="floating_patient"):
        st.session_state["selected_page"] = "Home"
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
    st.markdown("""
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 100;">
            <style>
                .stButton>button {
                    background-color: #008080;
                    color: white;
                    border-radius: 30px;
                    padding: 10px 20px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                }
                .stButton>button:hover {
                    background-color: #006666;
                }
            </style>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔙 Go Back to Home", key="floating_compliance"):
        st.session_state["selected_page"] = "Home"
# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 OncoLens AI | Empowering medical diagnostics through intelligent technology</p>", unsafe_allow_html=True)
