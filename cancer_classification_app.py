import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# Page configuration
st.set_page_config(
    page_title="Oncolens - Cancer Classification",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load model and label encoder
model = tf.keras.models.load_model('cancer_classifier_model.h5')
label_map = joblib.load('label_encoder.pkl')
inv_label_map = {v: k for k, v in label_map.items()}

# Header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧬 Oncolens</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>AI-Powered Cancer Image Classification</h3>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.subheader("📤 Upload an Image")
st.write("Upload a medical image (JPG, JPEG, PNG) to predict the cancer type using our trained deep learning model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Label descriptions
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

# Prediction section
if uploaded_file:
    st.markdown("---")
    st.subheader("🔍 Image Preview")
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = list(label_map.keys())[list(label_map.values()).index(class_idx)]
    description = label_descriptions.get(class_name, "Unknown class")
    confidence = round(float(np.max(pred)) * 100, 2)


        # Prepare result summary
    result_text = f"""
    🧬 Oncolens - Cancer Classification Result

    Raw Label: {class_name}
    Prediction: {description}
    Confidence: {confidence}%

    This result was generated using a deep learning model trained on histopathological and radiological cancer images.
    """

    # Encode as bytes for download
    result_bytes = result_text.encode("utf-8")

    # Download button
    st.download_button(
        label="📥 Download Result",
        data=result_bytes,
        file_name="cancer_prediction_result.txt",
        mime="text/plain"
    )


    # Results
    st.markdown("---")
    st.subheader("📈 Prediction Results")
    st.success(f"**Cancer Type:** {description}")
    st.info(f"**Raw Label:** `{class_name}`")
    st.metric(label="Prediction Confidence", value=f"{confidence}%")

else:
    st.warning("Please upload an image to begin classification.")

# Compliance section
st.markdown("---")
st.subheader("📜 Model Authenticity & Healthcare Standards")
st.markdown("""
This AI model is trained on a curated dataset of histopathological and radiological images. While it provides high-confidence predictions, it is **not a substitute for clinical diagnosis**.

**Regulatory Notes:**
- This tool is intended for **educational and research purposes** only.
- It does **not meet FDA or NAFDAC certification** for clinical deployment.
- All predictions should be reviewed by a licensed medical professional.

**Ethical Standards:**
- Model development followed principles of **data privacy**, **bias mitigation**, and **transparency**.
- Dataset sources were anonymized and ethically approved for research use.

For more information on AI in healthcare, visit [WHO's guidance on AI ethics](https://www.who.int/publications/i/item/9789240029200).
""")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 Oncolens AI | Built for innovation in medical diagnostics</p>", unsafe_allow_html=True)
