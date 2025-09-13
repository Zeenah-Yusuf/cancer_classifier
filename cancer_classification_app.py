
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# Load model and label encoder
model = tf.keras.models.load_model('cancer_classifier_model.h5')
label_map = joblib.load('label_encoder.pkl')
inv_label_map = {v: k for k, v in label_map.items()}

# Streamlit UI
st.title("🧬 Cancer Image Classifier")
st.write("Upload an image to predict the cancer type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Image', width='stretch')

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)


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
    "oral_normal": "Healthy oral tissue"
}





    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = list(label_map.keys())[list(label_map.values()).index(class_idx)]

# Get human-readable description
description = label_descriptions.get(class_name, "Unknown class")

st.write(f"**Raw Label:** {class_name}")
st.write(f"**Prediction:** {description}")
confidence = round(float(np.max(pred)) * 100, 2)
st.write(f"**Confidence:** {confidence}%")


# import streamlit as st
# import json

# if st.button("📊 Load Evaluation Results"):
#     import json
#     try:
#         with open("evaluation_results.json", "r") as f:
#             results = json.load(f)
#             test_acc = results["Test Accuracy"]
#             test_loss = results["Test Loss"]

#             st.subheader("🧪 Model Evaluation")
#             st.metric("Test Accuracy", f"{test_acc:.2%}")
#             st.metric("Test Loss", f"{test_loss:.4f}")
    # except FileNotFoundError:
    #     st.warning("Evaluation results not found. Please run the evaluation script first.")
