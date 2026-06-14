# STREAMLIT SKIN DISEASE DETECTION SYSTEM (FINAL STABLE VERSION)

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import json


# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

if "last_camera" not in st.session_state:
    st.session_state.last_camera = None


# -----------------------------
# LOAD CHATBOT DATA
# -----------------------------
with open("chatbot_data.json") as f:
    chatbot_data = json.load(f)


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="centered"
)

st.title("🩺 Skin Disease Detection AI Assistant")

st.warning(
    "Educational screening tool only. Not a replacement for medical diagnosis."
)


# -----------------------------
# CLASS LABELS
# -----------------------------
label_names = ['nv','mel','bkl','bcc','akiec','vasc','df']

full_names = {
    'nv':'Melanocytic Nevi (Mole)',
    'mel':'Melanoma',
    'bkl':'Benign Keratosis',
    'bcc':'Basal Cell Carcinoma',
    'akiec':'Actinic Keratosis',
    'vasc':'Vascular Lesion',
    'df':'Dermatofibroma'
}


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 7)
    )

    model.load_state_dict(
        torch.load("best_model.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# -----------------------------
# SMART IMAGE CROPPING
# -----------------------------
def preprocess_external_image(image):

    width, height = image.size

    crop_size = int(min(width, height) * 0.7)

    left = (width - crop_size)//2
    top = (height - crop_size)//2

    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))


# -----------------------------
# RUN PREDICTION
# -----------------------------
def run_prediction(image):

    st.session_state.messages = []  # reset chatbot history

    with torch.no_grad():

        tensor = transform(image).unsqueeze(0).to(device)

        probs = torch.softmax(model(tensor), dim=1)[0]

        top_idx = probs.argmax().item()

        confidence = probs[top_idx].item() * 100

    st.session_state.prediction_done = True
    st.session_state.disease_key = label_names[top_idx]
    st.session_state.confidence = confidence
    st.session_state.probs = probs


# -----------------------------
# CHATBOT ENGINE
# -----------------------------
def ask_chatbot(question, disease_key):

    question = question.lower().strip()

    info = chatbot_data[disease_key]


    # ---------- SEVERITY ----------
    if any(x in question for x in [
        "serious", "danger", "risk", "life threatening"
    ]):
        return f"This condition is {info['severity']}."


    # ---------- TREATMENT ----------
    if any(x in question for x in [
        "medicine", "medication", "drug",
        "cream", "ointment", "tablet",
        "cure", "treatment"
    ]):
        return info["treatment"]


    # ---------- TREATMENT DURATION ----------
    if (
        "how long" in question and "treat" in question
    ) or "treatment time" in question:
        return info["treatment_duration"]


    # ---------- RECOVERY TIME ----------
    if any(x in question for x in [
        "recover",
        "recovery",
        "healing",
        "heal"
    ]):
        return info["recovery_time"]


    # ---------- PERMANENCE ----------
    if any(x in question for x in [
        "permanent",
        "stay forever",
        "go away",
        "disappear",
        "will it go away"
    ]):
        return info["lesion_behavior"]


    # ---------- CAUSE ----------
    if any(x in question for x in [
        "cause",
        "why",
        "reason"
    ]):
        return info["description"]


    # ---------- PRECAUTIONS ----------
    if any(x in question for x in [
        "precaution",
        "care",
        "avoid"
    ]):
        return info["precaution"]


    # ---------- DOCTOR ----------
    if any(x in question for x in [
        "doctor",
        "consult",
        "hospital",
        "visit"
    ]):
        return info["consult"]


    # ---------- SPREAD ----------
    if any(x in question for x in [
        "spread",
        "contagious",
        "transfer"
    ]):
        return info["spread"]


    # ---------- CANCER ----------
    if any(x in question for x in [
        "cancer",
        "malignant"
    ]):
        return info["cancer_risk"]


    # ---------- PREVENTION ----------
    if any(x in question for x in [
        "prevent",
        "prevention"
    ]):
        return info["prevention"]


    # ---------- DEFAULT RESPONSE ----------
    return (
        f"{info['description']}\n\n"
        f"Treatment: {info['treatment']}\n\n"
        f"Treatment duration: {info['treatment_duration']}\n\n"
        f"Recovery time: {info['recovery_time']}\n\n"
        f"When to consult doctor: {info['consult']}"
    )
# -----------------------------
# DISPLAY RESULTS
# -----------------------------
def display_results(image):

    disease_key = st.session_state.disease_key
    confidence = st.session_state.confidence
    probs = st.session_state.probs

    info = chatbot_data[disease_key]

    st.success(f"Prediction: {full_names[disease_key]}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("📋 Dermatology Assistant Explanation")

    st.write(info["description"])
    st.write(info["precaution"])
    st.write(info["consult"])


    # Probability chart
    st.subheader("Class Probability Distribution")

    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.bar(label_names, probs.cpu().numpy())
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig, use_container_width=False)


    # Grad-CAM
    st.subheader("Grad-CAM Heatmap")

    tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    grayscale_cam = cam(input_tensor=tensor)[0]

    img_display = np.array(image.resize((224,224))) / 255.0

    heatmap = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    st.image(heatmap, width=350)


    # Chatbot UI
    st.subheader("💬 Dermatology Chat Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything about your condition")

    if user_input:

        st.session_state.messages.append(
            {"role":"user","content":user_input}
        )

        reply = ask_chatbot(user_input, disease_key)

        st.session_state.messages.append(
            {"role":"assistant","content":reply}
        )

        st.rerun()


# -----------------------------
# INPUT MODE
# -----------------------------
input_method = st.radio(
    "Select Input Method",
    ["📁 Upload Image", "📷 Capture From Camera"]
)


# -----------------------------
# UPLOAD MODE
# -----------------------------
if input_method == "📁 Upload Image":

    uploaded = st.file_uploader("Upload Skin Lesion Image")

    if uploaded:

        # reset state when new image uploaded
        if uploaded != st.session_state.last_uploaded:
            st.session_state.prediction_done = False
            st.session_state.messages = []
            st.session_state.last_uploaded = uploaded

        image = Image.open(uploaded).convert("RGB")

        st.image(image, width=350)

        if st.button("🔍 Analyze Image"):

            if min(image.size) < 120:
                st.error("Image too small. Upload closer lesion image.")

            else:
                image = preprocess_external_image(image)
                run_prediction(image)

        if st.session_state.prediction_done:
            display_results(image)


# -----------------------------
# CAMERA MODE
# -----------------------------
else:

    picture = st.camera_input("Capture Image")

    if picture:

        # reset state when new photo captured
        if picture != st.session_state.last_camera:
            st.session_state.prediction_done = False
            st.session_state.messages = []
            st.session_state.last_camera = picture

        image = Image.open(picture).convert("RGB")

        st.image(image, width=350)

        if st.button("🔍 Analyze Image"):

            if min(image.size) < 120:
                st.error("Move camera closer to lesion.")

            else:
                image = preprocess_external_image(image)
                run_prediction(image)

        if st.session_state.prediction_done:
            display_results(image)