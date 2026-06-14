# STREAMLIT SKIN DISEASE DETECTION + SMART CHATBOT

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# LABEL DEFINITIONS
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
# SMART DERMATOLOGY KNOWLEDGE BASE
# -----------------------------
DERM_KNOWLEDGE = {

    "nv": {
        "seriousness": "Usually benign and harmless. Monitor changes using ABCDE rule.",
        "cause": "Clusters of melanocyte cells forming a mole.",
        "treatment": "No treatment needed unless appearance changes.",
        "precautions": "Monitor size, color, shape changes regularly.",
        "doctor": "Consult dermatologist if mole changes or bleeds."
    },

    "mel": {
        "seriousness": "Serious skin cancer. Needs urgent dermatologist consultation.",
        "cause": "UV radiation damages melanocyte DNA.",
        "treatment": "Surgery, immunotherapy, chemotherapy depending on stage.",
        "precautions": "Avoid sun exposure. Use SPF 50+ sunscreen.",
        "doctor": "Immediate consultation required."
    },

    "bkl": {
        "seriousness": "Non-cancerous and generally harmless.",
        "cause": "Ageing and long-term sun exposure.",
        "treatment": "Usually no treatment needed.",
        "precautions": "Avoid scratching lesion.",
        "doctor": "Consult doctor if lesion changes rapidly."
    },

    "bcc": {
        "seriousness": "Slow growing skin cancer but treatable.",
        "cause": "Long-term sun exposure damage.",
        "treatment": "Minor surgery removes lesion successfully.",
        "precautions": "Avoid sun exposure and monitor lesion.",
        "doctor": "Dermatologist consultation recommended."
    },

    "akiec": {
        "seriousness": "Pre-cancerous lesion.",
        "cause": "Chronic UV exposure.",
        "treatment": "Cryotherapy or topical creams.",
        "precautions": "Daily sunscreen essential.",
        "doctor": "Consult dermatologist early."
    },

    "vasc": {
        "seriousness": "Usually benign vascular growth.",
        "cause": "Blood vessel clustering.",
        "treatment": "Treatment optional unless cosmetic concern.",
        "precautions": "Monitor size changes.",
        "doctor": "Consult doctor if lesion enlarges."
    },

    "df": {
        "seriousness": "Benign harmless skin nodule.",
        "cause": "Minor skin injury or insect bite.",
        "treatment": "No treatment needed normally.",
        "precautions": "Avoid irritation.",
        "doctor": "Consult if painful or growing."
    }
}


# -----------------------------
# CHATBOT FUNCTION
# -----------------------------
def dermatology_chatbot(question, disease):

    question = question.lower()

    info = DERM_KNOWLEDGE[disease]

    if "serious" in question:
        return info["seriousness"]

    if "cause" in question:
        return info["cause"]

    if "cure" in question or "treatment" in question:
        return info["treatment"]

    if "precaution" in question or "care" in question:
        return info["precautions"]

    if "doctor" in question or "hospital" in question:
        return info["doctor"]

    return (
        f"{full_names[disease]} detected.\n\n"
        "You can ask:\n"
        "• Is it serious?\n"
        "• What is the cause?\n"
        "• What is the treatment?\n"
        "• Precautions?\n"
        "• Should I see a doctor?"
    )


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
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
# PREDICTION FUNCTION
# -----------------------------
def predict(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)[0]

        idx = probs.argmax().item()

        confidence = probs[idx].item()

    return label_names[idx], confidence, tensor


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🩺 Skin Disease Detection AI Assistant")


uploaded = st.file_uploader(
    "Upload skin image",
    type=["jpg","jpeg","png"]
)


if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, width=300)

    label, confidence, tensor = predict(image)

    st.success(f"Prediction: {full_names[label]}")

    st.info(f"Confidence: {confidence*100:.2f}%")

    st.warning(
        "This system is AI-assisted screening only.\n"
        "Consult dermatologist for medical confirmation."
    )


    # -----------------------------
    # GRADCAM HEATMAP
    # -----------------------------
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=tensor)[0]

    img = np.array(image.resize((224,224))) / 255.0

    visualization = show_cam_on_image(
        img,
        grayscale_cam,
        use_rgb=True
    )

    st.image(
        visualization,
        caption="Grad-CAM Explanation",
        width=300
    )


    # -----------------------------
    # CHATBOT SECTION
    # -----------------------------
    st.header("💬 Dermatology Chat Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:

        st.chat_message(msg["role"]).write(msg["content"])


    if prompt := st.chat_input("Ask about your condition"):

        st.session_state.messages.append(
            {"role":"user","content":prompt}
        )

        reply = dermatology_chatbot(prompt, label)

        st.session_state.messages.append(
            {"role":"assistant","content":reply}
        )

        st.rerun()