# REAL-TIME CAMERA APP (UPGRADED VERSION)

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from bestmodel_path import load_best_model, DEVICE


# ============================
# SETTINGS
# ============================
FRAME_SIZE = (300, 300)
CONFIDENCE_THRESHOLD = 40


# ============================
# LABELS
# ============================
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


# ============================
# LOAD MODEL
# ============================
def load_model():

    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features,7)
    )

    model = load_best_model(model)

    print("Using device:", DEVICE)

    return model


model = load_model()


# ============================
# IMAGE TRANSFORM
# ============================
transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# ============================
# PREDICTION FUNCTION
# ============================
def predict(image):

    with torch.no_grad():

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):

            outputs = model(tensor)

        probs = torch.softmax(outputs, dim=1)[0]

        top_idx = probs.argmax().item()

        confidence = probs[top_idx].item()*100

    return label_names[top_idx], confidence, probs


# ============================
# GRADCAM
# ============================
cam = GradCAM(
    model=model,
    target_layers=[model.features[-1]]
)


def generate_heatmap(image):

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(input_tensor=tensor)[0]

    img_display = np.array(image.resize(FRAME_SIZE))/255.0

    heatmap = show_cam_on_image(
        img_display,
        grayscale_cam,
        use_rgb=True
    )

    return heatmap


# ============================
# CAMERA LOOP
# ============================
cap = cv2.VideoCapture(0)

print("\nCamera ready")
print("Press S = capture")
print("Press C = toggle heatmap")
print("Press Q = quit\n")


show_heatmap = False


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(frame_rgb)


    disease, confidence, probs = predict(image)


    if confidence > CONFIDENCE_THRESHOLD:

        text = f"{full_names[disease]} ({confidence:.1f}%)"

    else:

        text = "Low confidence — move camera closer"


    cv2.putText(
        frame,
        text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )


    if show_heatmap:

        heatmap = generate_heatmap(image)

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        heatmap = cv2.resize(heatmap,(300,300))

        frame[0:300,0:300] = heatmap


    cv2.imshow("Skin Disease Detection Camera", frame)


    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break

    elif key == ord('c'):
        show_heatmap = not show_heatmap

    elif key == ord('s'):

        print("\nCaptured prediction:")
        print(full_names[disease])
        print("Confidence:",confidence)


cap.release()

cv2.destroyAllWindows()