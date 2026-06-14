# STEP 6: MODEL EVALUATION + CONFUSION MATRIX + CLASS-WISE ACCURACY + GRAD-CAM READY

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from torchvision import models
from dataload import val_loader


# -----------------------------
# DEVICE SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# LOAD MODEL ARCHITECTURE
# -----------------------------
model = models.efficientnet_b3(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features, 7)
)


# -----------------------------
# LOAD TRAINED MODEL WEIGHTS
# -----------------------------
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model = model.to(device)
model.eval()


# -----------------------------
# PREDICTION LOOP
# -----------------------------
all_preds = []
all_labels = []
all_probs = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probs = softmax(outputs)

        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())


# -----------------------------
# LABEL NAMES
# -----------------------------
label_names = [
    "nv",
    "mel",
    "bkl",
    "bcc",
    "akiec",
    "vasc",
    "df"
]


# -----------------------------
# OVERALL ACCURACY
# -----------------------------
overall_acc = accuracy_score(all_labels, all_preds)

print("\nOverall Accuracy:", round(overall_acc * 100, 2), "%")


# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        digits=4
    )
)


# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)


plt.figure(figsize=(9,7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()


# -----------------------------
# CLASS-WISE ACCURACY
# -----------------------------
cm_diag = np.diag(cm)

class_totals = cm.sum(axis=1)

class_accuracy = cm_diag / class_totals


print("\nClass-wise Accuracy:\n")

for i, name in enumerate(label_names):

    print(
        f"{name} : {round(class_accuracy[i]*100,2)}%"
    )


print("\nSTEP 6 COMPLETED SUCCESSFULLY!")