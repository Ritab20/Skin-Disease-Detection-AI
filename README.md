# 🩺 Skin Disease Detection AI Assistant

An AI-powered skin disease classification and screening system built using **EfficientNet-B3**, **Grad-CAM**, and **Streamlit**. The system classifies skin lesions into seven dermatological categories, provides visual explanations using Grad-CAM, and offers disease-specific guidance through an integrated chatbot.

---

## 📌 Features

### 🔍 Skin Disease Classification
- Classifies skin lesion images into 7 categories.
- Built using EfficientNet-B3 deep learning architecture.
- Trained on the HAM10000 dataset.

### 🔥 Explainable AI (Grad-CAM)
- Generates heatmaps showing which lesion regions influenced the prediction.
- Improves transparency and trustworthiness of the model.

### 📊 Prediction Confidence Visualization
- Displays probability distribution across all classes.
- Helps users understand prediction confidence.

### 🤖 Dermatology Chat Assistant
- Provides disease-specific information.
- Answers questions related to:
  - Severity
  - Treatment
  - Recovery Time
  - Prevention
  - Cancer Risk
  - Medical Consultation Advice

### 📷 Camera-Based Detection
- Supports real-time skin lesion analysis using a webcam.
- Optional Grad-CAM visualization during live predictions.

---

## 🧠 Disease Classes

The model predicts the following seven skin disease categories:

| Code | Disease |
|--------|----------|
| nv | Melanocytic Nevi (Mole) |
| mel | Melanoma |
| bkl | Benign Keratosis-like Lesions |
| bcc | Basal Cell Carcinoma |
| akiec | Actinic Keratoses |
| vasc | Vascular Lesions |
| df | Dermatofibroma |

---

## 🗂 Dataset

Dataset Used:

**HAM10000 (Human Against Machine with 10000 Training Images)**

The HAM10000 dataset contains dermatoscopic images of common pigmented skin lesions and is widely used for skin cancer classification research.

Dataset Source:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## 🏗 Project Architecture

```text
Image Input
      ↓
Preprocessing & Augmentation
      ↓
EfficientNet-B3
      ↓
Softmax Classification
      ↓
Prediction & Confidence Score
      ↓
Grad-CAM Visualization
      ↓
Chatbot-Based Disease Guidance
```

---

## 📂 Project Structure

```text
Skin-Disease-Detection-AI/
│
├── app.py
├── camera_app.py
├── train.py
├── test.py
├── dataload.py
├── grad.py
├── load.py
├── bestmodel_path.py
│
├── chatbot_data.json
├── balanced_data.csv
├── HAM10000_metadata.csv
│
├── best_model.pth
├── training_curve.png
│
├── requirements.txt
├── README.md
└── README_CAMERA.md
```

---

## ⚙️ Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Grad-CAM

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Skin-Disease-Detection-AI.git

cd Skin-Disease-Detection-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Streamlit Web App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📷 Running Webcam Detection

```bash
python camera_app.py
```

Controls:

| Key | Action |
|------|----------|
| C | Toggle Grad-CAM |
| S | Capture Prediction |
| Q | Quit Application |

---

## 📈 Model Performance

### Training Accuracy

~99%

### Validation Accuracy

~96%

### Key Observations

- High classification performance.
- Minimal overfitting due to:
  - Data augmentation
  - Dropout regularization
  - Balanced dataset generation

---

## 🔥 Explainability Using Grad-CAM

Grad-CAM is integrated to visualize the lesion regions responsible for model predictions.

Benefits:

- Improves interpretability.
- Reduces black-box behavior.
- Helps verify medically relevant feature extraction.

---

## ⚠ Disclaimer

This project is intended for educational and research purposes only.

It is not a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Always consult a qualified dermatologist for medical advice.

---

## 👨‍💻 Author

Ritabrata Mandal

B.Tech Project – Skin Disease Detection AI Assistant

Built using EfficientNet-B3, Grad-CAM, and Streamlit.