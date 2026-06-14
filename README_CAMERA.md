# 🩺 Camera Input Options - User Guide

This project now supports **two methods** for camera input:

---

## Option 1️⃣: **Streamlit Web App** (Recommended for Web Interface)

### Features:
- ✅ Cross-platform (Windows, Mac, Linux)
- ✅ File upload AND camera capture in one app
- ✅ Automatic fallback if dependencies missing
- ✅ Beautiful web UI with instant feedback

### How to Run:
```bash
streamlit run app.py
```

### Usage:
1. **Upload Image**: Click "📁 Upload Image" tab
2. **Camera Capture**: Click "📷 Capture from Camera" tab
   - If `streamlit-webrtc` is installed: Full live streaming with capture button
   - If not installed: Simple camera snapshot button

### Optional: Install WebRTC for better camera streaming
```bash
pip install streamlit-webrtc
```

---

## Option 2️⃣: **Desktop OpenCV App** (Real-time Detection)

### Features:
- ✅ **Real-time** continuous analysis (every 30 frames)
- ✅ **Grad-CAM heatmaps** overlay
- ✅ **Probability bars** for all disease classes
- ✅ **No web browser needed**
- ✅ Fast processing with threading

### How to Run:
```bash
python camera_app.py
```

### Controls:
| Key | Action |
|-----|--------|
| `s` | Capture and log current prediction |
| `c` | Toggle Grad-CAM heatmap display |
| `q` | Quit application |

### Features Explained:
- **Live Streaming**: Continuous video feed with 30 FPS
- **Auto-Processing**: Analyzes frame every 30 frames (≈1 per second)
- **Probability Bars**: Color-coded confidence for each disease
- **Heatmap Display**: Shows which parts of image influenced prediction
- **GPU Support**: Automatically uses CUDA if available

---

## 🔧 Setup Instructions

### 1️⃣ Ensure All Dependencies
```bash
pip install streamlit opencv-python torch torchvision pytorch-grad-cam pillow numpy
```

### 2️⃣ (Optional) WebRTC for Enhanced Camera Streaming
```bash
pip install streamlit-webrtc
pip install av
```

### 3️⃣ Verify Model File
Make sure `best_model.pth` exists in the project directory

---

## 📋 Comparison Table

| Feature | Streamlit | OpenCV Desktop |
|---------|-----------|-----------------|
| Real-time Analysis | ❌ Manual | ✅ Auto (30 FPS) |
| Processes Multiple Frames | ❌ One at a time | ✅ Continuous |
| Grad-CAM Heatmap | ✅ Yes | ✅ Yes |
| Web Interface | ✅ Yes | ❌ No |
| GUI Toggle Options | ✅ Radio buttons | ✅ Keyboard (s/c/q) |
| Processing Speed | Slower | ⚡ Faster |
| Multi-user Access | ✅ Yes | ❌ Local only |
| GPU Enabled | ✅ Yes | ✅ Yes |

---

## 🚀 Quick Start

### For Web Interface (Easiest):
```bash
streamlit run app.py
```
→ Opens browser at `http://localhost:8501`

### For Desktop (Best Real-time Performance):
```bash
python camera_app.py
```
→ Opens window with live camera feed

---

## 🐛 Troubleshooting

### Camera Not Detected
- **OpenCV**: Check device in Device Manager (Windows) or `System Information` (Mac)
- **Streamlit**: Browser must have camera permission - check browser settings

### Slow Performance
- **Solution 1**: Reduce camera resolution in `camera_app.py` (change `FRAME_SIZE`)
- **Solution 2**: Increase processing interval (change frame skip from 30 to 60)
- **Solution 3**: Use GPU-enabled machine or reduce image input size

### Missing Dependencies
```bash
# Install all at once
pip install streamlit opencv-python torch torchvision pytorch-grad-cam pillow numpy streamlit-webrtc av
```

---

## 📝 Notes

- **IMPORTANT**: Both apps are screening tools only - always consult a dermatologist
- Model was trained on HAM10000 dataset
- For best results: Ensure good lighting and clear image of lesion
- Confidence < 60% may indicate ambiguous lesions - get professional opinion

---

## File Structure
```
skin-disease-project/
├── app.py                    # Streamlit web app (upload + camera)
├── camera_app.py            # Desktop OpenCV app (real-time)
├── best_model.pth           # Trained model weights
├── train.py                 # Model training script
├── test.py                  # Model testing script
└── README_CAMERA.md         # This file
```

---

**Choose your method based on your needs:**
- 🌐 **Web + Multiple Users** → Use Streamlit (`app.py`)
- 🖥️ **Fast Real-time Detection** → Use OpenCV (`camera_app.py`)
