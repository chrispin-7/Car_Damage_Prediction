import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 🎨 Page config
st.set_page_config(page_title="Car Damage Detector", layout="wide")

# 🏷️ Class names
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

# 📦 Load models
@st.cache_resource
def load_classifier():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load("car_damage_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")  # Replace with your custom model if needed

classifier = load_classifier()
detector = load_detector()

# 🔄 Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🖼️ UI
st.title("🚗 Car Damage Detection & Classification")
st.write("Upload an image of a car to detect and classify damage using deep learning.")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running object detection..."):
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = detector(img_bgr)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = results.names[int(box.cls[0])]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label}: {conf*100:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(img_rgb, caption="Detected Damage", use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Classification Result")

    with st.spinner("🧠 Classifying damage type..."):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = classifier(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            predicted_class = class_names[preds.item()]
            confidence_score = confidence.item() * 100

        st.success(f"**Predicted Class:** `{predicted_class}`")
        st.progress(min(int(confidence_score), 100))
        st.caption(f"Model Confidence: **{confidence_score:.2f}%**")

    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 0.9em;'>Built using Streamlit, PyTorch & YOLOv8<br>Developed by <strong>Chrispin Joseph</strong></p>", unsafe_allow_html=True)
