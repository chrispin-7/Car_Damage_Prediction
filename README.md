# 🚗 Car Damage Detection & Classification

This project uses deep learning to detect and classify car damage from images. It combines a **ResNet18 classifier** to identify the type of damage and a **YOLOv8 object detector** to localize the damaged areas with bounding boxes — all wrapped in an interactive **Streamlit web app**.

---

## 📸 Demo

![App Screenshot](https://github.com/chrispin-7/Car_Damage_Prediction/blob/main/model.png))  

---

## 🧠 Features

- 🔍 **Object Detection** with YOLOv8 to highlight damaged areas
- 🧪 **Image Classification** with ResNet18 to label damage type:
  - `F_Breakage`, `F_Crushed`, `F_Normal`
  - `R_Breakage`, `R_Crushed`, `R_Normal`
- 🖼️ Streamlit UI for easy interaction
- 📊 Confidence scores and visual feedback

---

## 🗂️ Project Structure

This app detects and classifies car damage using deep learning. It combines YOLOv8 for object detection and ResNet18 for damage classification, offering instant predictions through an interactive Streamlit interface. Ideal for insurance, inspections, or fleet assessment.
