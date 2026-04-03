# 🚨 AI-Powered Public Safety Surveillance System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>Multi-Model AI Surveillance Framework for Detecting Violence, Weapons, Police Presence, and Accident Scenarios in Real Time</b>
</p>

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Project Architecture](#-project-architecture)
* [Features](#-features)
* [Model Pipelines](#-model-pipelines)
* [Folder Structure](#-folder-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [Results](#-results)
* [Tech Stack](#-tech-stack)
* [Future Improvements](#-future-improvements)
* [Contributors](#-contributors)

---

## 🚀 Overview

This project is an **AI-powered public safety monitoring system** that combines multiple deep learning models into a **unified real-time surveillance framework**.

The system detects:

* 🔫 **Weapons**
* ⚠️ **Violence**
* 👮 **Police presence**
* 🚑 **Accident scenes**
* 🟢 **Normal scenes**

The goal is to assist in **smart surveillance, crowd monitoring, emergency detection, and law-enforcement support systems**.

---

## 🧠 Project Architecture

```text
Input Image / Video
        │
        ▼
 ┌──────────────────────┐
 │ Unified Inference API │
 │      (FastAPI)        │
 └──────────────────────┘
        │
        ▼
 ┌──────────┬──────────┬──────────┬──────────┐
 │ Weapon   │ Violence │ Police   │ Accident │
 │ Detector │ Model    │ Detector │ Detector │
 └──────────┴──────────┴──────────┴──────────┘
        │
        ▼
 Final Scene Classification + Annotated Output
        │
        ▼
 Streamlit Dashboard / Saved Output
```

---

## ✨ Features

<details>
<summary><b>Click to Expand Features</b></summary>

### 🔍 Detection Modules

* YOLO-based weapon detection
* CNN + LSTM violence recognition
* Police presence classification
* Accident scene detection
* Unified scene labeling

### 📹 Input Support

* Static images
* Video files
* Webcam feed
* API upload requests

### 🌐 Deployment

* FastAPI backend
* Streamlit frontend dashboard
* Real-time inference support

### 📊 Outputs

* Bounding box annotations
* Confidence scores
* Final scene labels
* Saved output images/videos

</details>

---

## 🤖 Model Pipelines

### 🔫 Weapon Detection

* **Model:** YOLOv8
* **Framework:** Ultralytics / PyTorch
* **Weights:** `best.pt`

Detects:

* guns
* knives
* dangerous objects

---

### ⚠️ Violence Detection

* **Model:** MobileNet + LSTM
* **Framework:** TensorFlow / Keras
* **Weights:** `final_model.h5`

Classifies:

* violence
* non-violence

---

### 👮 Police Detection

* **Model:** MobileNetV2-based classifier
* **Framework:** TensorFlow
* **Training:** Custom dataset + COCO format pipeline

Detects:

* police personnel
* uniforms
* security presence

---

### 🚑 Accident Detection

* **Model:** Custom ResNet50-based PyTorch model
* **Weights:** `accidentDetect.pth`

Detects:

* accident scenes
* collision cases
* emergency situations

---

## 📁 Folder Structure

```text
project-root/
│
├── WeaponDetection/
│   ├── weapon.py
│   └── best.pt
│
├── VoilenceDetect/
│   ├── model.py
│   ├── detect.py
│   └── final_model.h5
│
├── PoliceUniDetect/
│   ├── notebooks/
│   ├── RT-DETR/
│   └── datasets/
│
├── yolo_deploy/
│   ├── main.py
│   ├── inference.py
│   ├── models.py
│   ├── run.py
│   └── dashboard/
│
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/public-safety-ai.git
cd public-safety-ai
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Backend

```bash
uvicorn main:app --reload
```

---

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

### Run Local Inference

```bash
python run.py
```

---

## 📈 Results

| Module             | Status               |
| ------------------ | -------------------- |
| Weapon Detection   | ✅ Working            |
| Violence Detection | ✅ Working            |
| Police Detection   | ⚠️ Under Improvement |
| Accident Detection | ✅ Working            |
| Unified Deployment | ✅ Working            |

---

## 🛠 Tech Stack

* Python
* PyTorch
* TensorFlow / Keras
* OpenCV
* YOLOv8
* FastAPI
* Streamlit
* RT-DETR

---

## 🔮 Future Improvements

* Improve police detection robustness
* Add crowd anomaly detection
* deploy on edge devices
* optimize inference latency
* improve multi-model fusion confidence logic

---

## 👨‍💻 Contributors

**HC**
AI / ML Engineer | Computer Vision | Deep Learning

---

<p align="center">
  ⭐ If you like this project, consider giving it a star!
</p>
