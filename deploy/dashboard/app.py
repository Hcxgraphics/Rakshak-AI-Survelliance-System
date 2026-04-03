from __future__ import annotations

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("MULTI_MODEL_API_URL", "http://localhost:8000/detect")

st.title("Multi-Model Detection Dashboard")

mode = st.radio("Select input type:", ["Image", "Video"])
file_types = ["jpg", "jpeg", "png", "webp"] if mode == "Image" else ["mp4", "avi", "mov", "mkv"]
uploaded_file = st.file_uploader(f"Upload a {mode.lower()} file", type=file_types)

if uploaded_file:
    try:
        response = requests.post(
            API_URL,
            data={"mode": mode.lower()},
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach detection server: {exc}")
        st.stop()

    if response.status_code != 200:
        try:
            error_payload = response.json()
            detail = error_payload.get("detail") or error_payload.get("error") or response.text
        except ValueError:
            detail = response.text
        st.error(f"Detection failed ({response.status_code}): {detail}")
        st.stop()

    data = response.json()
    st.subheader(f"Interpretation: {data.get('title', '---')}")
    st.write(f"Police detected: {data.get('police_detected', False)}")
    st.write(f"Weapon count: {data.get('weapon_count', 0)}")

    if mode == "Video":
        v_max = max(data.get("violence_score_police", 0.0), data.get("violence_score_lstm", 0.0))
        if v_max > 0:
            st.write(f"Violence score: {v_max:.2f}")
        if data.get("accident_confidence", 0.0) > 0:
            st.write(f"Accident class: {data.get('accident_class', 'N/A')}")
            st.write(f"Accident confidence: {data.get('accident_confidence', 0.0):.2f}")

    component_errors = data.get("component_errors", {})
    if component_errors:
        st.warning("Some model components failed during inference.")
        st.json(component_errors)

    img_bytes = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(img_bytes))
    caption = "Detection Output" if mode == "Image" else "Representative Video Frame"
    st.image(image, caption=caption, use_container_width=True)
