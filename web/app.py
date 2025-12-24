import os
import cv2
import torch
import timm
import gdown
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

# -------------------------
# Model download
# -------------------------
MODEL_PATH = "efficientnet_deepfake.pth"
MODEL_URL = "https://drive.google.com/uc?id=1luWShga3o0VRiDt55X-yrSBRjn7ZOwxU"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------
# Device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# -------------------------
# Load Model
# -------------------------
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------
# Image Prediction
# -------------------------
def analyze_image(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)

    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()

    if fake_prob >= 0.65:
        return "FAKE", f"{fake_prob*100:.2f}%"
    elif 0.45 < fake_prob < 0.65:
        return "UNCERTAIN", f"{fake_prob*100:.2f}%"
    else:
        return "REAL", f"{real_prob*100:.2f}%"

# -------------------------
# Video Prediction (FAST)
# -------------------------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(fps, 1)

    fake_frames = 0
    total = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            total += 1
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)

            if probs[0][1].item() >= 0.65:
                fake_frames += 1

        frame_id += 1

    cap.release()

    if total == 0:
        return "UNCERTAIN", "0%"

    ratio = fake_frames / total

    if ratio >= 0.6:
        return "FAKE", f"{ratio*100:.2f}%"
    elif ratio >= 0.4:
        return "UNCERTAIN", f"{ratio*100:.2f}%"
    else:
        return "REAL", f"{(1-ratio)*100:.2f}%"

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="AI Deepfake Detection") as demo:
    gr.Markdown("# AI-Based Deepfake Detection")
    gr.Markdown("EfficientNet · Image & Video Analysis · Explainable AI")

    with gr.Tab("Image"):
        img = gr.Image(type="pil")
        out1 = gr.Textbox(label="Prediction")
        out2 = gr.Textbox(label="Confidence")
        btn1 = gr.Button("Analyze Image")
        btn1.click(analyze_image, img, [out1, out2])

    with gr.Tab("Video"):
        vid = gr.Video()
        out3 = gr.Textbox(label="Prediction")
        out4 = gr.Textbox(label="Confidence")
        btn2 = gr.Button("Analyze Video")
        btn2.click(analyze_video, vid, [out3, out4])

demo.launch()
