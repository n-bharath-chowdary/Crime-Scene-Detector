import os
import cv2
import torch
import streamlit as st
import numpy as np
from torchvision import transforms
from torch import nn

# ========== CONFIG ==========
MODEL_PATH = "crime_model.pth"
CLASSES = os.listdir("dataset")  # Same folder names used for labels
FRAME_COUNT = 16
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== MODEL ==========
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super(Simple3DCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    model = Simple3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ========== UTILS ==========
def extract_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < FRAME_COUNT:
        cap.release()
        st.error("Video too short for analysis.")
        return None

    start = np.random.randint(0, total - FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for _ in range(FRAME_COUNT):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(frame)

    cap.release()
    if len(frames) != FRAME_COUNT:
        return None

    video_tensor = torch.stack(frames)  # shape: (16, 3, 112, 112)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # -> (3, 16, 112, 112)
    video_tensor = video_tensor.unsqueeze(0).to(DEVICE)  # -> (1, 3, 16, 112, 112)

    return video_tensor


# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Crime Scene Detector", layout="centered")
st.title("🔍 Crime Scene Video Detector")
st.caption("Upload a video, and the AI will guess the crime category.")

video_file = st.file_uploader("Upload a crime video (.mp4 / .avi)", type=["mp4", "avi"])

if video_file:
    temp_path = "temp_vid.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())
    st.video(temp_path)

    with st.spinner("Analyzing the video..."):
        model = load_model()
        input_tensor = extract_clip(temp_path)

        if input_tensor is not None:
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_label = CLASSES[pred_idx]
                confidence = probs[0][pred_idx].item() * 100

            st.success(f"🧠 Predicted Crime Type: **{pred_label}**")
            # st.info(f"🔒 Confidence: `{confidence:.2f}%`")
        else:
            st.error("❌ Could not process the video properly.")

    os.remove(temp_path)
