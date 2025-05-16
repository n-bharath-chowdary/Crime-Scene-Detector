import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ========== CONFIG ==========
VIDEO_DIR = "dataset/"
CLASSES = os.listdir(VIDEO_DIR)
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAME_COUNT = 16  # Use 16-frame clips
IMG_SIZE = 112    # Downscale to 112x112
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.001

# ========== DATASET ==========
class CrimeVideoDataset(Dataset):
    def __init__(self, video_dir, classes, transform=None):
        self.video_paths = []
        self.labels = []
        self.classes = classes
        self.transform = transform

        for idx, cls in enumerate(classes):
            class_folder = os.path.join(video_dir, cls)
            for file in os.listdir(class_folder):
                if file.endswith(".mp4") or file.endswith(".avi"):
                    self.video_paths.append(os.path.join(class_folder, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.load_video(video_path)

        # Already tensors, so skip transform
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)

        return video_tensor, label


    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < FRAME_COUNT:
            cap.release()
            raise ValueError(f"Video too short: {path}")

        start = random.randint(0, total_frames - FRAME_COUNT)
        frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(FRAME_COUNT):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        frames = [torch.tensor(f, dtype=torch.float32).permute(2, 0, 1) / 255.0 for f in frames]
        return frames

# ========== MODEL ==========
class Simple3DCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)

# ========== TRAINING ==========
def train():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CrimeVideoDataset(VIDEO_DIR, CLASSES, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Simple3DCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for clips, labels in tqdm(dataloader):
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "crime_model.pth")
    print("✅ Model saved as crime_model.pth")

if __name__ == "__main__":
    train()
