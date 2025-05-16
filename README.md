# 🕵️‍♂️ Crime Scene Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---
## 📄 Project Overview
An AI-powered video classification system designed to detect and categorize crime scenes into Arrest, Abuse, Arson, and Assault. This project leverages a 3D Convolutional Neural Network (3D CNN) to learn both spatial and temporal features from video clips, enabling efficient and accurate crime detection.

---
## 🧰 Tech Stack
Python 3.x

PyTorch

OpenCV

Streamlit

---
## 🏗️ Folder Structure

📁 dataset/
    ├── Arrest/
    ├── Abuse/
    ├── Arson/
    └── Assault/
📁 model/
    └── crime_detection_3dcnn.pth
📁 app/
    └── streamlit_app.py
📁 utils/
    └── data_loader.py
README.md
train.py
inference.py
requirements.txt

---
## Download the model:

[Click here to download model](https://drive.google.com/file/d/1aSRYi-b53IHcZ2qi-7SS16FQY_5uZ3ek/view?usp=sharing)

Place encodings.pickle in the project root folder.

---
## ⚙️ Installation
```
git clone https://github.com/yourusername/crime-scene-detection.git
cd crime-scene-detection
pip install -r requirements.txt
```
---
## 🚀 How to Run
### 1️⃣ Train the Model
```
python train.py
```
---
### 2️⃣ Run Inference on Video
```
python inference.py --video path_to_video.mp4
```
---
### 3️⃣ Launch the Streamlit Web App
```
streamlit run app/streamlit_app.py
```
---
## 📊 Expected Output
Upload a video clip of a crime scene.

Model predicts one of the four categories: Arrest, Abuse, Arson, Assault.

Result displayed in real-time on the Streamlit dashboard.

---
## 🎯 Project Goal
To assist law enforcement and security agencies by automating the detection of critical crime incidents in surveillance and bodycam footage, improving response time and efficiency.

---
## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

---
## 🙋‍♂️ Author
#### Bharath Chowdary
##### [GitHub](https://github.com/n-bharath-chowdary) 
##### [LinkedIn](https://www.linkedin.com/in/n-bharath-chowdary/)
