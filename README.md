# Facial Emotion Recognition using Deep Learning (CNN-based)

## Introduction
This project implements a real-time facial emotion recognition system using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The model is trained and evaluated on multiple benchmark datasets, including FER2013, AffectNet, MicroExpression, and a custom combined dataset (AFM). The system can recognize various facial emotions such as happiness, sadness, anger, surprise, fear, and more.

## Features:
* Preprocessing of facial images using OpenCV and Haar Cascades
* A CNN-based architecture built with TensorFlow/Keras
* Training with class balancing, early stopping, and learning rate scheduling
* Evaluation using classification reports, confusion matrices, and training curves
* Real-time emotion recognition with a GUI built using Tkinter
* Support for multiple datasets and flexible model loading

## Datasets:
* [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
* [AffectNet](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned)
* [MicroExpression (custom-labeled)](https://www.kaggle.com/datasets/kmirfan/micro-expressions)
* [Combined dataset: AFM](https://drive.google.com/file/d/1gmLwu5rtfynbFO2ScImslFZGVHPCE8sN/view)

## How to Use
* Clone the Repository
```bash
git clone https://github.com/hamidrg/Facial_Emotion_Recognition.git
cd Facial_Emotion_Recognition
```

* Create and Activate a Virtual Environment (optional but recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

* Install Dependencies
```bash
pip install -r requirements.txt
```

* Run the Real-Time Emotion Detection App
```bash
python Realtime_Emotion_detect.py
```
This will open a GUI where your webcam stream is analyzed for facial emotions in real time.

---

> *Thank you for checking out this project!*
> *If you have suggestions for improvements, new features, or bug fixes, feel free to fork the repository and open a pull request.*
> *If you find this project useful, please ⭐️ star it — it helps others discover it!*
