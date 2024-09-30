import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

# 1. Load the Trained Model
def load_model(model_path, device):
    model = models.resnet50(pretrained=False)
    num_classes = 7  # FER-2013 has 7 emotion classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 2. Set Up OpenCV for Video Capture
def initialize_video_capture():
    for camera_index in range(0, 5):  # Try indices 0 to 4
        # Option 1: Let OpenCV choose the default backend
        cap = cv2.VideoCapture(camera_index)
        
        # Option 2: Specify DirectShow backend (uncomment the line below)
        # cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            print(f"Webcam opened at index {camera_index}")
            return cap
        cap.release()
    raise IOError("Cannot open webcam")

def load_face_detector():
    # Using Haar Cascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def get_preprocess_transform():
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def predict_emotion(face_img, model, preprocess, device):
    input_tensor = preprocess(face_img).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    emotion = EMOTION_LABELS.get(predicted.item(), "Unknown")
    return emotion

def run_emotion_recognition(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(model_path, device)
    cap = initialize_video_capture()
    face_cascade = load_face_detector()
    preprocess = get_preprocess_transform()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y+h, x:x+w]
            # Predict emotion
            emotion = predict_emotion(face, model, preprocess, device)
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Put emotion label
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36,255,12), 2)

        cv2.imshow('Real-Time Facial Emotion Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "best_model.pth"  # Path to your trained model
    run_emotion_recognition(model_path)
