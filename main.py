import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load trained CNN model
model = load_model("speech_emotion_cnn.keras")

# Load emotion labels from dataset

DATA_DIR = "data"

emotions = []

for root, dirs, files in os.walk(DATA_DIR):
    wav_files = [f for f in files if f.endswith(".wav")]

    if wav_files:
        folder = os.path.basename(root)
        emotion = folder.split("_")[-1].lower()
        emotions.append(emotion)

emotions = sorted(list(set(emotions)))

encoder = LabelEncoder()
encoder.fit(emotions)

# MFCC Feature Extraction

def extract_mfcc(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc

# Predict Function

def predict_emotion(audio_path):

    features = extract_mfcc(audio_path)

    features = features.reshape(1, 40, 1, 1)

    prediction = model.predict(features, verbose=0)

    predicted_index = np.argmax(prediction)

    emotion = encoder.inverse_transform([predicted_index])[0]

    confidence = np.max(prediction) * 100

    print("\nPredicted Emotion :", emotion.upper())
    print(f"Confidence : {confidence:.2f}%")

# Main

audio_file = input("Enter audio file path: ")

if os.path.exists(audio_file):
    predict_emotion(audio_file)
else:
    print("File not found.")
