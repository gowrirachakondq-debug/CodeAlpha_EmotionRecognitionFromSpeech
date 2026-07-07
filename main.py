import os
import numpy as np
import librosa
import joblib

# =====================================================================
# 1. FEATURE EXTRACTION FUNCTION
# =====================================================================
def extract_mfcc(file_path):
    """Loads an audio file and extracts its mean MFCC features."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

# =====================================================================
# 2. LOAD TRAINED MODEL
# =====================================================================
MODEL_PATH = 'speech_emotion_model.pkl'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Could not find '{MODEL_PATH}'. Please run train_model.py first!")
    exit()

print("Loading Speech Emotion Recognition Model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!\n")

# =====================================================================
# 3. INTERACTIVE PREDICTION LOOP
# =====================================================================
print("="*60)
print("              SPEECH EMOTION RECOGNITION SYSTEM               ")
print("="*60)

while True:
    print("\nEnter the full path to a .wav file to test (or type 'quit' to exit):")
    file_path = input(">>> ").strip().strip('"')  # Trims whitespace or dragged-in quotes
    
    if file_path.lower() == 'quit':
        print("\nExiting system. Goodbye!")
        break
        
    if not os.path.exists(file_path):
        print("❌ Error: Invalid file path. Please check the location and try again.")
        continue
        
    if not file_path.endswith('.wav'):
        print("⚠️ Warning: Please provide an audio file ending in '.wav'")
        continue

    print("\nAnalyzing vocal frequencies...")
    features = extract_mfcc(file_path)
    
    if features is not None:
        # Reshape for single sample prediction
        features_reshaped = features.reshape(1, -1)
        
        # Predict emotion class
        prediction = model.predict(features_reshaped)[0]
        
        # Predict probability distributions
        probabilities = model.predict_proba(features_reshaped)[0]
        max_prob = np.max(probabilities) * 100
        
        print("\n" + "-"*40)
        print(f"  PREDICTED EMOTION : [ {prediction.upper()} ]")
        print(f"  CONFIDENCE SCORE  : {max_prob:.2f}%")
        print("-"*40)