import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
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
        return None

# =====================================================================
# 2. AUTOMATIC DEEP SEARCH FOR AUDIO FILES
# =====================================================================
DATA_DIR = "data"
X = []
y = []

print("Searching recursively for audio files inside the 'data' directory...")

# os.walk scans all nested folders automatically
for root, dirs, files in os.walk(DATA_DIR):
    # Check if there are any .wav files in the current folder
    wav_files = [f for f in files if f.endswith('.wav')]
    
    if wav_files:
        # Get the emotion name from the current folder name (e.g., 'OAF_angry' -> 'angry')
        current_folder = os.path.basename(root)
        emotion = current_folder.split('_')[-1].lower()
        
        print(f" -> Found {len(wav_files)} files for emotion: [{emotion.upper()}] inside '{current_folder}'")
        
        for file in wav_files:
            file_path = os.path.join(root, file)
            features = extract_mfcc(file_path)
            
            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

# Safety check if still empty
if len(X) == 0:
    print("\n❌ Error: Still could not find any .wav files!")
    print(f"Please ensure your files are unzipped inside: {os.path.abspath(DATA_DIR)}")
    exit()

print(f"\nExtraction complete! Processed {len(X)} total audio files.")

# =====================================================================
# 3. SPLIT DATASET
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 4. TRAIN THE NEURAL NETWORK
# =====================================================================
print("\nTraining MLP Neural Network Model...")
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                      hidden_layer_sizes=(300,), learning_rate='adaptive', 
                      max_iter=500, random_state=42)
model.fit(X_train, y_train)

# =====================================================================
# 5. EVALUATION METRICS (As per Internship instructions)
# =====================================================================
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("             FINAL MODEL PERFORMANCE REPORT            ")
print("="*50)
print(f"Overall Speech Emotion Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Metrics:")
print(classification_report(y_test, y_pred))
print("="*50)

# =====================================================================
# 6. SAVE MODEL ARTIFACTS
# =====================================================================
print("\nSaving speech emotion model...")
joblib.dump(model, 'speech_emotion_model.pkl')
print("Saved successfully! Created 'speech_emotion_model.pkl'")