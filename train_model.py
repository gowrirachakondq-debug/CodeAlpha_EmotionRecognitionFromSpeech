mport os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. FEATURE EXTRACTION

def extract_mfcc(file_path):
    try:
        signal, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception:
        return None


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_DIR = "data"

X = []
y = []

print("Searching dataset...\n")

for root, dirs, files in os.walk(DATA_DIR):

    wav_files = [f for f in files if f.endswith(".wav")]

    if len(wav_files) > 0:

        folder = os.path.basename(root)
        emotion = folder.split("_")[-1].lower()

        print(f"{emotion}: {len(wav_files)} files")

        for file in wav_files:

            path = os.path.join(root, file)

            features = extract_mfcc(path)

            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("No audio files found.")
    exit()

print("\nTotal samples:", len(X))

# 3. ENCODE LABELS

encoder = LabelEncoder()
y = encoder.fit_transform(y)
num_classes = len(np.unique(y))
y = to_categorical(y)
# 4. RESHAPE FOR CNN

X = X.reshape(X.shape[0], 40, 1, 1)

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# 6. BUILD CNN

print("\nBuilding CNN...\n")

model = Sequential()
model.add(
    Conv2D(
        32,
        (3,1),
        activation="relu",
        input_shape=(40,1,1)
    )
)
model.add(MaxPooling2D((2,1)))

model.add(
    Conv2D(
        64,
        (3,1),
        activation="relu"
    )
)
model.add(MaxPooling2D((2,1)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# 7. TRAIN MODEL

print("\nTraining CNN...\n")
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 8. EVALUATE MODEL

loss, accuracy = model.evaluate(X_test, y_test)
print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy : {accuracy*100:.2f}%")
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)
print(classification_report(
    true_labels,
    predictions,
    target_names=encoder.classes_
))


# 9. SAVE MODEL

model.save("speech_emotion_cnn.keras")
print("\nModel saved successfully!")
