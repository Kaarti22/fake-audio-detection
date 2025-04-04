from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    print(f"Converted {mp3_path} -> {wav_path}")

import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.load('./processed_data/X_for-norm_training.npy')
scaler = StandardScaler()
scaler.fit(X_train.reshape(len(X_train), -1))

import librosa
from tensorflow.keras.models import load_model

SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
MAX_LEN = SAMPLE_RATE * DURATION

def extract_mfcc_scaled(file_path, scaler):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(signal) > MAX_LEN:
        signal = signal[:MAX_LEN]
    else:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    mfcc_scaled = scaler.transform(mfcc.reshape(1, -1)).reshape(mfcc.shape)
    return mfcc_scaled

model = load_model("deepfake_voice_model.h5")

convert_mp3_to_wav("sample.mp3", "converted.wav")

mfcc_scaled = extract_mfcc_scaled("converted.wav", scaler)

input_tensor = mfcc_scaled.reshape(1, 40, 157, 1)

prediction = model.predict(input_tensor)[0][0]
label = "FAKE" if prediction > 0.5 else "REAL"

print(f"Prediction: {label} (Confidence: {prediction:.4f})")
