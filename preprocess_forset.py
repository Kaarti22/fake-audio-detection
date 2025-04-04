import os
import numpy as np
import librosa
import tqdm

SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
MAX_LEN = SAMPLE_RATE * DURATION

DATASET_PATHS = {
    'for-norm': './dataset/for-norm',
    'for-rerec': './dataset/for-rerecorded'
}

OUTPUT_DIR = './processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(signal) > MAX_LEN:
            signal = signal[:MAX_LEN]
        else:
            signal = np.pad(signal, (0, MAX_LEN - len(signal)))

        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

for version, base_path in DATASET_PATHS.items():
    print(f"\nProcessing version: {version}")

    for split in ['training', 'validation', 'testing']:
        X = []
        y = []

        for label in ['real', 'fake']:
            folder = os.path.join(base_path, split, label)
            files = os.listdir(folder)

            for file in tqdm.tqdm(files, desc=f"{split}/{label}"):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(0 if label == 'real' else 1)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        np.save(os.path.join(OUTPUT_DIR, f'X_{version}_{split}.npy'), X)
        np.save(os.path.join(OUTPUT_DIR, f'y_{version}_{split}.npy'), y)

        print(f"Saved: X_{version}_{split}.npy and y_{version}_{split}.npy")
