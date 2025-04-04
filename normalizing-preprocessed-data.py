import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.load('./processed_data/X_for-norm_training.npy')  
X_val = np.load('./processed_data/X_for-norm_validation.npy')
X_test = np.load('./processed_data/X_for-norm_testing.npy')

print("Before normalization: Mean =", np.mean(X_train), "Std =", np.std(X_train))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

print("After normalization: Mean =", np.mean(X_train_scaled), "Std =", np.std(X_train_scaled))

np.save('./processed_data/X_for-norm_training_scaled.npy', X_train_scaled)
np.save('./processed_data/X_for-norm_validation_scaled.npy', X_val_scaled)
np.save('./processed_data/X_for-norm_testing_scaled.npy', X_test_scaled)

print("Normalized data saved successfully!")