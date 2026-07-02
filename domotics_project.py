from pathlib import Path
import numpy as np

#loading the path of the dataset
base_path = Path("/Users/computer/Desktop/project_domotics/UCI HAR Dataset")

def load_inertial_signals(split="train"):
    signals = []
    folder = base_path / split / "Inertial Signals" #the  files inside "Inertial Signals" there are 6 channels that will be stored in the list signals
    print("Looking in:", folder) 
    files = [
        f"total_acc_x_{split}.txt", #total acceleration x-axis
        f"total_acc_y_{split}.txt", #total acceleration y-axis
        f"total_acc_z_{split}.txt", #total acceleration z-axis
        f"body_gyro_x_{split}.txt", #body gyroscope x-axis
        f"body_gyro_y_{split}.txt", #body gyroscope y-axis
        f"body_gyro_z_{split}.txt" #body gyroscope z-axis
    ]
    for fname in files:
        data = np.loadtxt(folder / fname)
        signals.append(data)
    X = np.stack(signals, axis=1).astype(np.float32)  # (N, 6, 128)
    return X

def load_labels(split="train"):
    y_path = base_path / split / f"y_{split}.txt"
    y = np.loadtxt(y_path).astype(int) - 1  # da 1..6 a 0..5
    return y

X_train = load_inertial_signals("train")
y_train = load_labels("train")

X_test = load_inertial_signals("test")
y_test = load_labels("test")

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
