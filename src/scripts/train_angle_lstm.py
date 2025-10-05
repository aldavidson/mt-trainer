import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# ---------------------------
# 1. LSTM Model
# ---------------------------

class AngleLSTMClassifier(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ---------------------------
# 2. Dataset
# ---------------------------

class AngleSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.data = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ---------------------------
# 3. Load JSONs and Prepare Data
# ---------------------------

def extract_angle_vector(json_path, angle_keys=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
        angles = data.get("angles", {})
        if angle_keys is None:
            angle_keys = sorted(angles.keys())
        return [angles.get(k, 0.0) for k in angle_keys], angle_keys


def load_dataset(root_dir, sequence_length=16):
    X, y = [], []
    label_names = []

    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        label_names.append(label)
        for clip_folder in os.listdir(class_dir):
            clip_path = os.path.join(class_dir, clip_folder)
            if not os.path.isdir(clip_path):
                continue
            frame_files = sorted([
                os.path.join(clip_path, f)
                for f in os.listdir(clip_path)
                if f.endswith(".json")
            ])
            if len(frame_files) < sequence_length:
                continue  # skip too short

            sequence = []
            angle_keys = None
            for f in frame_files[:sequence_length]:
                vec, angle_keys = extract_angle_vector(f, angle_keys)
                sequence.append(vec)

            if len(sequence) == sequence_length:
                X.append(sequence)
                y.append(label)

    return X, y, angle_keys


# ---------------------------
# 4. Training Function
# ---------------------------

def train(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                logits = model(val_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(val_y.numpy())
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Epoch {epoch+1}/{num_epochs} - Val Accuracy: {acc:.3f}")


# ---------------------------
# 5. Main
# ---------------------------

if __name__ == "__main__":
    DATASET_PATH = "dataset"  # folder with labeled technique folders
    SEQ_LEN = 16
    BATCH_SIZE = 8
    EPOCHS = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    X, y, angle_keys = load_dataset(DATASET_PATH, sequence_length=SEQ_LEN)
    print(f"Loaded {len(X)} sequences with {len(angle_keys)} angles.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

    train_dataset = AngleSequenceDataset(X_train, y_train)
    val_dataset = AngleSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AngleLSTMClassifier(input_size=len(angle_keys), num_classes=num_classes)
    train(model, train_loader, val_loader, num_epochs=EPOCHS, lr=1e-3, device=DEVICE)

    # Print label mapping
    print("Label classes:", list(label_encoder.classes_))
    