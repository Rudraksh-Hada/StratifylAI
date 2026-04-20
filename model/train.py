import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.autoencoder import Autoencoder, get_device

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
LOGS_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")

def extract_features(df, vectorizer=None, scaler=None, fit=True):
    # 1. Convert the log text into numerical features using TF-IDF
    if fit:
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(df["log"].astype(str)).toarray()
    else:
        tfidf = vectorizer.transform(df["log"].astype(str)).toarray()

    # 2. Extract the hour of day from the log timestamp
    def parse_hour(t):
        try:
            return int(str(t).split(":")[0])
        except:
            return 12
    hours = df["time"].apply(parse_hour).values.reshape(-1, 1)

    # 3. Turn the IP address into four numeric octet features
    def parse_ip(ip):
        try:
            parts = str(ip).split(".")
            return [int(p) for p in parts[:4]]
        except:
            return [0, 0, 0, 0]
    ip_features = np.array(df["ip"].apply(parse_ip).tolist())

    features = np.hstack([tfidf, hours, ip_features])

    if fit:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    return features, vectorizer, scaler


def train(epochs=50, batch_size=64, lr=1e-3):
    print("━" * 60)
    print("  🧠 AUTOENCODER TRAINING STARTED")
    print("━" * 60)

    # Load training log data from disk
    data_path = os.path.join(LOGS_DIR, "training_data.csv")
    if not os.path.exists(data_path):
        print("⚠️  Training data not found. Generating...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
        import generate_training_data
        df = generate_training_data.generate_normal_logs(10000)
        os.makedirs(LOGS_DIR, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"  📂 Loaded {len(df)} training logs")

    # Feature extraction: create the input vectors used for training
    print("  🔢 Extracting features (TF-IDF + Time + IP)...")
    features, vectorizer, scaler = extract_features(df, fit=True)
    print(f"  ✅ Feature shape: {features.shape}")

    # Save the preprocessing objects so inference can use the same scaling and vectorization
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))

    # Build the autoencoder model and optimizer for training
    device = get_device()
    input_dim = features.shape[1]
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(X, X)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n  📐 Model input dim : {input_dim}")
    print(f"  ⚙️  Epochs          : {epochs}")
    print(f"  📦 Batch size       : {batch_size}")
    print(f"  💻 Device           : {device}")
    print()

    # Training loop: run through all epochs and save the best model
    best_loss = float("inf")
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            loss = criterion(pred, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pth"))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}]  Loss: {avg_loss:.6f}  {'⭐ Best' if avg_loss == best_loss else ''}")

    # Compute the anomaly threshold from reconstruction errors and save it
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pth")))
    with torch.no_grad():
        X_dev = X.to(device)
        recon = model(X_dev)
        errors = torch.mean((X_dev - recon) ** 2, dim=1).cpu().numpy()

    threshold = float(np.mean(errors) + 2 * np.std(errors))
    joblib.dump({"threshold": threshold, "input_dim": input_dim},
                os.path.join(MODEL_DIR, "threshold.pkl"))

    print()
    print("━" * 60)
    print(f"  ✅ Training complete!")
    print(f"  📉 Best Loss      : {best_loss:.6f}")
    print(f"  🎯 Threshold set  : {threshold:.6f}")
    print(f"  💾 Model saved    → model/autoencoder.pth")
    print("━" * 60)

    return threshold


if __name__ == "__main__":
    train()
