#!/usr/bin/env python3
"""
AI-Based System Log Anomaly Detection
======================================
Hybrid: Autoencoder (PyTorch) + Rule Engine

Usage:
    python main.py          → train + launch dashboard
    python main.py --train  → train only
    python main.py --serve  → serve only (model must exist)
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

BANNER = """
╔══════════════════════════════════════════════════════╗
║       StratifylAI — Log Anomaly Detection System      ║
║   Autoencoder (PyTorch) + Hybrid Rule Engine         ║
╚══════════════════════════════════════════════════════╝
"""

def train_model():
    print(BANNER)
    # Step 1: generate training data
    print("📂 Step 1/3 — Generating training data...")
    from data.generate_training_data import generate_normal_logs
    import pandas as pd

    logs_dir   = os.path.join(BASE_DIR, "logs")
    data_path  = os.path.join(logs_dir, "training_data.csv")
    os.makedirs(logs_dir, exist_ok=True)

    if os.path.exists(data_path):
        print(f"   ✅ Training data already exists ({data_path}), skipping generation.")
    else:
        df = generate_normal_logs(10000)
        df.to_csv(data_path, index=False)
        print(f"   ✅ Generated 10,000 logs → {data_path}")

    # Step 2: train autoencoder
    print("\n🧠 Step 2/3 — Training Autoencoder...")
    from model.train import train
    threshold = train()

    print(f"\n✅ Model ready! Threshold = {threshold:.6f}")
    return threshold


def model_exists():
    model_dir = os.path.join(BASE_DIR, "model")
    return (
        os.path.exists(os.path.join(model_dir, "autoencoder.pth")) and
        os.path.exists(os.path.join(model_dir, "vectorizer.pkl")) and
        os.path.exists(os.path.join(model_dir, "threshold.pkl"))
    )


def launch_dashboard():
    port = int(os.environ.get("PORT", 5000))
    print("\n🌐 Step 3/3 — Launching Dashboard...")
    print(f"   Open your browser → http://0.0.0.0:{port}")
    print("   Press Ctrl+C to stop\n")
    from app import start
    start()


def main():
    parser = argparse.ArgumentParser(description="StratifylAI Anomaly Detection")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--serve", action="store_true", help="Serve dashboard only")
    args = parser.parse_args()

    if args.train:
        train_model()

    elif args.serve:
        if not model_exists():
            print("⚠️  Model not found. Training first...")
            train_model()
        launch_dashboard()

    else:
        # Default: train + serve
        if not model_exists():
            train_model()
        else:
            print(BANNER)
            print("✅ Trained model found — skipping training.")
            print("   (Delete model/*.pth and *.pkl to retrain)\n")
        launch_dashboard()


if __name__ == "__main__":
    main()
