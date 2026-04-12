# 🛡️ StratifyAI — Log Anomaly Detection System

> Hybrid AI system combining **Autoencoder (PyTorch)** + **Rule-Based Engine** to detect anomalies in real-time system logs.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (train + dashboard)
python main.py

# 3. Open browser
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
log_anomaly_detection/
├── data/
│   ├── generate_training_data.py   # Generates 10,000 normal logs
│   └── real_time_simulator.py      # Standalone simulator (optional)
├── model/
│   ├── autoencoder.py              # PyTorch Autoencoder architecture
│   ├── train.py                    # Training pipeline
│   └── detector.py                 # Hybrid detection engine
├── logs/
│   ├── training_data.csv           # Generated after training
│   └── real_world.csv              # Live log stream
├── templates/
│   └── index.html                  # Real-time dashboard
├── app.py                          # Flask server + API
├── main.py                         # Entry point
└── requirements.txt
```

---

## 🧠 How It Works

### Model: Autoencoder (PyTorch)
```
Input (105 features)
  → Encoder [128 → 64 → 32]
  → Latent Space (32)
  → Decoder [32 → 64 → 128]
  → Output (105 features)

Reconstruction Error = MSE(Input, Output)
Error > Threshold → 🚨 ANOMALY
```

### Feature Extraction
| Feature | Method |
|---|---|
| Log text | TF-IDF (100 features) |
| Time | Hour extraction (1 feature) |
| IP Address | 4 octets (4 features) |
| **Total** | **105 features** |

### Hybrid Detection
| Layer | Detects |
|---|---|
| 🤖 ML (Autoencoder) | Unknown patterns, system failures |
| 📏 Rule Engine | Brute force, DDoS, unusual time |

---

## 🚨 Anomalies Detected

| Type | Layer |
|---|---|
| Brute Force Login | Rule |
| DDoS / Flooding | Rule + ML |
| SQL Injection | Rule |
| Privilege Escalation | Rule + ML |
| Data Exfiltration | ML |
| System Crash | ML |
| CPU Spike | ML |
| Unusual Login Time | Rule |
| Port Scanning | Rule |
| Session Spam | Rule |
| Memory Overflow | ML |

---

## 🌐 API Endpoints

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard |
| `GET /api/logs` | Last 50 logs with detection results |
| `GET /api/stats` | Total/anomaly counts, breakdown, timeline |
| `GET /api/threshold` | Current ML threshold |

---

## 🎓 Viva Line

> *"We developed a hybrid anomaly detection system combining machine learning and rule-based logic to detect both structural and behavioral anomalies in real-time system logs."*
