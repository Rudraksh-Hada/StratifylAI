import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.autoencoder import Autoencoder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# ── Rule-based thresholds ──────────────────────────────────────────────────
MAX_FAILED_LOGINS   = 3      # per IP in window
BRUTE_FORCE_WINDOW  = 30     # seconds
NIGHT_HOURS         = (0, 5) # suspicious login window
DDoS_REQUEST_LIMIT  = 20     # requests per IP in window
DDoS_WINDOW         = 10     # seconds
SESSION_SPAM_LIMIT  = 10     # sessions per IP in window

ANOMALY_KEYWORDS = {
    "unauthorized":       "unauthorized_access",
    "privilege escalation": "privilege_escalation",
    "exfiltration":       "data_exfiltration",
    "sql injection":      "sql_injection",
    "port scan":          "port_scan",
    "kernel panic":       "system_crash",
    "memory overflow":    "memory_overflow",
    "ddos":               "ddos",
    "flood":              "ddos",
    "crash":              "system_crash",
    "overflow":           "memory_overflow",
}

class HybridDetector:
    def __init__(self):
        self._load_model()
        # Rule-based state tracking
        self.failed_logins   = defaultdict(list)  # ip → [timestamps]
        self.request_times   = defaultdict(list)
        self.session_counts  = defaultdict(list)

    def _load_model(self):
        meta       = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))
        self.threshold  = meta["threshold"]
        self.input_dim  = meta["input_dim"]
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        self.scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

        self.model = Autoencoder(self.input_dim)
        self.model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "autoencoder.pth"),
                       map_location="cpu")
        )
        self.model.eval()

    def _extract(self, df):
        from model.train import extract_features
        features, _, _ = extract_features(df, self.vectorizer, self.scaler, fit=False)
        return features

    # ── ML Layer ────────────────────────────────────────────────────────────
    def ml_detect(self, df):
        features = self._extract(df)
        X = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(X)
            errors = torch.mean((X - recon) ** 2, dim=1).numpy()
        return errors

    # ── Rule Layer ───────────────────────────────────────────────────────────
    def rule_detect(self, row):
        now     = time.time()
        ip      = str(row.get("ip", ""))
        log_msg = str(row.get("log", "")).lower()
        flags   = []

        # 1. Keyword-based detection
        for kw, atype in ANOMALY_KEYWORDS.items():
            if kw in log_msg:
                flags.append(atype)

        # 2. Brute force: failed login tracking
        if "failed login" in log_msg:
            self.failed_logins[ip].append(now)
        self.failed_logins[ip] = [
            t for t in self.failed_logins[ip]
            if now - t < BRUTE_FORCE_WINDOW
        ]
        if len(self.failed_logins[ip]) >= MAX_FAILED_LOGINS:
            flags.append("brute_force")

        # 3. DDoS: request flooding
        self.request_times[ip].append(now)
        self.request_times[ip] = [
            t for t in self.request_times[ip]
            if now - t < DDoS_WINDOW
        ]
        if len(self.request_times[ip]) >= DDoS_REQUEST_LIMIT:
            flags.append("ddos")

        # 4. Unusual login hour
        try:
            hour = int(str(row.get("time", "12:00:00")).split(":")[0])
            if NIGHT_HOURS[0] <= hour <= NIGHT_HOURS[1]:
                if "login" in log_msg:
                    flags.append("unusual_time")
        except:
            pass

        # 5. Session spam
        if "session" in log_msg:
            self.session_counts[ip].append(now)
            self.session_counts[ip] = [
                t for t in self.session_counts[ip]
                if now - t < BRUTE_FORCE_WINDOW
            ]
            if len(self.session_counts[ip]) >= SESSION_SPAM_LIMIT:
                flags.append("session_spam")

        return flags

    # ── Combined Detection ────────────────────────────────────────────────
    def detect(self, row: dict) -> dict:
        df = pd.DataFrame([row])
        errors = self.ml_detect(df)
        ml_error = float(errors[0])
        ml_anomaly = ml_error > self.threshold

        rule_flags = self.rule_detect(row)
        rule_anomaly = len(rule_flags) > 0

        is_anomaly = ml_anomaly or rule_anomaly

        anomaly_type = "normal"
        if rule_flags:
            anomaly_type = rule_flags[0]
        elif ml_anomaly:
            anomaly_type = "ml_detected"

        return {
            "time":         row.get("time", ""),
            "date":         row.get("date", ""),
            "ip":           row.get("ip", ""),
            "log":          row.get("log", ""),
            "ml_error":     round(ml_error, 6),
            "threshold":    round(self.threshold, 6),
            "ml_anomaly":   ml_anomaly,
            "rule_flags":   rule_flags,
            "is_anomaly":   is_anomaly,
            "anomaly_type": anomaly_type,
            "severity":     _severity(anomaly_type, ml_error, self.threshold),
        }


def _severity(atype, error, threshold):
    critical = {"brute_force", "ddos", "privilege_escalation",
                "data_exfiltration", "sql_injection", "system_crash"}
    high     = {"unauthorized_access", "port_scan", "memory_overflow"}
    if atype in critical:
        return "CRITICAL"
    if atype in high:
        return "HIGH"
    if atype == "ml_detected":
        ratio = error / threshold if threshold > 0 else 1
        return "HIGH" if ratio > 3 else "MEDIUM"
    if atype != "normal":
        return "MEDIUM"
    return "NORMAL"
