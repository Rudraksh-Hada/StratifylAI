import os
import sys
import csv
import time
import threading
import random
from datetime import datetime
from collections import deque
from flask import Flask, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from model.detector import HybridDetector

app = Flask(__name__)

BASE_DIR    = os.path.dirname(__file__)
REAL_WORLD  = os.path.join(BASE_DIR, "logs", "real_world.csv")

# ── Shared state for the simulator and dashboard ───────────────────────────────────
recent_logs   = deque(maxlen=200)
anomaly_stats = {
    "total": 0, "anomalies": 0, "normal": 0,
    "by_type": {}, "timeline": deque(maxlen=60)
}
stats_lock    = threading.Lock()
detector      = None

# ── Simulator logic that generates log entries in the background ───────────────────
NORMAL_LOGS = [
    "connection established successfully",
    "login successful", "normal authentication success",
    "query executed successfully", "session started",
    "file read successfully", "data fetched from database",
    "user logged out", "session closed normally",
    "request processed successfully", "cache hit for request",
    "user profile loaded", "password verified successfully",
    "token refreshed successfully", "api call completed",
    "session reopened after timeout", "login from new ip verified",
    "failed login attempt then success",
    "brief database connection retry success",
]

ANOMALY_LOGS = [
    ("failed login attempt", "brute_force"),
    ("failed login attempt", "brute_force"),
    ("failed login attempt", "brute_force"),
    ("unauthorized access attempt detected", "unauthorized_access"),
    ("cpu usage spike 98 percent", "cpu_spike"),
    ("system crash detected kernel panic", "system_crash"),
    ("ddos request flood 5000 requests per second", "ddos"),
    ("privilege escalation attempt root access", "privilege_escalation"),
    ("data exfiltration large file transfer 2gb", "data_exfiltration"),
    ("database connection failed repeatedly", "db_failure"),
    ("sql injection attempt detected", "sql_injection"),
    ("memory overflow critical error", "memory_overflow"),
    ("port scanning detected multiple ports", "port_scan"),
    ("infinite retry loop detected", "infinite_retry"),
    ("multiple session spam from single ip", "session_spam"),
]

NORMAL_IPS   = [f"{random.randint(10,250)}.{random.randint(1,250)}.{random.randint(1,250)}.{random.randint(1,250)}" for _ in range(50)]
ATTACKER_IPS = ["192.168.66.66","45.33.32.156","103.21.244.0","198.51.100.1","203.0.113.42"]


def simulate_and_detect():
    global detector
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

    write_header = not os.path.exists(REAL_WORLD)
    last_anomaly = time.time()

    with open(REAL_WORLD, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time","date","ip","log","label"])
        if write_header:
            writer.writeheader()

        while True:
            now  = time.time()
            ts   = datetime.now()
            t    = ts.strftime("%H:%M:%S")
            d    = ts.strftime("%d-%m-%Y")

            if now - last_anomaly >= 10:
                log_msg, label = random.choice(ANOMALY_LOGS)
                ip = random.choice(ATTACKER_IPS)
                last_anomaly = now
            else:
                log_msg = random.choice(NORMAL_LOGS)
                ip      = random.choice(NORMAL_IPS)
                label   = "normal"

            row = {"time": t, "date": d, "ip": ip, "log": log_msg, "label": label}
            writer.writerow(row)
            f.flush()

            # Run the anomaly detector for this new log entry
            result = detector.detect(row)
            result["true_label"] = label

            with stats_lock:
                recent_logs.appendleft(result)
                anomaly_stats["total"] += 1
                if result["is_anomaly"]:
                    anomaly_stats["anomalies"] += 1
                    atype = result["anomaly_type"]
                    anomaly_stats["by_type"][atype] = anomaly_stats["by_type"].get(atype, 0) + 1
                else:
                    anomaly_stats["normal"] += 1

                anomaly_stats["timeline"].appendleft({
                    "time": t,
                    "is_anomaly": result["is_anomaly"],
                    "ml_error": result["ml_error"]
                })

            time.sleep(1)


# ── Flask routes for the dashboard and API ───────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/logs")
def api_logs():
    with stats_lock:
        logs = list(recent_logs)[:50]
    return jsonify(logs)


@app.route("/api/stats")
def api_stats():
    with stats_lock:
        s = dict(anomaly_stats)
        s["timeline"] = list(s["timeline"])[:30]
        rate = round(s["anomalies"] / s["total"] * 100, 1) if s["total"] else 0
        s["anomaly_rate"] = rate
    return jsonify(s)


@app.route("/api/threshold")
def api_threshold():
    return jsonify({"threshold": round(detector.threshold, 6)})


def start():
    global detector
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🔍 Loading Hybrid Detector...")
    detector = HybridDetector()
    print(f"  ✅ Model loaded | Threshold: {detector.threshold:.6f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    t = threading.Thread(target=simulate_and_detect, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    print("  📡 Log simulator running in background...")
    print(f"  🌐 Dashboard → http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", debug=False, port=port, use_reloader=False)


if __name__ == "__main__":
    start()
