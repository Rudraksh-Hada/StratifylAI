import csv
import time
import random
import threading
import os
from datetime import datetime

NORMAL_LOGS = [
    "connection established successfully",
    "login successful",
    "normal authentication success",
    "query executed successfully",
    "session started",
    "file read successfully",
    "data fetched from database",
    "user logged out",
    "session closed normally",
    "request processed successfully",
    "cache hit for request",
    "user profile loaded",
    "password verified successfully",
    "token refreshed successfully",
    "api call completed",
    "session reopened after timeout",
    "login from new ip verified",
    "failed login attempt then success",
    "brief database connection retry success",
]

ANOMALY_LOGS = [
    ("failed login attempt", "brute_force"),
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
    ("infinite retry loop detected", "infinite_retry"),
    ("multiple session spam from single ip", "session_spam"),
    ("login attempt at unusual hour 03:15:00", "unusual_time"),
    ("port scanning detected multiple ports", "port_scan"),
    ("sql injection attempt detected", "sql_injection"),
    ("memory overflow critical error", "memory_overflow"),
]

NORMAL_IPS = [
    f"{random.randint(10,250)}.{random.randint(1,250)}.{random.randint(1,250)}.{random.randint(1,250)}"
    for _ in range(50)
]

ATTACKER_IPS = [
    "192.168.66.66",
    "45.33.32.156",
    "103.21.244.0",
    "198.51.100.1",
    "203.0.113.42",
]

stop_event = threading.Event()

def get_now():
    now = datetime.now()
    return now.strftime("%H:%M:%S"), now.strftime("%d-%m-%Y")

def simulate_logs(output_path, interval=1.0):
    """Continuously write logs to CSV. Inject anomaly every 10 seconds."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_header = not os.path.exists(output_path)
    last_anomaly_time = time.time()
    log_count = 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "date", "ip", "log", "label"])
        if write_header:
            writer.writeheader()

        print(f"📡 Real-time log simulator started → {output_path}")
        print("   Normal logs every 1s | Anomaly injected every 10s")
        print("   Press Ctrl+C to stop\n")

        while not stop_event.is_set():
            t, d = get_now()
            now = time.time()

            if now - last_anomaly_time >= 10:
                anomaly_log, anomaly_type = random.choice(ANOMALY_LOGS)
                ip = random.choice(ATTACKER_IPS)
                row = {"time": t, "date": d, "ip": ip, "log": anomaly_log, "label": anomaly_type}
                writer.writerow(row)
                f.flush()
                print(f"  🚨 ANOMALY [{anomaly_type}] | {ip} | {anomaly_log}")
                last_anomaly_time = now
            else:
                log_msg = random.choice(NORMAL_LOGS)
                ip = random.choice(NORMAL_IPS)
                row = {"time": t, "date": d, "ip": ip, "log": log_msg, "label": "normal"}
                writer.writerow(row)
                f.flush()
                log_count += 1
                if log_count % 5 == 0:
                    print(f"  ✅ Normal log #{log_count} | {ip} | {log_msg[:50]}")

            time.sleep(interval)

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "logs", "real_world.csv")
    try:
        simulate_logs(out)
    except KeyboardInterrupt:
        print("\n⛔ Simulator stopped.")
