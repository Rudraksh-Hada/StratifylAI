import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

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
    "backup completed successfully",
    "scheduler task executed",
    "health check passed",
    "configuration loaded successfully",
    "service started normally",
    "database connection pool initialized",
    "email notification sent",
    "file uploaded successfully",
    "report generated successfully",
    "permission verified",
    "session reopened after timeout",
    "login from new ip verified",
    "brief database connection retry success",
    "failed login attempt then success",
    "minor timeout then reconnected",
    "temporary service delay resolved",
    "retry succeeded after one attempt",
    "connection reset then re-established",
    "slow query completed within limit",
    "elevated cpu usage returned to normal",
]

NORMAL_IPS = [
    f"{random.randint(10,250)}.{random.randint(1,250)}.{random.randint(1,250)}.{random.randint(1,250)}"
    for _ in range(200)
]

def random_time(hour_start=6, hour_end=22):
    h = random.randint(hour_start, hour_end)
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return f"{h:02d}:{m:02d}:{s:02d}"

def random_date(start_year=2025, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 3, 31)
    delta = end - start
    rand_days = random.randint(0, delta.days)
    d = start + timedelta(days=rand_days)
    return d.strftime("%d-%m-%Y")

def generate_normal_logs(n=10000):
    rows = []
    for _ in range(n):
        rows.append({
            "time": random_time(6, 22),
            "date": random_date(),
            "ip": random.choice(NORMAL_IPS),
            "log": random.choice(NORMAL_LOGS)
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    df = generate_normal_logs(10000)
    out = os.path.join(os.path.dirname(__file__), "..", "logs", "training_data.csv")
    df.to_csv(out, index=False)
    print(f"✅ Generated {len(df)} training logs → logs/training_data.csv")
    print(df.head(5).to_string())
