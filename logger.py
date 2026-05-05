import json
import os
from datetime import datetime

LOG_FILE = "query_log.json"


def _load():
    """Load all logs from file."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save(logs):
    """Save logs to file."""
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def _clean(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v) for v in obj]
    elif hasattr(obj, 'item'):      # catches numpy int32, float32, bool_ etc.
        return obj.item()
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, int):
        return int(obj)
    return obj


def log_prediction(user_id: str, user_name: str, churn_prob, is_high_risk, inputs: dict):
    """Log a prediction made by an employee."""
    logs = _load()
    logs.append({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "time":         datetime.now().strftime("%H:%M"),
        "user_id":      user_id,
        "user_name":    user_name,
        "churn_prob":   round(float(churn_prob) * 100, 1),
        "is_high_risk": bool(is_high_risk),
        "inputs":       _clean(inputs),
    })
    _save(logs)


def get_all_logs():
    """Return all logs as a list of dicts."""
    return _load()


def get_today_logs():
    """Return only today's logs."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [l for l in _load() if l.get("date") == today]


# Public alias for external use
def save_logs(logs):
    _save(logs)