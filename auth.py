import hashlib
import json
import os

USERS_FILE = "users.json"

# ── Default users (only used if users.json doesn't exist yet) ─────────────────
DEFAULT_USERS = {
    "EMP-1001": {
        "password_hash": hashlib.sha256("emp1001pass".encode()).hexdigest(),
        "role": "employee",
        "name": "Priya Sharma",
    },
    "EMP-1002": {
        "password_hash": hashlib.sha256("emp1002pass".encode()).hexdigest(),
        "role": "employee",
        "name": "Rahul Mehta",
    },
    "MGR-2001": {
        "password_hash": hashlib.sha256("mgr2001pass".encode()).hexdigest(),
        "role": "manager",
        "name": "Anita Rao",
    },
    "MGR-2002": {
        "password_hash": hashlib.sha256("mgr2002pass".encode()).hexdigest(),
        "role": "manager",
        "name": "Vikram Singh",
    },
}


def _load_users() -> dict:
    """Load users from JSON file. If file doesn't exist, create it from defaults."""
    if not os.path.exists(USERS_FILE):
        _save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_USERS.copy()


def _save_users(users: dict):
    """Save users to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_all_users() -> dict:
    """Return all users from file."""
    return _load_users()


def add_user(user_id: str, name: str, role: str, password: str) -> bool:
    """Add a new user and save to file. Returns False if user_id already exists."""
    users = _load_users()
    uid = user_id.strip().upper()
    if uid in users:
        return False
    users[uid] = {
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "role": role,
        "name": name,
    }
    _save_users(users)
    return True


def delete_user(user_id: str) -> bool:
    """Delete a user by ID. Returns False if not found."""
    users = _load_users()
    uid = user_id.strip().upper()
    if uid not in users:
        return False
    del users[uid]
    _save_users(users)
    return True


def authenticate(user_id: str, password: str):
    """
    Returns (True, user_dict) on success, (False, None) on failure.
    Always reads from file so newly added users work immediately.
    """
    users = _load_users()
    user  = users.get(user_id.strip().upper())
    if not user:
        return False, None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash == user["password_hash"]:
        return True, {"id": user_id.strip().upper(), **user}
    return False, None
