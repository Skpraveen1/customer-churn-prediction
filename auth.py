import hashlib

# ── User store ────────────────────────────────────────────────────────────────
# In production replace this with a real DB (SQLite / PostgreSQL etc.)
# Passwords are stored as SHA-256 hashes.
# To generate a hash: hashlib.sha256("yourpassword".encode()).hexdigest()

USERS = {
    # Employees
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
    # Managers
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


def authenticate(user_id: str, password: str):
    """
    Returns (True, user_dict) on success, (False, None) on failure.
    """
    user = USERS.get(user_id.strip().upper())
    if not user:
        return False, None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash == user["password_hash"]:
        return True, {"id": user_id.upper(), **user}
    return False, None