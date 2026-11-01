"""
scripts/create_admin.py
-----------------------
Creates an administrator account in the GeoPoverty AI database.
Reads admin credentials from environment variables:
  - ADMIN_NAME
  - ADMIN_EMAIL
  - ADMIN_PASSWORD
If any of these environment variables are missing, prompts the operator interactively.
No hardcoded or default credentials exist anywhere in this script.
"""

import getpass
import os
import re
import sys

from werkzeug.security import generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services import db_service


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def create_admin_user(name: str = None, email: str = None, password: str = None) -> int:
    """Creates an admin user, validating fields and hashing the password.
    Returns the new user_id, or raises ValueError on invalid input or failure."""
    name = (name or os.environ.get("ADMIN_NAME") or "").strip()
    email = (email or os.environ.get("ADMIN_EMAIL") or "").strip()
    password = (password or os.environ.get("ADMIN_PASSWORD") or "").strip()

    if not name:
        name = input("Enter admin full name: ").strip()
    if not name:
        raise ValueError("Admin name cannot be empty.")

    if not email:
        email = input("Enter admin email address: ").strip()
    if not email or not is_valid_email(email):
        raise ValueError("A valid email address is required.")

    if not password:
        try:
            password = getpass.getpass("Enter admin password (min 8 chars): ").strip()
        except (io.UnsupportedOperation, Exception):
            password = input("Enter admin password (min 8 chars): ").strip()

    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    existing = db_service.get_user_by_email(email)
    if existing:
        raise ValueError(f"User with email '{email}' already exists.")

    password_hash = generate_password_hash(password)
    user_id = db_service.create_user(name, email, password_hash, role="ADMIN")
    if not user_id:
        raise RuntimeError(f"Database error: failed to create admin user '{email}'.")

    db_service.log_audit_action(user_id, "ADMIN_CREATED", f"Admin user '{email}' created.")
    return user_id


def main():
    try:
        user_id = create_admin_user()
        print(f"Success: Admin user created successfully with ID {user_id}.")
        sys.exit(0)
    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

