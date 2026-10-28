import functools
import sys
import os

from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return view(*args, **kwargs)
    return wrapped_view

def roles_required(*roles):
    def decorator(view):
        @functools.wraps(view)
        def wrapped_view(*args, **kwargs):
            if "user_id" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            
            user = db_service.get_user_by_id(session["user_id"])
            if not user or user["role"] not in roles:
                return jsonify({"error": "Forbidden: insufficient permissions"}), 403
                
            return view(*args, **kwargs)
        return wrapped_view
    return decorator

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not data or not data.get("email") or not data.get("password") or not data.get("name"):
        return jsonify({"error": "Missing required fields"}), 400
        
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", data["email"]):
        return jsonify({"error": "Invalid email format"}), 400
    if len(data["password"]) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
        
    user = db_service.get_user_by_email(data["email"])
    if user:
        return jsonify({"error": "User with this email already exists"}), 400
        
    password_hash = generate_password_hash(data["password"])
    # Default to PUBLIC unless admin creation is handled separately
    user_id = db_service.create_user(data["name"], data["email"], password_hash, role="PUBLIC")
    
    if not user_id:
        return jsonify({"error": "Could not create user"}), 500
        
    db_service.log_audit_action(user_id, "REGISTER", "User registered")
    
    return jsonify({"message": "User registered successfully"}), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing email or password"}), 400
        
    user = db_service.get_user_by_email(data["email"])
    if not user or not check_password_hash(user["password_hash"], data["password"]):
        return jsonify({"error": "Invalid email or password"}), 401
        
    if user["status"] != "active":
        return jsonify({"error": "Account is inactive"}), 403
        
    session.clear()
    session["user_id"] = user["user_id"]
    
    db_service.log_audit_action(user["user_id"], "LOGIN", "User logged in")
    
    return jsonify({
        "message": "Logged in successfully",
        "user": {
            "id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    })

@auth_bp.route("/logout", methods=["POST"])
def logout():
    user_id = session.get("user_id")
    if user_id:
        db_service.log_audit_action(user_id, "LOGOUT", "User logged out")
    session.clear()
    return jsonify({"message": "Logged out successfully"})

@auth_bp.route("/profile", methods=["GET"])
@login_required
def profile():
    user = db_service.get_user_by_id(session["user_id"])
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    return jsonify({
        "id": user["user_id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "status": user["status"],
        "created_at": user["created_at"]
    })
