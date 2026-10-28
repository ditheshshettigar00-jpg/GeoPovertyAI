import sys
import os
from flask import Blueprint, jsonify, request

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service
from backend.api.auth import roles_required

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

@admin_bp.route("/dashboard-stats", methods=["GET"])
@roles_required("ADMIN")
def dashboard_stats():
    conn = db_service.get_connection()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    researchers = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'RESEARCHER'").fetchone()[0]
    active_users = conn.execute("SELECT COUNT(*) FROM users WHERE status = 'active'").fetchone()[0]
    total_districts = conn.execute("SELECT COUNT(*) FROM areas").fetchone()[0]
    datasets = conn.execute("SELECT COUNT(*) FROM satellite_data").fetchone()[0]
    conn.close()
    
    return jsonify({
        "total_users": total_users,
        "researchers": researchers,
        "active_users": active_users,
        "total_districts": total_districts,
        "available_datasets": datasets
    })

@admin_bp.route("/users", methods=["GET"])
@roles_required("ADMIN")
def get_users():
    conn = db_service.get_connection()
    rows = conn.execute("SELECT user_id, name, email, role, status, created_at FROM users").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@admin_bp.route("/audit-logs", methods=["GET"])
@roles_required("ADMIN")
def get_audit_logs():
    conn = db_service.get_connection()
    rows = conn.execute("""
        SELECT a.id, a.action, a.timestamp, a.details, u.name, u.email 
        FROM audit_logs a
        LEFT JOIN users u ON a.user_id = u.user_id
        ORDER BY a.timestamp DESC LIMIT 100
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@admin_bp.route("/users/<int:user_id>/role", methods=["PUT"])
@roles_required("ADMIN")
def update_user_role(user_id):
    data = request.get_json()
    new_role = data.get("role")
    if new_role not in ["PUBLIC", "RESEARCHER", "ADMIN"]:
        return jsonify({"error": "Invalid role"}), 400
        
    conn = db_service.get_connection()
    conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (new_role, user_id))
    conn.commit()
    conn.close()
    
    from flask import session
    db_service.log_audit_action(session["user_id"], "UPDATE_ROLE", f"Changed role of user {user_id} to {new_role}")
    
    return jsonify({"message": "Role updated successfully"})
