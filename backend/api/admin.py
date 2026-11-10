import sys
import os
from flask import Blueprint, jsonify, request, session

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service
from backend.api.auth import roles_required

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

VALID_ROLES = ("PUBLIC", "RESEARCHER", "ADMIN")


# NOTE: these handlers must NOT call conn.close(). The connection is shared for the
# request (flask.g) and closed by the app teardown; closing it early broke the
# audit-log write that follows.

@admin_bp.route("/dashboard-stats", methods=["GET"])
@roles_required("ADMIN")
def dashboard_stats():
    conn = db_service.get_connection()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    researchers = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'RESEARCHER'").fetchone()[0]
    active_users = conn.execute("SELECT COUNT(*) FROM users WHERE status = 'active'").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM users WHERE requested_role = 'RESEARCHER' AND role = 'PUBLIC'"
    ).fetchone()[0]
    total_districts = conn.execute("SELECT COUNT(*) FROM areas").fetchone()[0]
    datasets = conn.execute("SELECT COUNT(*) FROM satellite_data").fetchone()[0]

    return jsonify({
        "total_users": total_users,
        "researchers": researchers,
        "active_users": active_users,
        "pending_requests": pending,
        "total_districts": total_districts,
        "available_datasets": datasets
    })


@admin_bp.route("/users", methods=["GET"])
@roles_required("ADMIN")
def get_users():
    conn = db_service.get_connection()
    rows = conn.execute(
        "SELECT user_id, name, email, role, status, requested_role, created_at FROM users"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@admin_bp.route("/pending-researchers", methods=["GET"])
@roles_required("ADMIN")
def pending_researchers():
    conn = db_service.get_connection()
    rows = conn.execute(
        "SELECT user_id, name, email, created_at FROM users "
        "WHERE requested_role = 'RESEARCHER' AND role = 'PUBLIC' AND status = 'active' "
        "ORDER BY created_at"
    ).fetchall()
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
    return jsonify([dict(r) for r in rows])


def _guard_admin_change(target, new_role=None, new_status=None):
    """Return an error response if the change would remove the last active admin
    or if an admin tries to lock themselves out."""
    if target["role"] == "ADMIN" and target["status"] == "active":
        demoting = new_role is not None and new_role != "ADMIN"
        deactivating = new_status is not None and new_status != "active"
        if demoting or deactivating:
            if target["user_id"] == session["user_id"]:
                return jsonify({"error": "You cannot demote or deactivate your own account"}), 400
            if db_service.count_active_admins() <= 1:
                return jsonify({"error": "Cannot remove the last active administrator"}), 400
    return None


@admin_bp.route("/users/<int:user_id>/role", methods=["PUT"])
@roles_required("ADMIN")
def update_user_role(user_id):
    data = request.get_json(silent=True) or {}
    new_role = data.get("role")
    if new_role not in VALID_ROLES:
        return jsonify({"error": "Invalid role"}), 400

    target = db_service.get_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    err = _guard_admin_change(target, new_role=new_role)
    if err:
        return err

    # A manual role change settles any pending researcher request.
    if not db_service.update_user_fields(user_id, role=new_role, requested_role=None):
        return jsonify({"error": "Could not update role"}), 500
    db_service.log_audit_action(session["user_id"], "UPDATE_ROLE",
                                f"Changed role of user {user_id} to {new_role}")
    return jsonify({"message": "Role updated successfully"})


@admin_bp.route("/users/<int:user_id>/status", methods=["PUT"])
@roles_required("ADMIN")
def update_user_status(user_id):
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    if new_status not in ("active", "inactive"):
        return jsonify({"error": "Invalid status"}), 400

    target = db_service.get_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    err = _guard_admin_change(target, new_status=new_status)
    if err:
        return err

    if not db_service.update_user_fields(user_id, status=new_status):
        return jsonify({"error": "Could not update status"}), 500
    db_service.log_audit_action(session["user_id"], "UPDATE_STATUS",
                                f"Set status of user {user_id} to {new_status}")
    return jsonify({"message": "Status updated successfully"})


def _resolve_request(user_id, approve: bool):
    target = db_service.get_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    if target.get("requested_role") != "RESEARCHER" or target["role"] != "PUBLIC":
        return jsonify({"error": "No pending researcher request for this user"}), 400

    ok = db_service.update_user_fields(
        user_id,
        **({"role": "RESEARCHER", "requested_role": None} if approve else {"requested_role": None}),
    )
    if not ok:
        return jsonify({"error": "Could not update request"}), 500
    db_service.log_audit_action(
        session["user_id"], "APPROVE_RESEARCHER" if approve else "REJECT_RESEARCHER",
        f"{'Approved' if approve else 'Rejected'} researcher request of user {user_id}")
    return jsonify({"message": "Researcher approved" if approve else "Request rejected"})


@admin_bp.route("/users/<int:user_id>/approve-researcher", methods=["POST"])
@roles_required("ADMIN")
def approve_researcher(user_id):
    return _resolve_request(user_id, True)


@admin_bp.route("/users/<int:user_id>/reject-researcher", methods=["POST"])
@roles_required("ADMIN")
def reject_researcher(user_id):
    return _resolve_request(user_id, False)
