"""
db_service.py
----------------
Read-only database access functions used by the Flask API. Keeps SQL out
of the route handlers.
"""

import json
import sqlite3
import sys
import os

from flask import g, has_app_context

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DB_PATH

_offline_conn = None

def get_connection():
    if has_app_context():
        if "db" not in g:
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
        return g.db
    else:
        global _offline_conn
        if _offline_conn is None:
            _offline_conn = sqlite3.connect(DB_PATH)
            _offline_conn.row_factory = sqlite3.Row
        return _offline_conn

def close_connection(e=None):
    if has_app_context():
        db = g.pop("db", None)
        if db is not None:
            db.close()



def list_areas():
    conn = get_connection()
    rows = conn.execute(
        "SELECT area_id, area_name, state_name, centroid_lat, centroid_lon, data_mode FROM areas ORDER BY area_id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_area(area_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM areas WHERE area_id = ?", (area_id,)).fetchone()
    return dict(row) if row else None


def get_latest_prediction(area_id: int):
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM predictions WHERE area_id = ?
           ORDER BY prediction_timestamp DESC LIMIT 1""",
        (area_id,),
    ).fetchone()
    return dict(row) if row else None


def get_explanation_for_prediction(prediction_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM explanations WHERE prediction_id = ?", (prediction_id,)
    ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["shap_json"] = json.loads(result["shap_json"]) if result["shap_json"] else None
    return result


def get_all_areas_with_predictions():
    """Used by the /map endpoint -- one row per area with its latest prediction, if any."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.area_id, a.area_name, a.state_name, a.centroid_lat, a.centroid_lon, a.data_mode,
                  p.predicted_score, p.predicted_category, p.model_name
           FROM areas a
           LEFT JOIN predictions p
             ON p.area_id = a.area_id
             AND p.prediction_id = (
                 SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.area_id = a.area_id
             )
           ORDER BY a.area_id"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_model_performance():
    conn = get_connection()
    rows = conn.execute(
        """SELECT model_name, mae, rmse, r2, trained_on, eval_timestamp
           FROM model_results ORDER BY eval_timestamp DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_user_by_email(email: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(name: str, email: str, password_hash: str, role: str = 'PUBLIC'):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role, status) VALUES (?, ?, ?, ?, 'active')",
            (name, email, password_hash, role)
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        user_id = None
    return user_id


def log_audit_action(user_id: int, action: str, details: str = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    conn.commit()

def get_data_quality_stats():
    conn = get_connection()
    total_areas = conn.execute("SELECT COUNT(*) FROM areas").fetchone()[0]
    features_count = conn.execute("SELECT COUNT(*) FROM geospatial_features").fetchone()[0]
    missing_nightlight = conn.execute("SELECT COUNT(*) FROM geospatial_features WHERE nightlight_mean IS NULL").fetchone()[0]
    missing_ndvi = conn.execute("SELECT COUNT(*) FROM geospatial_features WHERE ndvi_mean IS NULL").fetchone()[0]
    missing_builtup = conn.execute("SELECT COUNT(*) FROM geospatial_features WHERE builtup_pct IS NULL").fetchone()[0]
    sat_count = conn.execute("SELECT COUNT(*) FROM satellite_data").fetchone()[0]
    gt_count = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    
    return {
        "total_districts": total_areas,
        "geospatial_features": {
            "total_records": features_count,
            "missing_nightlight": missing_nightlight,
            "missing_ndvi": missing_ndvi,
            "missing_builtup": missing_builtup
        },
        "satellite_data": {
            "total_records": sat_count,
            "missing": total_areas - sat_count
        },
        "ground_truth": {
            "total_records": gt_count,
            "missing": total_areas - gt_count
        }
    }
