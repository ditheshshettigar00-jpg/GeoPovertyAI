"""
tests/test_security_hotspots.py
---------------------------------
Tests covering:
- CORS configuration
- FLASK_SECRET_KEY requirement and dev key enforcement
- FLASK_DEBUG default
- Session cookie hardening (HttpOnly, SameSite=Lax)
- Password minimum length enforcement (8 chars)
- In-memory login rate limiter (5 failed attempts per 15 min per email & IP)
- scripts/create_admin.py admin creation and absence of default credentials
- Input validation and training range warnings in /api/scenario and /api/predict-live
- DBSCAN hotspot clustering using Haversine metric on radians and 'High' category filter
"""

import os
import sys
import time
import uuid
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app
from backend.services import db_service, hotspot_service, model_service
import config


# ---------------------------------------------------------------------------
# 1. CORS Configuration
# ---------------------------------------------------------------------------

def test_cors_default_allows_no_external_origins(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    client = app.test_client()
    res = client.get("/api/areas", headers={"Origin": "http://malicious-site.com"})
    assert res.headers.get("Access-Control-Allow-Origin") is None


def test_cors_allows_configured_origins_only(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://trusted-portal.gov, https://dashboard.org")
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    client = app.test_client()
    
    # Allowed origin 1
    res1 = client.get("/api/areas", headers={"Origin": "https://trusted-portal.gov"})
    assert res1.headers.get("Access-Control-Allow-Origin") == "https://trusted-portal.gov"
    
    # Allowed origin 2
    res2 = client.get("/api/areas", headers={"Origin": "https://dashboard.org"})
    assert res2.headers.get("Access-Control-Allow-Origin") == "https://dashboard.org"
    
    # Untrusted origin
    res3 = client.get("/api/areas", headers={"Origin": "http://evil.com"})
    assert res3.headers.get("Access-Control-Allow-Origin") is None


# ---------------------------------------------------------------------------
# 2. Secret Key Enforcement
# ---------------------------------------------------------------------------

def test_secret_key_fails_without_key_in_non_dev_env(monkeypatch):
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.setenv("FLASK_ENV", "production")
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app()


def test_secret_key_fails_without_key_and_unset_env(monkeypatch):
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app()


def test_secret_key_dev_key_allowed_in_development(monkeypatch):
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    assert app.secret_key == "dev_secret_key_geopoverty"


def test_secret_key_custom_key_accepted(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "super_secure_production_secret_key_12345")
    monkeypatch.setenv("FLASK_ENV", "production")
    app = create_app()
    assert app.secret_key == "super_secure_production_secret_key_12345"


# ---------------------------------------------------------------------------
# 3. FLASK_DEBUG Default
# ---------------------------------------------------------------------------

def test_flask_debug_defaults_to_false(monkeypatch):
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    assert app.config["DEBUG"] is False


def test_flask_debug_can_be_enabled_via_env(monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", "True")
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    assert app.config["DEBUG"] is True


# ---------------------------------------------------------------------------
# 4. Session Cookies (HttpOnly and SameSite=Lax)
# ---------------------------------------------------------------------------

def test_session_cookies_httponly_and_samesite_lax(client):
    unique_email = f"cookie_test_{int(time.time() * 1000)}@example.com"
    reg = client.post("/api/auth/register", json={
        "name": "Cookie User",
        "email": unique_email,
        "password": "Password123!"
    })
    assert reg.status_code == 201

    login_res = client.post("/api/auth/login", json={
        "email": unique_email,
        "password": "Password123!"
    })
    assert login_res.status_code == 200
    cookie_header = login_res.headers.get("Set-Cookie")
    assert cookie_header is not None
    assert "HttpOnly" in cookie_header
    assert "SameSite=Lax" in cookie_header


# ---------------------------------------------------------------------------
# 5. Password Minimum Length 8
# ---------------------------------------------------------------------------

def test_password_length_under_8_rejected(client):
    res = client.post("/api/auth/register", json={
        "name": "Short Pass User",
        "email": f"short_{int(time.time() * 1000)}@example.com",
        "password": "short7"
    })
    assert res.status_code == 400
    assert "at least 8 characters" in res.get_json()["error"]


def test_password_length_8_or_more_accepted(client):
    res = client.post("/api/auth/register", json={
        "name": "Valid Pass User",
        "email": f"valid_{int(time.time() * 1000)}@example.com",
        "password": "eight888"
    })
    assert res.status_code == 201


# ---------------------------------------------------------------------------
# 6. Login Rate Limiting (5 failed attempts per 15 min per email & IP)
# ---------------------------------------------------------------------------

def test_login_rate_limiting_triggers_after_5_failures(client):
    from backend.api.auth import FAILED_LOGIN_ATTEMPTS
    FAILED_LOGIN_ATTEMPTS.clear()

    target_email = f"ratelimit_{int(time.time() * 1000)}@example.com"
    client.post("/api/auth/register", json={
        "name": "Rate Limit Target",
        "email": target_email,
        "password": "CorrectPassword123"
    })

    # 5 failed attempts
    for i in range(5):
        res = client.post("/api/auth/login", json={
            "email": target_email,
            "password": f"WrongPass_{i}"
        }, environ_base={"REMOTE_ADDR": "192.168.1.100"})
        assert res.status_code == 401, f"Attempt {i+1} expected 401, got {res.status_code}"

    # 6th attempt should be blocked by rate limit
    res6 = client.post("/api/auth/login", json={
        "email": target_email,
        "password": "WrongPass_6"
    }, environ_base={"REMOTE_ADDR": "192.168.1.100"})
    assert res6.status_code == 429
    assert "Too many failed login attempts" in res6.get_json()["error"]

    # Even with correct password, blocked until cooldown
    res_correct_blocked = client.post("/api/auth/login", json={
        "email": target_email,
        "password": "CorrectPassword123"
    }, environ_base={"REMOTE_ADDR": "192.168.1.100"})
    assert res_correct_blocked.status_code == 429

    # Different IP should not be blocked for this email
    res_diff_ip = client.post("/api/auth/login", json={
        "email": target_email,
        "password": "CorrectPassword123"
    }, environ_base={"REMOTE_ADDR": "10.0.0.50"})
    assert res_diff_ip.status_code == 200

    FAILED_LOGIN_ATTEMPTS.clear()


# ---------------------------------------------------------------------------
# 7. scripts/create_admin.py
# ---------------------------------------------------------------------------

def test_create_admin_script_success(monkeypatch):
    from scripts.create_admin import create_admin_user
    admin_email = f"admin_test_{int(time.time() * 1000)}@geopoverty.ai"
    monkeypatch.setenv("ADMIN_NAME", "System Administrator")
    monkeypatch.setenv("ADMIN_EMAIL", admin_email)
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecureAdminPass999")

    user_id = create_admin_user()
    assert user_id is not None

    user = db_service.get_user_by_email(admin_email)
    assert user is not None
    assert user["role"] == "ADMIN"
    assert user["name"] == "System Administrator"


def test_create_admin_script_rejects_short_password(monkeypatch):
    from scripts.create_admin import create_admin_user
    monkeypatch.setenv("ADMIN_NAME", "Short Pass Admin")
    monkeypatch.setenv("ADMIN_EMAIL", f"admin_{int(time.time() * 1000)}@test.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "short")

    with pytest.raises(ValueError, match="at least 8 characters"):
        create_admin_user()


def test_create_admin_script_has_no_default_credentials():
    admin_script_path = os.path.join(config.BASE_DIR, "scripts", "create_admin.py")
    with open(admin_script_path, "r", encoding="utf-8") as f:
        code = f.read()
    # Check that common hardcoded patterns are absent
    for banned in ["admin@admin.com", "admin123", "password123", "default_password", "DEFAULT_PASSWORD"]:
        assert banned not in code.lower(), f"Found banned default credential pattern: {banned}"


# ---------------------------------------------------------------------------
# 8. /api/scenario and /api/predict-live Numeric Validation and Range Warnings
# ---------------------------------------------------------------------------

def test_scenario_rejects_non_numeric_values(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    res = client.post("/api/scenario", json={
        "area_id": area_id,
        "modified_features": {"builtup_pct": "not_a_number"}
    })
    assert res.status_code == 400
    assert "numeric" in res.get_json()["error"].lower()


def test_scenario_warns_on_out_of_range_values(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    res = client.post("/api/scenario", json={
        "area_id": area_id,
        "modified_features": {"builtup_pct": 500.0}  # Well beyond training max (~40.5)
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "warnings" in data
    assert len(data["warnings"]) > 0
    assert "outside the training range" in data["warnings"][0]


def test_predict_live_rejects_non_numeric_values(client):
    res = client.post("/api/predict-live", json={
        "nightlight_mean": "abc",
        "ndvi_mean": 0.35
    })
    assert res.status_code == 400
    assert "numeric" in res.get_json()["error"].lower()


def test_predict_live_rejects_boolean_values(client):
    res = client.post("/api/predict-live", json={
        "nightlight_mean": True
    })
    assert res.status_code == 400
    assert "numeric" in res.get_json()["error"].lower()


def test_predict_live_warns_on_out_of_range_values(client):
    res = client.post("/api/predict-live", json={
        "nightlight_mean": 999.0,  # Training max is ~45.05
        "ndvi_mean": 0.40,
        "builtup_pct": 10.0,
        "population_density": 200.0
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "warnings" in data
    assert len(data["warnings"]) > 0
    assert any("nightlight_mean" in w and "outside the training range" in w for w in data["warnings"])


# ---------------------------------------------------------------------------
# 9. DBSCAN Hotspots with Haversine Metric on Radians and 'High' Category
# ---------------------------------------------------------------------------

def test_hotspot_constants_in_config():
    assert hasattr(config, "EARTH_RADIUS_KM")
    assert np.isclose(config.EARTH_RADIUS_KM, 6371.0)
    assert hasattr(config, "HOTSPOT_DISTANCE_KM")
    assert np.isclose(config.HOTSPOT_DISTANCE_KM, 150.0)
    assert hasattr(config, "HOTSPOT_EPS")
    assert np.isclose(config.HOTSPOT_EPS, 150.0 / 6371.0, atol=1e-6)
    assert config.HOTSPOT_MIN_SAMPLES == 3


def test_hotspot_service_identifies_clusters_for_high_category_districts(client):
    clusters = hotspot_service.identify_hotspots()
    assert isinstance(clusters, list)
    if clusters:
        for cluster in clusters:
            assert "cluster_id" in cluster
            assert "avg_score" in cluster
            assert "districts" in cluster
            assert "states" in cluster
            assert "center_lat" in cluster
            assert "center_lon" in cluster
            assert len(cluster["districts"]) >= config.HOTSPOT_MIN_SAMPLES
            # Confirm every district in the hotspot is categorized as 'High'
            for d in cluster["districts"]:
                assert d["predicted_category"] == "High"

    # Also verify the API endpoint returns the same result
    api_res = client.get("/api/hotspots")
    assert api_res.status_code == 200
    api_clusters = api_res.get_json()
    assert len(api_clusters) == len(clusters)


# ---------------------------------------------------------------------------
# 10. Security Hardening & Edge Cases (Task 3)
# ---------------------------------------------------------------------------

def test_login_rate_limit_not_bypassed_by_spoofed_x_forwarded_for(client, monkeypatch):
    """Verifies that an attacker cannot bypass the 5-attempt login rate limit by
    spoofing X-Forwarded-For headers when TRUST_PROXY is not enabled."""
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    test_email = f"ratelimit_spoof_{uuid.uuid4().hex[:8]}@example.com"

    # Send 5 failed login attempts with differing spoofed client IPs
    for i in range(5):
        res = client.post(
            "/api/auth/login",
            json={"email": test_email, "password": "WrongPassword1!"},
            headers={"X-Forwarded-For": f"198.51.100.{i+1}"}
        )
        assert res.status_code == 401

    # 6th attempt with yet another spoofed IP must be rate-limited (HTTP 429)
    res6 = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": "WrongPassword1!"},
        headers={"X-Forwarded-For": "203.0.113.99"}
    )
    assert res6.status_code == 429
    assert "Too many failed login attempts" in res6.get_json()["error"]


def test_login_rate_limit_respects_proxy_when_trust_proxy_enabled(monkeypatch):
    """Verifies _get_client_ip returns X-Forwarded-For when TRUST_PROXY=1."""
    from backend.api.auth import _get_client_ip
    from flask import Flask

    app = Flask(__name__)
    with app.test_request_context("/", headers={"X-Forwarded-For": "203.0.113.50, 10.0.0.1"}):
        # Default: untrusted -> uses remote_addr
        monkeypatch.delenv("TRUST_PROXY", raising=False)
        assert _get_client_ip() != "203.0.113.50"

        # Explicit trust: TRUST_PROXY=1 -> uses first IP from header
        monkeypatch.setenv("TRUST_PROXY", "1")
        assert _get_client_ip() == "203.0.113.50"


def test_session_cookie_secure_in_production(monkeypatch):
    """Verifies SESSION_COOKIE_SECURE is True when FLASK_ENV != 'development'."""
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test_production_secret_key_1234567890")
    from backend.app import create_app
    prod_app = create_app()
    assert prod_app.config["SESSION_COOKIE_SECURE"] is True
    assert prod_app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert prod_app.config["SESSION_COOKIE_SAMESITE"] == "Lax"


def test_path_traversal_on_images_rejected(client):
    """Verifies that directory traversal payloads on satellite and gradcam routes fail safely."""
    # Routing rejection of non-integer paths
    assert client.get("/api/satellite-image/..%2F..%2Fconfig.py").status_code == 404
    assert client.get("/api/gradcam/..%2F..%2Fconfig.py").status_code == 404

    # Non-existent integer area_id returns 404 safely
    assert client.get("/api/satellite-image/999999").status_code == 404

    # Grad-CAM endpoint safely deactivated for all districts
    gradcam_res = client.get("/api/gradcam/1")
    assert gradcam_res.status_code == 404
    assert "Grad-CAM not available" in gradcam_res.get_json()["error"]


def test_database_parameterized_queries_prevent_sql_injection():
    """Verifies parameterized query safety against SQL injection payloads."""
    from backend.services import db_service
    sqli_payload = "' OR '1'='1' --"

    # User lookup
    assert db_service.get_user_by_email(sqli_payload) is None

    # Area lookup
    assert db_service.get_area(sqli_payload) is None

    # Prediction lookup
    assert db_service.get_latest_prediction(sqli_payload) is None


def test_jinja_templates_autoescape_user_supplied_content(anon_client):
    """Verifies user-supplied content (e.g. name with HTML/JS tags) is auto-escaped by Jinja."""
    xss_name = "Attacker <script>alert('xss')</script>"
    xss_email = f"xss_{uuid.uuid4().hex[:8]}@example.com"

    reg = anon_client.post("/api/auth/register", json={
        "name": xss_name,
        "email": xss_email,
        "password": "Password123!"
    })
    assert reg.status_code == 201

    login = anon_client.post("/api/auth/login", json={
        "email": xss_email,
        "password": "Password123!"
    })
    assert login.status_code == 200

    # View map page which renders current_user.name in the navbar
    map_page = anon_client.get("/map")
    assert map_page.status_code == 200
    html = map_page.get_data(as_text=True)

    # Raw script tags must NOT appear; must be HTML escaped
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html


