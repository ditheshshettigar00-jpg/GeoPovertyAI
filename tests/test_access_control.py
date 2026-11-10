"""Role-based access control: PUBLIC / RESEARCHER / ADMIN, researcher approval, redirects."""
import uuid

import pytest

from conftest import make_user, login_session, _new_client

PUBLIC_PAGES = ["/", "/about", "/login", "/register", "/map", "/district/1"]
PUBLIC_APIS = ["/api/areas", "/api/map", "/api/predict/1", "/api/indicators/1",
               "/api/area-profile/1"]

RESEARCHER_PAGES = ["/overview", "/prediction", "/explanation", "/performance", "/report/1",
                    "/compare", "/matchmaker", "/hidden-gaps", "/hotspots", "/data-quality",
                    "/whatif", "/scenarios", "/scenario-explorer"]
RESEARCHER_GET_APIS = ["/api/explain/1", "/api/model-performance", "/api/scenario-presets",
                       "/api/scenario-targets", "/api/area-features/1", "/api/hotspots",
                       "/api/data-quality", "/api/similar-districts/1", "/api/residual-atlas",
                       "/api/reports/district-data/1", "/api/reports/district/1"]
RESEARCHER_POST_APIS = ["/api/compare", "/api/scenario", "/api/predict-live"]

ADMIN_PAGES = ["/admin"]
ADMIN_GET_APIS = ["/api/admin/dashboard-stats", "/api/admin/users",
                  "/api/admin/pending-researchers", "/api/admin/audit-logs"]


def as_role(role, **kw):
    c = _new_client()
    uid = make_user(role, **kw)
    login_session(c, uid)
    return c, uid


@pytest.fixture
def public_c():
    return as_role("PUBLIC")[0]


@pytest.fixture
def researcher_c():
    return as_role("RESEARCHER")[0]


@pytest.fixture
def admin_c():
    return as_role("ADMIN")[0]


# ---------------------------------------------------------------- anonymous
@pytest.mark.parametrize("path", PUBLIC_PAGES + PUBLIC_APIS)
def test_anonymous_can_open_public_routes(anon_client, path):
    assert anon_client.get(path).status_code == 200


@pytest.mark.parametrize("path", RESEARCHER_PAGES + ADMIN_PAGES)
def test_anonymous_redirected_from_protected_pages(anon_client, path):
    r = anon_client.get(path)
    assert r.status_code == 302
    assert r.headers["Location"].startswith("/login?next=")


@pytest.mark.parametrize("path", RESEARCHER_GET_APIS + ADMIN_GET_APIS)
def test_anonymous_gets_401_on_protected_get_apis(anon_client, path):
    assert anon_client.get(path).status_code == 401


@pytest.mark.parametrize("path", RESEARCHER_POST_APIS)
def test_anonymous_gets_401_on_protected_post_apis(anon_client, path):
    assert anon_client.post(path, json={}).status_code == 401


# ------------------------------------------------------------------- public
@pytest.mark.parametrize("path", [p for p in PUBLIC_PAGES if p != "/login"] + PUBLIC_APIS)
def test_public_user_can_open_public_routes(public_c, path):
    assert public_c.get(path).status_code == 200


def test_logged_in_user_visiting_login_goes_to_role_home(public_c, researcher_c, admin_c):
    assert public_c.get("/login").headers["Location"] == "/map"
    assert researcher_c.get("/login").headers["Location"] == "/overview"
    assert admin_c.get("/login").headers["Location"] == "/admin"


@pytest.mark.parametrize("path", RESEARCHER_PAGES + ADMIN_PAGES)
def test_public_user_gets_403_page(public_c, path):
    assert public_c.get(path).status_code == 403


@pytest.mark.parametrize("path", RESEARCHER_GET_APIS + ADMIN_GET_APIS)
def test_public_user_gets_403_on_get_apis(public_c, path):
    r = public_c.get(path)
    assert r.status_code == 403 and "error" in r.get_json()


@pytest.mark.parametrize("path", RESEARCHER_POST_APIS)
def test_public_user_gets_403_on_post_apis(public_c, path):
    assert public_c.post(path, json={}).status_code == 403


# --------------------------------------------------------------- researcher
@pytest.mark.parametrize("path", RESEARCHER_PAGES)
def test_researcher_can_open_researcher_pages(researcher_c, path):
    assert researcher_c.get(path).status_code == 200


@pytest.mark.parametrize("path", RESEARCHER_GET_APIS)
def test_researcher_not_blocked_on_researcher_apis(researcher_c, path):
    # Not blocked by access control (404 for a missing district is fine, 401/403 is not).
    assert researcher_c.get(path).status_code not in (401, 403)


@pytest.mark.parametrize("path", RESEARCHER_POST_APIS)
def test_researcher_not_blocked_on_post_apis(researcher_c, path):
    assert researcher_c.post(path, json={}).status_code not in (401, 403)


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_researcher_cannot_open_admin_pages(researcher_c, path):
    assert researcher_c.get(path).status_code == 403


@pytest.mark.parametrize("path", ADMIN_GET_APIS)
def test_researcher_cannot_use_admin_apis(researcher_c, path):
    assert researcher_c.get(path).status_code == 403


# -------------------------------------------------------------------- admin
@pytest.mark.parametrize("path", RESEARCHER_PAGES + ADMIN_PAGES)
def test_admin_can_open_all_pages(admin_c, path):
    assert admin_c.get(path).status_code == 200


@pytest.mark.parametrize("path", ADMIN_GET_APIS)
def test_admin_can_use_admin_apis(admin_c, path):
    assert admin_c.get(path).status_code == 200


# ----------------------------------------------- inactive / role comes from DB
def test_inactive_user_is_blocked_even_with_session(anon_client):
    uid = make_user("RESEARCHER", status="inactive")
    login_session(anon_client, uid)
    assert anon_client.get("/api/compare", json={}).status_code in (403, 405)
    assert anon_client.get("/hotspots").status_code == 403


def test_role_is_read_from_database_not_session(anon_client):
    uid = make_user("PUBLIC")
    with anon_client.session_transaction() as sess:
        sess["user_id"] = uid
        sess["role"] = "ADMIN"          # forged client-side value must be ignored
    assert anon_client.get("/admin").status_code == 403
    assert anon_client.get("/api/admin/users").status_code == 403


# ------------------------------------------------------------- registration
def _register(c, **extra):
    email = f"reg_{uuid.uuid4().hex[:12]}@example.com"
    body = {"name": "Reg User", "email": email, "password": "password123"}
    body.update(extra)
    return email, c.post("/api/auth/register", json=body)


def _login(c, email, **extra):
    return c.post("/api/auth/login", json={"email": email, "password": "password123", **extra})


@pytest.mark.parametrize("injected", [{"role": "ADMIN"}, {"role": "RESEARCHER"},
                                      {"requested_role": "ADMIN"}, {"status": "active", "role": "ADMIN"}])
def test_register_ignores_client_supplied_role(anon_client, injected):
    from backend.services import db_service
    email, r = _register(anon_client, **injected)
    assert r.status_code == 201
    u = db_service.get_user_by_email(email)
    assert u["role"] == "PUBLIC"
    assert u["requested_role"] is None


def test_register_researcher_request_is_pending_public(anon_client):
    from backend.services import db_service
    email, r = _register(anon_client, request_researcher=True)
    assert r.status_code == 201 and r.get_json()["researcher_pending"] is True
    u = db_service.get_user_by_email(email)
    assert u["role"] == "PUBLIC" and u["requested_role"] == "RESEARCHER"


# ------------------------------------------------------- login redirect / next
def test_login_redirects_by_role():
    from backend.services import db_service
    from werkzeug.security import generate_password_hash
    pw = generate_password_hash("password123")
    expected = {"PUBLIC": "/map", "RESEARCHER": "/overview", "ADMIN": "/admin"}
    for role, dest in expected.items():
        email = f"rd_{role.lower()}_{uuid.uuid4().hex[:10]}@example.com"
        assert db_service.create_user("RD", email, pw, role=role)
        c = _new_client()
        r = _login(c, email)
        assert r.status_code == 200 and r.get_json()["redirect"] == dest


@pytest.mark.parametrize("bad", ["//evil.com", "https://evil.com", "/\\evil.com", "javascript:alert(1)", "evil.com"])
def test_login_rejects_open_redirect(bad):
    from backend.services import db_service
    from werkzeug.security import generate_password_hash
    email = f"or_{uuid.uuid4().hex[:10]}@example.com"
    db_service.create_user("OR", email, generate_password_hash("password123"), role="PUBLIC")
    r = _login(_new_client(), email, next=bad)
    assert r.get_json()["redirect"] == "/map"


def test_login_honours_safe_next():
    from backend.services import db_service
    from werkzeug.security import generate_password_hash
    email = f"sn_{uuid.uuid4().hex[:10]}@example.com"
    db_service.create_user("SN", email, generate_password_hash("password123"), role="RESEARCHER")
    r = _login(_new_client(), email, next="/compare")
    assert r.get_json()["redirect"] == "/compare"


# ------------------------------------------------------ approval workflow
def test_pending_researcher_has_public_access_until_approved(admin_c):
    from backend.services import db_service
    c = _new_client()
    email, _ = _register(c, request_researcher=True)
    assert _login(c, email).status_code == 200
    assert c.get("/map").status_code == 200
    assert c.get("/hotspots").status_code == 403
    assert c.get("/api/hotspots").status_code == 403

    uid = db_service.get_user_by_email(email)["user_id"]
    pending = admin_c.get("/api/admin/pending-researchers").get_json()
    assert uid in [p["user_id"] for p in pending]

    assert admin_c.post(f"/api/admin/users/{uid}/approve-researcher").status_code == 200
    assert c.get("/hotspots").status_code == 200          # same session, now approved
    assert c.get("/api/hotspots").status_code == 200
    assert db_service.get_user_by_email(email)["requested_role"] is None


def test_reject_keeps_user_public(admin_c):
    from backend.services import db_service
    c = _new_client()
    email, _ = _register(c, request_researcher=True)
    uid = db_service.get_user_by_email(email)["user_id"]
    assert admin_c.post(f"/api/admin/users/{uid}/reject-researcher").status_code == 200
    u = db_service.get_user_by_email(email)
    assert u["role"] == "PUBLIC" and u["requested_role"] is None
    # nothing pending any more
    assert admin_c.post(f"/api/admin/users/{uid}/approve-researcher").status_code == 400


def test_non_admin_cannot_approve(public_c, researcher_c):
    uid = make_user("PUBLIC", requested_role="RESEARCHER")
    assert public_c.post(f"/api/admin/users/{uid}/approve-researcher").status_code == 403
    assert researcher_c.post(f"/api/admin/users/{uid}/approve-researcher").status_code == 403
    from backend.services import db_service
    assert db_service.get_user_by_id(uid)["role"] == "PUBLIC"


def test_admin_actions_are_audited(admin_c):
    uid = make_user("PUBLIC", requested_role="RESEARCHER")
    admin_c.post(f"/api/admin/users/{uid}/approve-researcher")
    logs = admin_c.get("/api/admin/audit-logs").get_json()
    assert any(l["action"] == "APPROVE_RESEARCHER" for l in logs)


# ------------------------------------------------------- admin safety rules
def _only_admin():
    """Leave exactly one active admin (the returned one); restore others afterwards."""
    from backend.services import db_service
    conn = db_service.get_connection()
    others = [r[0] for r in conn.execute(
        "SELECT user_id FROM users WHERE role='ADMIN' AND status='active'").fetchall()]
    c, uid = as_role("ADMIN")
    for o in others:
        db_service.update_user_fields(o, status="inactive")
    return c, uid, others


def _restore(others):
    from backend.services import db_service
    for o in others:
        db_service.update_user_fields(o, status="active")


def test_admin_cannot_demote_or_deactivate_self_or_last_admin():
    from backend.services import db_service
    c, uid, others = _only_admin()
    try:
        assert c.put(f"/api/admin/users/{uid}/role", json={"role": "PUBLIC"}).status_code == 400
        assert c.put(f"/api/admin/users/{uid}/status", json={"status": "inactive"}).status_code == 400
        assert db_service.get_user_by_id(uid)["role"] == "ADMIN"
        assert db_service.get_user_by_id(uid)["status"] == "active"
    finally:
        _restore(others)


def test_admin_cannot_remove_last_other_admin():
    from backend.services import db_service
    c, uid, others = _only_admin()
    try:
        # a second admin exists -> demoting it is allowed ...
        other = make_user("ADMIN")
        assert c.put(f"/api/admin/users/{other}/role", json={"role": "RESEARCHER"}).status_code == 200
        # ... but the only remaining admin (self) is protected
        assert c.put(f"/api/admin/users/{uid}/role", json={"role": "RESEARCHER"}).status_code == 400
    finally:
        _restore(others)


def test_admin_role_change_and_status_validation(admin_c):
    uid = make_user("PUBLIC")
    assert admin_c.put(f"/api/admin/users/{uid}/role", json={"role": "SUPERUSER"}).status_code == 400
    assert admin_c.put(f"/api/admin/users/{uid}/status", json={"status": "banned"}).status_code == 400
    assert admin_c.put("/api/admin/users/99999999/role", json={"role": "PUBLIC"}).status_code == 404
    assert admin_c.put(f"/api/admin/users/{uid}/role", json={"role": "RESEARCHER"}).status_code == 200
    assert admin_c.put(f"/api/admin/users/{uid}/status", json={"status": "inactive"}).status_code == 200


# --------------------------------------------------------------------- UI
def test_navbar_hides_researcher_links_from_public_and_anonymous(anon_client, public_c):
    for c in (anon_client, public_c):
        html = c.get("/map").get_data(as_text=True)
        assert "/hotspots" not in html and "/hidden-gaps" not in html and "/admin" not in html


def test_navbar_shows_links_for_researcher_and_admin_link_only_for_admin(researcher_c, admin_c):
    r_html = researcher_c.get("/overview").get_data(as_text=True)
    assert "/hotspots" in r_html and 'href="/admin"' not in r_html
    assert 'href="/admin"' in admin_c.get("/overview").get_data(as_text=True)


def test_navbar_escapes_user_name():
    from backend.services import db_service
    uid = db_service.create_user("<script>alert(1)</script>", f"xss_{uuid.uuid4().hex[:8]}@example.com", "x")
    c = _new_client()
    login_session(c, uid)
    html = c.get("/map").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html


def test_landing_page_has_role_cards(anon_client):
    html = anon_client.get("/").get_data(as_text=True)
    assert "/map" in html and "/login?as=researcher" in html and "/login?as=admin" in html


def test_register_is_not_a_way_to_become_admin_via_page(anon_client):
    assert anon_client.get("/register?as=admin").status_code == 200
    assert "request-researcher" in anon_client.get("/register").get_data(as_text=True)
