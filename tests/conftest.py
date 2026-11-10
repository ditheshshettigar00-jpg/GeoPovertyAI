import os
import sys
import uuid
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set FLASK_ENV=development for testing suite as permitted by security guidelines
os.environ.setdefault("FLASK_ENV", "development")


def _artifacts_available():
    from config import DB_PATH, MODELS_DIR
    return os.path.exists(DB_PATH) and os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


def make_user(role="PUBLIC", requested_role=None, status="active"):
    """Create a throw-away user with a unique email and return its user_id.
    Never relies on a pre-existing user or a hard-coded id."""
    from backend.services import db_service
    email = f"test_{role.lower()}_{uuid.uuid4().hex[:12]}@example.com"
    uid = db_service.create_user("Test " + role.title(), email, "x", role=role,
                                 requested_role=requested_role)
    assert uid is not None
    if status != "active":
        db_service.update_user_fields(uid, status=status)
    return uid


def login_session(client, user_id):
    with client.session_transaction() as sess:
        sess["user_id"] = user_id


def _new_client():
    if not _artifacts_available():
        pytest.skip("Run database/init_db.py, training/train_models.py, and "
                    "scripts/precompute_predictions.py before running API tests.")
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def anon_client():
    """Not logged in."""
    return _new_client()


@pytest.fixture
def client():
    """Logged in as a RESEARCHER (the feature tests need researcher-level pages/APIs).
    Access-control behaviour itself is tested in tests/test_access_control.py."""
    c = _new_client()
    login_session(c, make_user("RESEARCHER"))
    return c
