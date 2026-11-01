import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set FLASK_ENV=development for testing suite as permitted by security guidelines
os.environ.setdefault("FLASK_ENV", "development")


def _artifacts_available():
    from config import DB_PATH, MODELS_DIR
    return os.path.exists(DB_PATH) and os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


@pytest.fixture
def client():
    if not _artifacts_available():
        pytest.skip("Run database/init_db.py, training/train_models.py, and "
                    "scripts/precompute_predictions.py before running API tests.")
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()
