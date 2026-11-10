"""
app.py
--------
Flask application entry point. Serves:
  - the REST API (backend/api/routes.py, under /api)
  - the HTML dashboard pages (frontend/templates), rendered server-side
    with Jinja2 and populated via JavaScript calling the /api endpoints

Run with:
    python backend/app.py
Then open http://127.0.0.1:5000/ in a browser.
"""

import os
import sys

from flask import Flask, render_template, session, redirect, url_for
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.api.routes import api_bp
from backend.api.auth import auth_bp
from backend.api.reports import reports_bp
from backend.api.admin import admin_bp
from backend.services import db_service, model_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(FRONTEND_DIR, "templates"),
        static_folder=os.path.join(FRONTEND_DIR, "static"),
    )
    # CORS: allow only origins listed in env CORS_ORIGINS; default is none
    cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
    if cors_origins_env:
        origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    else:
        origins = []
    CORS(app, origins=origins)
    
    # Secret key: read FLASK_SECRET_KEY. Dev key ONLY when FLASK_ENV=development
    flask_secret_key = os.environ.get("FLASK_SECRET_KEY")
    flask_env = os.environ.get("FLASK_ENV")
    if flask_secret_key:
        app.secret_key = flask_secret_key
    elif flask_env == "development":
        app.secret_key = "dev_secret_key_geopoverty"
    else:
        raise RuntimeError("FLASK_SECRET_KEY must be set in the environment unless FLASK_ENV=development")

    # Session cookie hardening
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # FLASK_DEBUG defaults to False
    app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "yes"]

    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(admin_bp)

    @app.teardown_appcontext
    def teardown_db(exception):
        db_service.close_connection(exception)

    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/prediction")
    def prediction_page():
        return render_template("prediction.html")

    @app.route("/map")
    def map_page():
        return render_template("map.html")

    @app.route("/explanation")
    def explanation_page():
        return render_template("explanation.html")

    @app.route("/performance")
    def performance_page():
        return render_template("performance.html")

    @app.route("/district/<int:area_id>")
    def district_page(area_id):
        return render_template("district.html", area_id=area_id)

    @app.route("/report/<int:area_id>")
    def report_page(area_id):
        """In-browser HTML report preview page for a district."""
        return render_template("report.html", area_id=area_id)

    @app.route("/compare")
    def compare_page():
        return render_template("compare.html")

    @app.route("/matchmaker")
    def matchmaker_page():
        return render_template("matchmaker.html")



    @app.route("/hidden-gaps")
    def hidden_gaps_page():
        return render_template("hidden_gaps.html")

    @app.route("/hotspots")
    def hotspots_page():
        return render_template("hotspots.html")

    @app.route("/data-quality")
    def data_quality_page():
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return render_template("data_quality.html")

    @app.route("/admin")
    def admin_page():
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        user = db_service.get_user_by_id(session["user_id"])
        if not user or user["role"] != "ADMIN":
            return redirect(url_for("home"))
        return render_template("admin_dashboard.html")

    @app.route("/scenarios")
    @app.route("/scenario-explorer")
    @app.route("/whatif")
    def scenarios_page():
        return render_template("whatif.html")

    @app.route("/login")
    def login_page():
        return render_template("login.html")

    @app.route("/register")
    def register_page():
        return render_template("register.html")

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    return app


app = create_app()

if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "yes"]
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=is_debug, port=port)
