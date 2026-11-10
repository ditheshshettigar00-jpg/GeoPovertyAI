# SECURITY ARCHITECTURE & VULNERABILITY AUDIT NOTES
## GeoPoverty AI: Geospatial AI Platform Security Specification

**Document Version:** 1.0  
**Audit Date:** November 2026  
**Status:** Complete & Verified  

---

## 1. Executive Summary

GeoPoverty AI implements a defense-in-depth security model protecting remote sensing telemetry, predictive pipelines, explainable AI outputs, and administrative governance. This specification details the platform's authentication architecture, role-based authorization tiers, cookie policies, request rate limiting, input sanitization, database query parameterization, and cryptographic guarantees, followed by an exhaustive repository audit for hardcoded secrets and known system limitations.

---

## 2. Authentication & Credential Architecture

### 2.1 Password Hashing & Verification
* **Algorithm:** Password hashing uses Werkzeug's security framework implementing `scrypt` key derivation with random per-password salting (`generate_password_hash`, `check_password_hash`).
* **Complexity Requirements:** All account registrations (`/register`, `POST /api/auth/register`) and interactive admin creation (`scripts/create_admin.py`) enforce a mandatory minimum length of **8 characters** (`len(password) >= 8`). Submissions under 8 characters are rejected with HTTP 400 Bad Request.
* **Storage Invariant:** Plaintext passwords are never stored in memory longer than the request duration, never logged, and never written to disk. The `users` table stores only the salted cryptographic hash in `password_hash TEXT NOT NULL`.

### 2.2 Account Lifecycle & Inactivity
* User accounts carry a `status` field (`'active'` or `'inactive'`).
* When an account is marked `'inactive'` by an administrator, subsequent login attempts are rejected with HTTP 403 Forbidden (`{"error": "Account is inactive"}`).

---

## 3. Tiered Role-Based Access Control (RBAC) & Approval Gate

### 3.1 Authorization Tiers
The platform defines three mutually exclusive authorization tiers:
1. **`PUBLIC` (Public / Student / Anonymous):**
   - Unauthenticated visitors can view the Landing Page (`/`), About Page (`/about`), National Choropleth Map (`/map`), and District Drilldown (`/district/<id>`).
   - Read-only data endpoints (`/api/areas`, `/api/map`, `/api/predict/<id>`, `/api/indicators/<id>`, `/api/area-profile/<id>`, `/api/satellite-image/<id>`) require no credentials, ensuring open educational access.
2. **`RESEARCHER`:**
   - Full access to analytical and decision-support capabilities: Live Predictor (`/prediction`, `POST /api/predict-live`), Explainability (`/explanation`, `GET /api/explain/<id>`), Performance Benchmarks (`/performance`, `GET /api/model-performance`), District Dossiers & PDF Generation (`/report/<id>`, `GET /api/reports/district/<id>`), District Matching (`/compare`, `/matchmaker`, `POST /api/compare`, `GET /api/similar-districts/<id>`), Hotspots (`/hotspots`, `GET /api/hotspots`), Hidden Gap Residual Atlas (`/hidden-gaps`, `GET /api/residual-atlas`), and Scenario Explorer (`/scenario-explorer`, `POST /api/scenario`, `GET /api/scenario-presets`, `GET /api/scenario-targets`).
3. **`ADMIN`:**
   - Governance portal (`/admin`) and administrative management APIs (`/api/admin/*`): user role assignment, user activation/deactivation, pending researcher approvals/rejections, and immutable audit log inspection (`audit_logs` table).

### 3.2 Researcher Approval Gate
- **Client Untrusted:** During user self-registration, client requests cannot directly grant the `RESEARCHER` role. Any registration with "Request researcher access" checked sets `role='PUBLIC'` and flags `requested_role='RESEARCHER'`.
- **Immediate Public Access:** The applicant can immediately authenticate with `PUBLIC`-level privileges while their application is queued for administrative review.
- **Administrator Review:** Administrators review applicants via `/api/admin/pending-researchers` and approve or reject them via `POST /api/admin/users/<id>/approve-researcher` or `reject-researcher`. Approvals promote the user to `role='RESEARCHER'` and clear `requested_role`. Every moderation action is recorded in `audit_logs`.

### 3.3 Bootstrap Policy & Administrative Safety Invariants
- **No Self-Registration for Admins:** The `ADMIN` role cannot be requested or created via `/register` or any public API endpoint.
- **Bootstrapping:** Administrators can only be created via the secure command-line tool (`python scripts/create_admin.py`) or promoted by an existing active administrator.
- **Self-Protection Guard (`_guard_admin_change`):** Administrators cannot demote or deactivate their own active account.
- **Last-Admin Guard:** The system rejects any administrative operation that would leave the application with zero active administrators (HTTP 400 Bad Request).

---

## 4. Session & HTTP Header Security

### 4.1 Cookie Security Policy
* **`SESSION_COOKIE_HTTPONLY = True`**: Prevents client-side scripts from reading session identifiers via `document.cookie`, mitigating Cross-Site Scripting (XSS) session hijacking.
* **`SESSION_COOKIE_SAMESITE = 'Lax'`**: Protects against Cross-Site Request Forgery (CSRF) by preventing the browser from sending session cookies with cross-site state-changing requests.
* **`SESSION_COOKIE_SECURE = (FLASK_ENV != 'development')`**: Automatically set to `True` in production and testing environments where `FLASK_ENV != 'development'`, ensuring session cookies are only transmitted over TLS/HTTPS connections. In local development (`FLASK_ENV='development'`), it defaults to `False` to facilitate local localhost HTTP development.

### 4.2 Cross-Origin Resource Sharing (CORS)
* Enforces an explicit, deny-by-default CORS policy via Flask-CORS.
* External origin requests are rejected unless explicitly whitelisted in `config.CORS_ORIGINS`.

---

## 5. Rate Limiting & Denial-of-Service Defense

### 5.1 Login Brute-Force Rate Limiting
* **Policy:** Tracks consecutive failed login attempts keyed on the tuple `(email, client_ip)`.
* **Threshold:** If **5 consecutive failed attempts** occur within a **15-minute sliding window** (`RATE_LIMIT_WINDOW_SECONDS = 900`), subsequent login attempts for that pair are blocked immediately with **HTTP 429 Too Many Requests** (`{"error": "Too many failed login attempts. Please try again later."}`).
* **Reset on Success:** A successful login clears the failed counter for that `(email, client_ip)` key.

### 5.2 Client IP Resolution & Proxy Header Hardening
* **Header Spoofing Defense:** The client IP resolution helper `_get_client_ip()` does **not** blindly trust the `X-Forwarded-For` HTTP header.
* **Controlled Trust:** `X-Forwarded-For` is only parsed when an explicit environment variable `TRUST_PROXY=1` is set (indicating the application runs behind a verified reverse proxy like Nginx or Cloudflare). Otherwise, it defaults strictly to `request.remote_addr`, preventing attackers from evading rate limits by rotating spoofed IP headers.

---

## 6. Input Validation & Mathematical Bounds Checking

### 6.1 What-If & Live Inference Input Validation
The analytical endpoints `POST /api/scenario` and `POST /api/predict-live` perform strict server-side validation on all incoming feature values:
* **Type Validation:** Rejects non-numeric payloads (strings, arrays, nulls, booleans) with HTTP 400 Bad Request. Booleans (which inherit from `int` in Python) are explicitly validated and rejected.
* **Feature Whitelisting:** Rejects unknown feature keys with HTTP 400 Bad Request. The model accepts only the 4 canonical satellite features: `nightlight_mean`, `ndvi_mean`, `builtup_pct`, and `population_density`.
* **Range Bounds Warnings:** Compares user-supplied values against empirical min/max training boundaries (`config.FEATURE_TRAINING_RANGES`). Out-of-bounds inputs generate prominent warning notifications (`outside the training range`) to prevent silent, misleading out-of-distribution extrapolation.

### 6.2 Open-Redirect Defense
The login redirection parameter `?next=<path>` is validated using `safe_next()`. Any target URL that is not a relative, same-origin path starting with `/` (such as `//evil.com` or `https://attacker.org`) is rejected and replaced with the user's role-appropriate default landing route.

---

## 7. SQL Injection & Database Parameterization Audit

Every SQL query in `backend/services/db_service.py` was audited for parameter binding. Dynamic values are exclusively passed via SQLite `?` placeholder parameters:
* `SELECT * FROM areas WHERE area_id = ?` `(area_id,)`
* `SELECT * FROM predictions WHERE area_id = ? ...` `(area_id,)`
* `SELECT * FROM explanations WHERE prediction_id = ?` `(prediction_id,)`
* `SELECT * FROM users WHERE email = ?` `(email,)`
* `SELECT * FROM users WHERE user_id = ?` `(user_id,)`
* `INSERT INTO users (...) VALUES (?, ?, ?, ?, 'active', ?)`
* `INSERT INTO audit_logs (...) VALUES (?, ?, ?)`
* `UPDATE users SET ... WHERE user_id = ?`: column names are strictly whitelisted against `{'role', 'status', 'requested_role'}` and all values are passed through parameter arrays.

Zero instances of Python string formatting (`%`, `.format()`, or f-strings) exist in SQL execution statements.

---

## 8. Path Traversal & Image Endpoint Defense

* **Integer Route Converters:** Endpoints serving district imagery (`/api/satellite-image/<int:area_id>` and `/api/gradcam/<int:area_id>`) enforce Flask's `<int:area_id>` URL converter, rejecting directory traversal characters (`../`, `..%2F`) at the routing layer with HTTP 404.
* **Canonical Path Confinement:** In `satellite_image(area_id)`, the resolved filesystem path is strictly validated to ensure it resides within the absolute canonical directory:
  ```python
  if not path or not os.path.abspath(path).startswith(os.path.abspath(SATELLITE_DIR)):
      return jsonify({"error": "No satellite image available for this district"}), 404
  ```
* **Grad-CAM Deactivation:** The Grad-CAM endpoint (`/api/gradcam/<int:area_id>`) returns HTTP 404 immediately without touching the filesystem, documenting that the final tabular model uses satellite statistics, not deep neural network features.

---

## 9. Cross-Site Scripting (XSS) & Template Escaping Audit

* **Jinja2 Auto-Escaping:** Flask configures Jinja2 with `autoescape=True` for all `.html` templates. Dynamic variables (such as `{{ current_user.name }}`) are automatically HTML-escaped.
* **No `| safe` on User Input:** An exhaustive repository search confirmed that the Jinja `| safe` filter is never applied to user-supplied strings or database values.
* **Client-Side JavaScript Escaping:** In administrative interfaces (`admin_dashboard.html`) and analytical tables (`hidden_gaps.html`), dynamic DOM insertions via `innerHTML` pass all data through an explicit sanitization function:
  ```javascript
  function esc(v) {
      return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  ```

---

## 10. Repository Hardcoded Secrets Audit

An exhaustive search across all files in the Git index (`git grep -n -i -E "secret|password|token|api[_-]?key"`) verified that no credentials, API keys, or private tokens are hardcoded.

### Audit Findings Table

| File | Context / Match | Classification | Security Assessment |
| :--- | :--- | :---: | :--- |
| `backend/app.py:46-53` | `os.environ.get("FLASK_SECRET_KEY")`<br/>`app.secret_key = "dev_secret_key_geopoverty"` | Safe Dev Fallback | Allowed **only** when `FLASK_ENV=development`. If `FLASK_ENV != 'development'` and `FLASK_SECRET_KEY` is missing, the application deliberately crashes at startup (`RuntimeError`). |
| `backend/api/auth.py` | `generate_password_hash`, `check_password_hash` | Safe Code | Uses Werkzeug scrypt hashing for credential verification. |
| `backend/services/db_service.py` | `password_hash` column parameter | Safe Code | Stores and updates cryptographic password hashes only. |
| `database/schema.sql:14` | `password_hash TEXT NOT NULL` | Safe Schema | Schema definition for password hashes. |
| `scripts/create_admin.py` | `getpass.getpass(...)`<br/>`os.environ.get("ADMIN_PASSWORD")` | Safe CLI | Interactive prompt with hidden input or optional deployment environment variable. No fallback defaults. |
| `tests/*.py` | Test strings (`password123`, `Password123!`) | Safe Test Mock | Ephemeral dummy credentials used in isolated test fixtures. |
| `geospatial/GEE_SETUP.md:42` | Reference to `earthengine` token path | Safe Docs | Explains Google Earth Engine local configuration path. |
| `preprocessing/name_keys.py:49` | `def distinguishing_tokens(value)` | Safe Code | NLP string tokenization utility for district names. |

---

## 11. Security Limitations & Operational Boundaries

To maintain scientific and operational honesty, the following constraints are documented:

1. **In-Memory Rate Limiter:** The failed login counter (`FAILED_LOGIN_ATTEMPTS`) is stored in process memory. It is local to each worker process and resets if the application restarts. In multi-worker production deployments, a distributed store (e.g., Redis) should be used.
2. **SQLite Concurrency & Locking:** SQLite operates with database-level write locks. High-frequency concurrent writes may encounter `sqlite3.OperationalError: database is locked`. The application enforces `conn.rollback()` on all exception blocks to prevent connection corruption.
3. **Absence of Email Verification:** User accounts are activated upon registration without email confirmation. In academic/research demonstration mode, identity verification is handled manually via the administrator approval queue.
4. **No Self-Service Password Reset:** There is no automated password reset or "forgot password" email flow. Administrative intervention is required to update credentials for locked users.
5. **No Individual Tracking:** The application operates strictly on aggregate administrative district statistics. It possesses no capability to identify, locate, or profile individual households.
