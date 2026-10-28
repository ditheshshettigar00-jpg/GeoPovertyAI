-- GeoPoverty AI database schema
-- Written for SQLite (default, zero-setup for a student laptop).
-- Notes for MySQL port are given as comments where syntax differs.

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- users : optional, only needed if you add login/admin features
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'PUBLIC',   -- 'PUBLIC', 'RESEARCHER', 'ADMIN'
    status        TEXT NOT NULL DEFAULT 'active',   -- 'active' or 'inactive'
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- areas : the geographic unit (district) - central table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS areas (
    area_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    area_name       TEXT NOT NULL,               -- e.g. district name
    state_name      TEXT NOT NULL,
    centroid_lat    REAL NOT NULL,
    centroid_lon    REAL NOT NULL,
    boundary_geojson TEXT,                       -- boundary polygon stored as GeoJSON text
    data_mode       TEXT NOT NULL DEFAULT 'demo'  -- 'demo' or 'real' -- ALWAYS distinguishes synthetic vs real rows
);

-- ------------------------------------------------------------
-- ground_truth : socioeconomic survey-based target values
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ground_truth (
    gt_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id             INTEGER NOT NULL REFERENCES areas(area_id),
    deprivation_score   REAL NOT NULL,           -- continuous target, 0 (least) - 1 (most deprived)
    data_source         TEXT NOT NULL,           -- e.g. 'Census2011_demo', 'Census2011'
    data_year           INTEGER,
    FOREIGN KEY (area_id) REFERENCES areas(area_id)
);

-- ------------------------------------------------------------
-- satellite_data : metadata about the image patch used per area
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satellite_data (
    sat_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id         INTEGER NOT NULL REFERENCES areas(area_id),
    image_path      TEXT,                        -- relative path to stored image patch
    source          TEXT NOT NULL DEFAULT 'Sentinel-2',
    capture_date    TEXT,
    cloud_pct       REAL,
    data_mode       TEXT NOT NULL DEFAULT 'demo',
    FOREIGN KEY (area_id) REFERENCES areas(area_id)
);

-- ------------------------------------------------------------
-- geospatial_features : nightlight / NDVI / built-up / accessibility etc.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS geospatial_features (
    feature_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id             INTEGER NOT NULL REFERENCES areas(area_id),
    nightlight_mean     REAL,
    ndvi_mean           REAL,
    builtup_pct         REAL,
    population_density  REAL,
    road_distance_km    REAL,
    school_distance_km  REAL,
    hospital_distance_km REAL,
    data_mode           TEXT NOT NULL DEFAULT 'demo',
    FOREIGN KEY (area_id) REFERENCES areas(area_id)
);

-- ------------------------------------------------------------
-- predictions : model output per area
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id              INTEGER NOT NULL REFERENCES areas(area_id),
    model_name           TEXT NOT NULL,          -- e.g. 'XGBoost_v1'
    predicted_score       REAL NOT NULL,
    predicted_category    TEXT,                  -- 'Low' / 'Medium' / 'High'
    prediction_timestamp   TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (area_id) REFERENCES areas(area_id)
);

-- ------------------------------------------------------------
-- explanations : SHAP + Grad-CAM output linked to a prediction
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS explanations (
    explain_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id        INTEGER NOT NULL REFERENCES predictions(prediction_id),
    shap_json             TEXT,                  -- serialized {feature: contribution}
    gradcam_image_path     TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- ------------------------------------------------------------
-- model_results : evaluation metrics per trained model version
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    mae             REAL,
    rmse            REAL,
    r2              REAL,
    trained_on      TEXT NOT NULL DEFAULT 'demo',  -- 'demo' or 'real'
    eval_timestamp  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- audit_logs : tracking user actions
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER,
    action      TEXT NOT NULL,
    timestamp   TEXT DEFAULT CURRENT_TIMESTAMP,
    details     TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
