# GeoPoverty AI: 5-Minute Viva Demo Walkthrough & Script

This document provides a click-by-click, step-by-step viva presentation script for **GeoPoverty AI**. It is designed to be completed in **5 to 7 minutes** in front of an external examiner, showing the core capabilities, role-based access control, scientific honesty features, and administrative workflows.

---

## 🛠️ Demo Setup & Pre-Flight Checklist

### 1. Start the Application
Open Windows PowerShell in the project root directory:
```powershell
$env:FLASK_ENV="development"
.\venv\Scripts\python backend\app.py
```
Confirm the console outputs:
```
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

### 2. Pre-Configure Demo Accounts
Run these two one-time commands in a separate terminal before the viva starts:

```powershell
# 1. Create the default Admin user
.\venv\Scripts\python scripts/create_admin.py --email admin@geopoverty.ai --password AdminPass123!

# 2. Register a demo Researcher user via Python or web UI
.\venv\Scripts\python -c "from backend.services.db_service import create_user, update_user_status; u = create_user('researcher@geopoverty.ai', 'ResearcherPass123!', 'RESEARCHER', 'Tata Institute of Social Sciences'); update_user_status(u['user_id'], 'ACTIVE')"

# 3. Register a PENDING researcher (to demonstrate approval live)
.\venv\Scripts\python -c "from backend.services.db_service import create_user; create_user('applicant@univ.edu', 'StudentPass123!', 'RESEARCHER', 'Department of Economics, Delhi University')"
```

### 3. Demo Credentials Reference Card
- **Public:** No login required.
- **Approved Researcher:** `researcher@geopoverty.ai` / `ResearcherPass123!`
- **Administrator:** `admin@geopoverty.ai` / `AdminPass123!`
- **Pending Applicant (for Step 11):** `applicant@univ.edu`

---

## ⏱️ The 5-Minute Click-by-Click Script

### Step 1: Landing Page (0:00 - 0:30)
- **URL:** [`http://127.0.0.1:5000/`](http://127.0.0.1:5000/)
- **What to Click:** Scroll down past the hero section, pointing to the project summary and telemetry cards.
- **What to Say:**
  > *"Good morning, sir/madam. This is **GeoPoverty AI**, an end-to-end geospatial machine learning platform designed to estimate multidimensional socioeconomic deprivation across India from satellite telemetry.*
  > *Our system covers **564 districts** across **31 states and union territories**, evaluated under strict **Leave-One-State-Out cross-validation** to prevent spatial data leakage.*
  > *Notice that we provide open public transparency for citizens and public officials, while protecting advanced policy and simulation tools behind Role-Based Access Control."*

---

### Step 2: Public National Map (0:30 - 1:00)
- **URL:** [`http://127.0.0.1:5000/map`](http://127.0.0.1:5000/map)
- **What to Click:**
  1. Click **"Explore National Map"** from the navigation bar or hero button.
  2. Hover over several districts to show the Leaflet tooltip displaying State, District, Predicted Score, and Deprivation Tier (`Low`, `Medium`, `High`).
  3. Type `Pashchim Singhbhum` or `Purulia` in the district search bar.
- **What to Say:**
  > *"Here is the interactive national choropleth map. The 564 districts are categorized into tertiles—Low, Medium, and High deprivation—derived directly from the empirical tertiles of our NFHS-5 training ground truth (cutoffs at 0.3198 and 0.4526).*
  > *All scores displayed on this map are **out-of-state estimates**, meaning each district was predicted by a model that never saw its home state during training.*
  > *Let's drill down into an individual district: Pashchim Singhbhum in Jharkhand."*

---

### Step 3: District Indicators Drilldown (1:00 - 1:35)
- **URL:** [`http://127.0.0.1:5000/district/182`](http://127.0.0.1:5000/district/182)
- **What to Click:** Click **"View District Profile"** from the map popup or navigate directly to `/district/182`.
- **What to Point At:**
  1. Point at the **Category Badge** (`Medium` deprivation, predicted score: `0.4423`, 80% range `[0.3060, 0.5860]`).
  2. Point at Section 5A: **Satellite-Derived Indicators** table (7 indicators: nightlights, NDVI, built-up %, population density, S2 scenes, lat/long).
  3. Point at Section 5B: **NFHS-5 Survey Indicators** table (5 components: electricity, sanitation, clean fuel, stunting, anaemia).
- **What to Say:**
  > *"On the district profile, every single number is an empirical metric—nothing is mocked or hardcoded.*
  > *Section 5A shows all 7 satellite indicators with the district value, national percentile, and state median.*
  > *Section 5B displays the 5 ground-truth components from NFHS-5 that form our target score.*
  > *Notice our target construction rule: household electricity, sanitation, and clean cooking fuel enter the deprivation score as $100 - x$ so higher means greater deprivation, while child stunting and maternal anaemia enter directly. All 5 are min-max scaled and averaged."*

---

### Step 4: Role-Based Access Control Demonstration (1:35 - 2:05)
- **URL:** Attempt to open [`http://127.0.0.1:5000/compare`](http://127.0.0.1:5000/compare) as an anonymous visitor.
- **What to Point At:** Observe the immediate redirect to `/login?next=%2Fcompare` with the flash alert: *"Please log in to access this page."*
- **What to Click:** Enter credentials for the pre-approved researcher:
  - **Email:** `researcher@geopoverty.ai`
  - **Password:** `ResearcherPass123!`
  - Click **Sign In**. The user is immediately redirected to the protected `/compare` page, and the navbar now displays the `RESEARCHER` badge.
- **What to Say:**
  > *"To protect advanced analytical endpoints from unmonitored scraping and automated misuse, our system implements three-tier Role-Based Access Control (PUBLIC, RESEARCHER, ADMIN).*
  > *When an anonymous user attempts to access an analytical tool, Flask's `@require_role` decorator redirects them to the login flow while preserving the target route.*
  > *Now that we are authenticated as an approved researcher from TISS, we have access to advanced research tools."*

---

### Step 5: Comparative District Analysis (2:05 - 2:30)
- **URL:** [`http://127.0.0.1:5000/compare`](http://127.0.0.1:5000/compare)
- **What to Click:** Select **Pashchim Singhbhum** (Jharkhand) as District 1 and **Purulia** (West Bengal) or **Kawardha** as District 2. Click **Compare**.
- **What to Point At:** The comparative delta table and radar/bar charts showing side-by-side satellite radiance, vegetative cover, and NFHS-5 health telemetry.
- **What to Say:**
  > *"The Compare tool enables inter-district cross-examination. Even when two neighboring districts share similar vegetation indices, nighttime luminosity and built-up settlement percentages reveal significant divergence in infrastructure density and access to improved sanitation."*

---

### Step 6: Hidden Gap Detector (2:30 - 3:05)
- **URL:** [`http://127.0.0.1:5000/hidden-gaps`](http://127.0.0.1:5000/hidden-gaps)
- **What to Click:** Click **"Hidden Gaps"** in the navigation bar.
- **What to Point At:**
  1. Point at the scatter plot showing Survey Deprivation vs. Satellite Estimate.
  2. Point at the summary chips: **43 Hidden-Deprivation Candidates**, **46 Resilient Candidates**, **475 Consistent (84.2%)**.
  3. Point at the two-stage candidate criteria in the methodology card.
- **What to Say:**
  > *"This is our **Hidden Gap Detector**, one of our core research contributions.*
  > *Because statistical models suffer from mean compression, simply looking at prediction errors would falsely label almost every severely poor district as a gap.*
  > *We enforce a strict two-stage filter: a district is only a candidate if (1) its survey score lies strictly outside the out-of-state 80% prediction interval, AND (2) its percentile-rank gap is at least 25 percentile points.*
  > *We explicitly emphasize to policymakers that this is a **screening hint for field audits**, not proof of underreporting or causal failure."*

---

### Step 7: Deprivation Hotspots via DBSCAN (3:05 - 3:35)
- **URL:** [`http://127.0.0.1:5000/hotspots`](http://127.0.0.1:5000/hotspots)
- **What to Click:** Click **"Hotspots"** in the navigation bar.
- **What to Point At:** The clustered geographic belts on the map and the cluster cards below.
- **What to Say:**
  > *"Rather than viewing districts in administrative isolation, the Hotspot Explorer groups contiguous regions of severe poverty.*
  > *We execute **Haversine DBSCAN** directly on district centroid coordinates with an $\text{eps}$ radius of 150 km ($0.02354$ radians) and a $\text{min\_samples}$ requirement of 3 districts in the 'High' deprivation tertile.*
  > *This identifies genuine cross-state deprivation belts—such as the Central Tribal Belt spanning eastern Madhya Pradesh, southern Bihar, and Jharkhand—allowing regional administrative coordination."*

---

### Step 8: Scenario Explorer & Sensitivity Disclaimer (3:35 - 4:05)
- **URL:** [`http://127.0.0.1:5000/scenario-explorer`](http://127.0.0.1:5000/scenario-explorer)
- **What to Click:**
  1. Select District: `Pashchim Singhbhum`.
  2. Select Preset: `Electrification Proxy (+25% NTL)`.
  3. Click **Simulate Scenario**.
- **What to Point At:**
  1. Point at the before/after score gauge and change delta.
  2. Point at the **Sub-MAE notice** if the delta is $< 0.0883$.
  3. Point at the **Mandatory Disclaimer banner** at the top.
- **What to Say:**
  > *"The Scenario Explorer allows researchers to test model sensitivity under hypothetical input shifts.*
  > *Notice two critical scientific safeguards: First, if the simulated change is smaller than our model's cross-validation MAE (0.0883), the system warns that the change is within the typical error margin.*
  > *Second, we display an unavoidable disclaimer: this is a **sensitivity proxy**, not a policy forecasting tool. Healthcare, schooling, and welfare programs are not in the model's feature space."*

---

### Step 9: Model Explainability & SHAP Attributions (4:05 - 4:30)
- **URL:** [`http://127.0.0.1:5000/explanation`](http://127.0.0.1:5000/explanation)
- **What to Click:** Click **"Explainability"**; view the global feature importance bar chart, then select `Pashchim Singhbhum` to view local SHAP values.
- **What to Point At:** Point at the negative contribution of nighttime lights and positive contribution of low built-up area.
- **What to Say:**
  > *"Here we expose TreeSHAP feature attributions. Nighttime luminosity and built-up land percentage are the strongest drivers reducing predicted deprivation.*
  > *Again, we remind examiners and users that SHAP values are strictly associational, not causal."*

---

### Step 10: District Executive Report & Live PDF Generation (4:30 - 4:55)
- **URL:** [`http://127.0.0.1:5000/report/182`](http://127.0.0.1:5000/report/182)
- **What to Click:**
  1. Scroll through the 7 executive report sections on screen.
  2. Click the **"Export PDF Report"** button (opens `/api/reports/district/182/pdf`).
- **What to Point At:** Show the downloaded publication-ready ReportLab PDF with Section 5A and 5B tables, metadata, and methodology notes.
- **What to Say:**
  > *"For district collectors and development officers, the platform generates a comprehensive 7-section executive report.*
  > *Clicking Export PDF produces an on-demand, print-ready document formatted with ReportLab, containing all real indicators, leave-one-state-out validation metrics, and calibrated prediction intervals."*

---

### Step 11: Admin Dashboard & Researcher Approval (4:55 - 5:20)
- **URL:** [`http://127.0.0.1:5000/logout`](http://127.0.0.1:5000/logout) $\rightarrow$ [`http://127.0.0.1:5000/login`](http://127.0.0.1:5000/login)
- **What to Click:**
  1. Log in as Admin: `admin@geopoverty.ai` / `AdminPass123!`.
  2. Navigate to [`http://127.0.0.1:5000/admin`](http://127.0.0.1:5000/admin).
  3. In the **Pending Approvals** table, find `applicant@univ.edu` (Department of Economics, Delhi University).
  4. Click the green **Approve** button.
- **What to Point At:** The applicant immediately transitions from `PENDING` to `ACTIVE` with role `RESEARCHER`, recorded in the security audit logs.
- **What to Say:**
  > *"Finally, the Admin Dashboard allows system administrators to oversee platform governance, review audit logs, and approve or reject researcher applications after verifying institutional credentials.*
  > *This completes the end-to-end walkthrough of GeoPoverty AI."*

---

## 🆘 Troubleshooting & Fallback Cheat Sheet

| Scenario | What to Do (Commands & Fixes) |
|---|---|
| **Port 5000 is already in use** | Run on a different port: `.\venv\Scripts\python backend/app.py --port 5001` or in PowerShell kill the PID holding 5000: `Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess \| Stop-Process -Force` |
| **`geopoverty.db` missing or corrupt** | Re-initialize in 3 commands: <br>`.\venv\Scripts\python database/init_db.py`<br>`.\venv\Scripts\python scripts/precompute_predictions.py`<br>`.\venv\Scripts\python scripts/create_admin.py --email admin@geopoverty.ai --password AdminPass123!` |
| **Examiner asks to verify test suite live** | Run: `.\venv\Scripts\pytest -q` (proves all 270 automated tests pass in ~2.5 minutes). |
| **Examiner asks: "Show me the training code"** | Open [`training/train_models.py`](../training/train_models.py) and [`training/pipelines.py`](../training/pipelines.py). Point out `LeaveOneGroupOut` on state codes and median imputation inside the pipeline. |
| **Examiner asks: "Where are the model weights saved?"** | Open [`models/model_RandomForest.joblib`](../models/) and [`models/model_comparison_results.csv`](../models/model_comparison_results.csv). |
| **Admin credentials forgotten** | Reset or create a new admin instantly from CLI: `.\venv\Scripts\python scripts/create_admin.py --email admin@geopoverty.ai --password AdminPass123!` |
