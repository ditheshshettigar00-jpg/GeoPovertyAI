# SECTION 2.6: USER INTERFACE DESIGN SPECIFICATION
## GeoPoverty AI Decision-Support Platform

This document provides the isolated **Section 2.6 (User Interface Design)** with figures, screenshots, and visual descriptions formatted for direct inclusion into your Software Design Specification (SDS).

---

## 2.6 User Interface Design

### 2.6.1 UI Architecture and Layout Strategy
The GeoPoverty AI user interface is engineered around an intuitive **Decision-Support Paradigm** designed for public policy analysts, researchers, and government planners. Built with a responsive, zero-framework CSS architecture and Jinja2 server-rendered views, the platform emphasizes clarity, high-contrast geographic telemetry, and interactive model interpretability.

The interface adheres to a standardized three-tier structure:
1. **Persistent Global Navigation (`base.html`):** Provides instant navigation between Home, Interactive Map, Predictions, What-If Simulator, Hotspots, Reports, and RBAC authentication states.
2. **Dynamic Analytical Workspace:** Responsive multi-column layout rendering geospatial choropleths, feature tables, Chart.js graphs, and interactive sliders.
3. **Contextual Action Panels:** Contextual buttons enabling immediate dynamic PDF dossier exports, scenario resets, and drill-down filters.

---

### 2.6.2 Key Interface Views and Visual Walkthrough

#### View 1: Executive Landing & Overview Dashboard
![Executive Landing & Overview Dashboard](screenshots/01_home_dashboard.png)
*Figure 2.6.1: Executive Landing Page and System Capabilities Overview (`home.html`)*

- **Purpose:** Introduces the GeoPoverty AI decision-support platform, summarizes national coverage metrics, and guides users to analytical tools.
- **Key UI Components:**
  - **Hero Banner:** Displays system value proposition with quick-action CTA buttons (`Live Predictor` and `Interactive Map`).
  - **Executive Stat Badges:** Real-time counters showing total geographic areas analyzed (`532 Districts Analyzed`), ground telemetry points (`60 Geospatial Records`), and active algorithm backbones (`RF / XGB Trained Models`).
  - **Capability Feature Cards:** Direct navigational cards highlighting Explainable AI (SHAP & Grad-CAM), Scenario Simulation, and Hotspot Detection.
- **User Interactions:** Single-click routing to map exploration, model diagnostics, or district prediction search.

---

#### View 2: Interactive Geospatial Choropleth Map
![Interactive Geospatial Choropleth Map](screenshots/02_interactive_map.png)
*Figure 2.6.2: Interactive Geospatial Deprivation Choropleth Map (`map.html`)*

- **Purpose:** Provides nationwide spatial intelligence, allowing users to visually detect spatial inequality patterns and geographic deprivation clusters across Indian districts.
- **Key UI Components:**
  - **Filter & Search Sidebar:** Input fields for instant district keyword filtering (`Search District`), state selection dropdown, and deprivation category multi-selector (`Low`, `Medium`, `High`).
  - **Leaflet.js Choropleth Canvas:** Vector polygon rendering with dynamic chloropleth styling:
    - 🔴 **Red:** High Deprivation (Score $> 0.61$)
    - 🟠 **Orange:** Medium Deprivation ($0.33 < \text{Score} \le 0.61$)
    - 🟢 **Green:** Low Deprivation ($\text{Score} \le 0.33$)
  - **Interactive Legend & Tooltips:** Dynamic tooltips on hover displaying district name, state, predicted deprivation score, and classification tertile.
- **User Interactions:** Hovering highlights district borders; clicking a district opens a contextual popup with a direct link to the full District Deep-Dive dossier.

---

#### View 3: Policy Intervention "What-If" Simulator
![Policy Intervention What-If Simulator](screenshots/03_whatif_simulator.png)
*Figure 2.6.3: What-If Policy Intervention and Scenario Simulation Engine (`whatif.html`)*

- **Purpose:** Enables evidence-based policy planning by allowing decision-makers to simulate the outcome of infrastructure projects (e.g., building roads or hospitals) before committing capital.
- **Key UI Components:**
  - **District Selector Card:** Centered administrative dropdown for selecting the baseline target district (e.g., *Anantapur, Andhra Pradesh*).
  - **Left Panel (Policy Intervention Controls):**
    - One-click policy preset cards (*Rural Electrification Drive*, *Healthcare Accessibility*, *Education Infrastructure*, *Road Expansion*).
    - Granular sliders for continuous feature overrides (e.g., reducing road distance from $12\text{ km}$ to $3\text{ km}$, or increasing nighttime lighting).
  - **Right Panel (Impact Analysis & Live Sensitivity):**
    - High-visibility comparison cards: **Current Baseline Score** vs. **Simulated Outcome Score**.
    - Real-time status shift badge indicating whether deprivation category transitioned (e.g., *High $\to$ Medium*).
    - Explanatory policy impact card articulating non-linear feature elasticities.
- **User Interactions:** Moving any slider fires an asynchronous AJAX request (`POST /api/scenario`) executing a live model inference in $< 50\text{ ms}$, instantly updating score comparison gauges without full page reloads.

---

#### View 4: District Deep-Dive & Dual-Tier Explainability (XAI)
![District Deep-Dive and XAI](screenshots/04_district_deepdive_xai.png)
*Figure 2.6.4: Granular District Dossier with Multimodal Indicators and SHAP Attributions (`district.html`)*

- **Purpose:** Provides complete transparency and explainability for a district's predicted score, establishing algorithmic trust through empirical feature contributions and satellite imagery inspection.
- **Key UI Components:**
  - **Header & Action Bar:** District identity, state jurisdiction, and an automated ReportLab PDF download button (`Generate District Report`).
  - **Deprivation Scorecard (Top-Left):** Prominent continuous prediction gauge (`0.405`), categorical classification badge (`Medium Deprivation`), model provenance (`XGBoost`), and audit timestamp.
  - **Comprehensive Geospatial Feature Table (Top-Right):** Tabulates raw infrastructure and environmental telemetry (VIIRS nightlights, NDVI greenness, built-up density, hospital/school distances) alongside extracted 16-d visual texture signals.
  - **Local SHAP Feature Importance Waterfall (Bottom-Left):** Horizontal Chart.js attribution chart displaying game-theoretic Shapley contributions:
    - Blue bars: Features pushing deprivation downwards (protective/affluence indicators like high nightlight radiance).
    - Red bars: Features driving deprivation upwards (vulnerability indicators like excessive hospital distances).
  - **Sentinel-2 Satellite Image Preview (Bottom-Right):** High-resolution $224 \times 224$ optical imagery patch showing true-color surface reflectance with modal zoom.
- **User Interactions:** Hovering over SHAP bars shows exact marginal delta ($\Delta y$); clicking the satellite thumbnail opens high-resolution inspection.

---

### 2.6.3 Design Aesthetics, Accessibility, and Responsiveness
1. **Curated Color Palette:**
   - **Primary Neutral:** Deep slate dark blue (`#0f172a` / `#1e293b`) for readability and modern appearance.
   - **Accent & Primary Actions:** Royal blue (`#2563eb` / `#1d4ed8`) for buttons, links, and active states.
   - **Semantic Risk Ramp:**
     - Critical Deprivation: Coral Red (`#ef4444`)
     - Moderate Deprivation: Amber Orange (`#f59e0b`)
     - Low Deprivation / Affluence: Emerald Green (`#10b981`)
2. **Typography:** Styled with modern, legible Google Web Fonts (`Inter` and `Segoe UI`) with strict hierarchy ($h_1$ to $h_4$) and high contrast ratios conforming to WCAG 2.1 AA accessibility guidelines.
3. **Responsive Grid Layout:** CSS Flexbox and Grid layouts automatically collapse from multi-column desktop dashboards into single-column vertical cards on tablet and mobile viewports, ensuring full usability for field teams on mobile devices.
