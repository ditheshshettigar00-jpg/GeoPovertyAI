"""
Generate professional 12-slide PowerPoint presentation for GeoPoverty AI 2nd Review.
Strictly conforms to departmental presentation guidelines (Max 12 slides):
- Slide 1: Title & Administrative Details
- Slides 2-4: Software Design Specification (SDS)
- Slides 5-8: Model & Methodology
- Slides 9-11: Performance Metrics & Evaluation
- Slide 12: Conclusion, Milestones & Roadmap / Q&A
Includes complete speaker notes embedded in every slide for Presenter View.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# -------------------------------------------------------------
# Color Palette Constants
# -------------------------------------------------------------
# Dark Theme (Slide 1 & 12)
DARK_BG = RGBColor(15, 23, 42)        # Slate 900
DARK_CARD = RGBColor(30, 41, 59)      # Slate 800
DARK_BORDER = RGBColor(51, 65, 85)    # Slate 700
WHITE = RGBColor(255, 255, 255)
LIGHT_MUTED = RGBColor(148, 163, 184) # Slate 400

# Light Theme (Slides 2-11)
LIGHT_BG = RGBColor(248, 250, 252)    # Slate 50
CARD_BG = RGBColor(255, 255, 255)     # Pure White
CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
DARK_TEXT = RGBColor(15, 23, 42)      # Slate 900
BODY_TEXT = RGBColor(51, 65, 85)      # Slate 700
MUTED_TEXT = RGBColor(100, 116, 139)  # Slate 500

# Accent Colors
ACCENT_BLUE = RGBColor(14, 165, 233)   # Sky 500
ACCENT_NAVY = RGBColor(30, 58, 138)   # Blue 900
ACCENT_TEAL = RGBColor(13, 148, 136)   # Teal 600
ACCENT_EMERALD = RGBColor(16, 185, 129)# Emerald 500
ACCENT_AMBER = RGBColor(245, 158, 11)  # Amber 500
ACCENT_PURPLE = RGBColor(124, 58, 237) # Purple 600
ACCENT_ROSE = RGBColor(225, 29, 72)    # Rose 600
ROW_ALT_BG = RGBColor(241, 245, 249)  # Slate 100

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, category_text, title_text, slide_idx):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_BLUE
    line.line.color.rgb = ACCENT_BLUE

    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(10.5), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_BLUE
    p_cat.space_after = Pt(2)

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_TEXT

    badge_box = slide.shapes.add_textbox(Inches(11.333), Inches(0.48), Inches(1.2), Inches(0.4))
    btf = badge_box.text_frame
    btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
    bp = btf.paragraphs[0]
    bp.alignment = PP_ALIGN.RIGHT
    bp.text = f"{slide_idx} / 12"
    bp.font.name = FONT_HEADING
    bp.font.size = Pt(11)
    bp.font.bold = True
    bp.font.color.rgb = MUTED_TEXT

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card

def add_stat_box(slide, left, top, width, height, value, label, subtext="", color=ACCENT_BLUE):
    add_card(slide, left, top, width, height, bg_color=WHITE, border_color=CARD_BORDER)
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.08), height)
    strip.fill.solid()
    strip.fill.fore_color.rgb = color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = str(value)
    p_val.font.name = FONT_HEADING
    p_val.font.size = Pt(22)
    p_val.font.bold = True
    p_val.font.color.rgb = color
    p_val.space_after = Pt(2)

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.font.name = FONT_HEADING
    p_lbl.font.size = Pt(11)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = DARK_TEXT

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = MUTED_TEXT

def create_bullet_card(slide, left, top, width, height, title, items, tag=None, tag_color=ACCENT_BLUE):
    add_card(slide, left, top, width, height, bg_color=WHITE, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(13)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_TEXT
    p_title.space_after = Pt(8)

    for item in items:
        p_item = tf.add_paragraph()
        if isinstance(item, tuple):
            lead, body = item
            run_lead = p_item.add_run()
            run_lead.text = "• " + lead + ": "
            run_lead.font.bold = True
            run_lead.font.name = FONT_BODY
            run_lead.font.size = Pt(10)
            run_lead.font.color.rgb = DARK_TEXT

            run_body = p_item.add_run()
            run_body.text = body
            run_body.font.name = FONT_BODY
            run_body.font.size = Pt(10)
            run_body.font.color.rgb = BODY_TEXT
        else:
            run = p_item.add_run()
            run.text = "• " + str(item)
            run.font.name = FONT_BODY
            run.font.size = Pt(10)
            run.font.color.rgb = BODY_TEXT
        p_item.space_after = Pt(5)

def create_table(slide, left, top, width, height, headers, data, col_widths=None):
    rows_count = len(data) + 1
    cols_count = len(headers)
    table_shape = slide.shapes.add_table(rows_count, cols_count, left, top, width, height)
    table = table_shape.table

    if col_widths and len(col_widths) == cols_count:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w

    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT_NAVY
        tf = cell.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.margin_top = tf.margin_bottom = Inches(0.05)
        p = tf.paragraphs[0]
        p.text = h_text
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        if c_idx > 0 and any(char.isdigit() or char in ['%', '±'] for char in str(data[0][c_idx])):
            p.alignment = PP_ALIGN.RIGHT

    for r_idx, row_values in enumerate(data):
        is_alt = (r_idx % 2 == 1)
        row_bg = ROW_ALT_BG if is_alt else WHITE
        for c_idx, val in enumerate(row_values):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg
            tf = cell.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = Inches(0.08)
            tf.margin_top = tf.margin_bottom = Inches(0.05)
            p = tf.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_BODY
            p.font.size = Pt(9.5)
            if "RandomForest" in str(row_values[0]):
                p.font.bold = True
                p.font.color.rgb = ACCENT_BLUE if c_idx == 0 else DARK_TEXT
            else:
                p.font.color.rgb = BODY_TEXT

            if c_idx > 0 and any(char.isdigit() or char in ['%', '±'] for char in str(val)):
                p.alignment = PP_ALIGN.RIGHT

    return table_shape

def set_speaker_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text.strip()


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title & Administrative Details
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, DARK_BG)

    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(3.8), Inches(0.42))
    pill.fill.solid()
    pill.fill.fore_color.rgb = DARK_CARD
    pill.line.color.rgb = ACCENT_BLUE
    pill.line.width = Pt(1)
    ptf = pill.text_frame
    ptf.margin_top = ptf.margin_bottom = ptf.margin_left = ptf.margin_right = 0
    pp = ptf.paragraphs[0]
    pp.text = "MAIN PROJECT – 2ND REVIEW  |  MCA 2026"
    pp.alignment = PP_ALIGN.CENTER
    pp.font.name = FONT_HEADING
    pp.font.size = Pt(11)
    pp.font.bold = True
    pp.font.color.rgb = ACCENT_BLUE

    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.2))
    ttf = tbox.text_frame
    ttf.word_wrap = True
    ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0

    p_title = ttf.paragraphs[0]
    p_title.text = "GeoPoverty AI: An Explainable Multimodal AI System for Area-Level Deprivation Estimation Using Satellite Imagery"
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE
    p_title.space_after = Pt(12)

    p_sub = ttf.add_paragraph()
    p_sub.text = "Integrating Sentinel-2, VIIRS Nightlights, and GHSL Settlement Telemetry with Spatial ML and TreeSHAP Explainability"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = LIGHT_MUTED

    card_w = Inches(2.65)
    card_h = Inches(2.0)
    card_top = Inches(4.5)

    meta_items = [
        ("Candidate Name", "Dithesh Ashlesh Kumar", "Master of Computer Applications", ACCENT_BLUE),
        ("University Seat No.", "4JK25MC036", "Batch 2024–2026", ACCENT_TEAL),
        ("Project Guide", "Mr. Sunith Kumar T", "Department of Computer Applications", ACCENT_EMERALD),
        ("Evaluation Date", "November 2026", "Main Project 2nd Review Stage", ACCENT_AMBER)
    ]

    for idx, (m_label, m_val, m_sub, m_col) in enumerate(meta_items):
        m_left = Inches(1.0) + idx * (card_w + Inches(0.24))
        m_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, m_left, card_top, card_w, card_h)
        m_card.fill.solid()
        m_card.fill.fore_color.rgb = DARK_CARD
        m_card.line.color.rgb = DARK_BORDER

        bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, m_left, card_top, card_w, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = m_col
        bar.line.fill.background()

        mtb = s1.shapes.add_textbox(m_left + Inches(0.18), card_top + Inches(0.22), card_w - Inches(0.36), card_h - Inches(0.35))
        mtf = mtb.text_frame
        mtf.word_wrap = True
        mtf.margin_left = mtf.margin_top = mtf.margin_right = mtf.margin_bottom = 0

        p1 = mtf.paragraphs[0]
        p1.text = m_label.upper()
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(9.5)
        p1.font.bold = True
        p1.font.color.rgb = m_col
        p1.space_after = Pt(4)

        p2 = mtf.add_paragraph()
        p2.text = m_val
        p2.font.name = FONT_HEADING
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = WHITE
        p2.space_after = Pt(4)

        p3 = mtf.add_paragraph()
        p3.text = m_sub
        p3.font.name = FONT_BODY
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = LIGHT_MUTED

    set_speaker_notes(s1, """
Respected Chairperson, esteemed members of the departmental review panel, and my guide Mr. Sunith Kumar T, good morning. 

I am Dithesh Ashlesh Kumar, USN 4JK25MC036. Today, I present the 2nd Review of my MCA Main Project: 'GeoPoverty AI: An Explainable Multimodal AI System for Area-Level Deprivation Estimation Using Satellite Imagery'.

Traditional ground socioeconomic surveys (such as the Census or NFHS) are conducted only every 5 to 10 years, costing hundreds of crores and experiencing severe latency. GeoPoverty AI addresses this gap by utilizing free, high-revisit satellite remote sensing telemetry to estimate district-level socioeconomic deprivation across India continuously.

In this 2nd Review presentation, I will strictly adhere to the departmental guidelines covering:
1. Software Design Specification (SDS): System architecture, 9-table schema, and sequence workflows.
2. Methodology: NFHS-5 ground truth index, 4 core satellite features, and our spatial leakage-free validation framework.
3. Performance Metrics: Leave-One-State-Out (LOSO) benchmarks across 564 districts, ablation studies, and qualitative decision-support features.
""")

    # =========================================================================
    # SLIDE 2: SDS – Architectural Design & System Modularity
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, LIGHT_BG)
    add_header(s2, "Software Design Specification (SDS)  |  Architecture & Modularity", "6-Tier Modular Architecture & System Decomposition", 2)

    col_w = Inches(3.68)
    col_h = Inches(5.4)
    top_pos = Inches(1.5)

    create_bullet_card(s2, Inches(0.8), top_pos, col_w, col_h,
        "Data & Preprocessing Tiers",
        [
            ("Tier 1: Multi-Source Remote Sensing", "Ingests Sentinel-2 MSI (vegetation indices), NOAA/NASA VIIRS DNB (nightlights), and JRC GHSL (built-up & population grids)."),
            ("Ground Truth Benchmarking", "Ingests National Family Health Survey (NFHS-5) 5-dimensional socioeconomic survey indicators across 564 matched districts."),
            ("Tier 2: Pipeline Feature Fusion", "Combines satellite indicators into a unified district feature vector with strict temporal anchoring (2019-11-01 to 2021-04-30)."),
            ("Leakage-Free Preprocessing", "Pipeline-contained training-fold median imputation and 3.0×IQR outlier filtering to ensure zero cross-fold snooping.")
        ]
    )

    create_bullet_card(s2, Inches(4.82), top_pos, col_w, col_h,
        "Modeling & Explainability Tiers",
        [
            ("Tier 3: Spatial ML Inference", "Deploys Random Forest (Best Model), XGBoost, Gradient Boosting, and Ridge regression models trained with spatial CV."),
            ("Spatial Generalization Engine", "Employs Leave-One-State-Out (LOSO) 31-fold cross-validation to assess generalization on entirely unseen states."),
            ("Tier 4: Game-Theoretic XAI", "Implements TreeSHAP (SHapley Additive exPlanations) for global feature rankings and local district waterfall attributions."),
            ("Spatial Clustering (DBSCAN)", "Haversine metric density clustering (150 km radius) to discover continuous geographic deprivation hot zones.")
        ]
    )

    create_bullet_card(s2, Inches(8.84), top_pos, col_w, col_h,
        "Application & Presentation Tiers",
        [
            ("Tier 5: Backend & REST APIs", "Lightweight Flask 3.0.3 microframework managing REST endpoints (/api/districts, /api/predict, /api/simulate)."),
            ("Database & Caching", "SQLite 3 relational engine (sqlite3 / schema.sql) for fast retrieval of district metrics and precomputed out-of-state scores."),
            ("Tier 6: Web Presentation", "Interactive Leaflet.js choropleth GIS interface with real-time district drilldowns, Chart.js XAI plots, and scenario simulators."),
            ("Decoupling & Modularity", "Clean separation of concerns: client-side UI communicates exclusively via JSON contracts; backend pipelines run headless.")
        ]
    )

    set_speaker_notes(s2, """
Moving to the Software Design Specification: Slide 2 illustrates our 6-Tier Modular Architecture. 

To ensure industrial-grade software engineering, we enforced strict separation of concerns, high cohesion, and low coupling across the layers:
- Tier 1: Ingests raw multi-sensor telemetry (Sentinel-2, VIIRS nightlights, and GHSL human settlement rasters), anchored strictly to the NFHS-5 survey period (Nov 2019 to Apr 2021).
- Tier 2: Preprocessing and Feature Fusion encapsulated inside Scikit-learn pipelines. All imputation and IQR outlier boundaries are computed solely on training partitions.
- Tier 3: Predictive Modeling & Spatial Inference, hosting frozen models validated via Leave-One-State-Out partitioning.
- Tier 4: Explainability & Analytics, hosting TreeSHAP for exact game-theoretic attributions and Haversine DBSCAN for trans-boundary cluster detection.
- Tier 5: Application Backend powered by Flask 3.0.3 and SQLite 3 relational storage.
- Tier 6: Client Presentation using Leaflet.js, Bootstrap 5, and Chart.js.

The UI communicates with the backend exclusively through defined REST JSON interfaces, enabling headless ML execution or independent UI redesign without breaking model pipelines.
""")

    # =========================================================================
    # SLIDE 3: SDS – Component Design & Relational Schema / Data Modeling
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, LIGHT_BG)
    add_header(s3, "Software Design Specification (SDS)  |  Component & Data Design", "Relational Schema Modeling & Subsystem Interfaces", 3)

    schema_headers = ["Table Name", "Primary Key", "Foreign Key", "Purpose & Logical Scope"]
    schema_rows = [
        ["areas", "area_id", "None", "564 Indian districts, states, coordinates, GAUL boundary data"],
        ["ground_truth", "gt_id", "area_id", "NFHS-5 composite deprivation target and 5 raw indicator metrics"],
        ["geospatial_features", "feature_id", "area_id", "Engineered 4-feature satellite telemetry (NDVI, NTL, Built-up, Pop)"],
        ["satellite_data", "sat_id", "area_id", "Raster metadata, acquisition windows, cloud coverage fractions"],
        ["predictions", "prediction_id", "area_id", "Out-of-state predicted deprivation score, tertile categories"],
        ["explanations", "explain_id", "prediction_id", "SHAP base values, per-feature local attributions, feature ranks"],
        ["model_results", "result_id", "None", "LOSO benchmark metrics (R2, MAE, RMSE, Spearman rho, Top-20%)"],
        ["users / audit_logs", "user_id / id", "users.user_id", "RBAC roles, authentication credentials, and system audit logs"]
    ]
    col_w_s3 = [Inches(1.8), Inches(1.1), Inches(1.2), Inches(2.9)]
    create_table(s3, Inches(0.8), Inches(1.5), Inches(7.0), Inches(5.4), schema_headers, schema_rows, col_w_s3)

    create_bullet_card(s3, Inches(8.0), Inches(1.5), Inches(4.5), Inches(5.4),
        "Component Architecture & Integrity",
        [
            ("District Relational Pivot", "The 'areas' table acts as the unified primary anchor (area_id), guaranteeing 1:1 alignment between remote sensing telemetry and survey truth."),
            ("State-Contained Boundary Match", "GAUL 2015 geometries reconciled with NFHS-5 2019-21 districts using strict state-contained fuzzy matching (564 districts across 31 states with 0 cross-state leakage)."),
            ("Dual Operational Modes", "Features 'real' mode (validated empirical records) and 'demo' mode (synthetic sandbox with data_mode='demo' flag for zero-side-effect testing)."),
            ("PredictionService Component", "Singleton inference engine loading frozen serialized models ('best_model.pkl') with strictly enforced input schemas."),
            ("ExplainabilityService Component", "On-demand and pre-cached SHAP TreeExplainer delivering exact local feature attributions within 15 ms per district.")
        ]
    )

    set_speaker_notes(s3, """
Slide 3 details our component design and relational data model.

Our database schema is organized into 9 normalized tables within SQLite 3:
- The 'areas' table serves as the primary relational pivot via 'area_id', maintaining official boundaries, canonical names, and centroid coordinates for 564 districts.
- 'ground_truth' stores the official NFHS-5 survey metrics and our composite index.
- 'geospatial_features' holds the 4 satellite indicators.
- 'predictions' and 'explanations' record our out-of-state model predictions, tertile classifications (Low, Medium, High), and SHAP attribution vectors.
- 'users' and 'audit_logs' enforce Role-Based Access Control (Admin, Researcher, Public).

A critical architectural feature is our state-contained boundary matching: because FAO GAUL 2015 boundaries predate some district splits in NFHS-5 (2019-21), we enforced strict in-state entity resolution (with Telangana/Andhra bifurcation as the only cross-state exception). This yielded 564 matched districts across 31 states with zero false cross-state matches.
Furthermore, the system supports dual operating modes: Real-World Mode for verified empirical analytics, and Demo Mode for sandboxed testing.
""")

    # =========================================================================
    # SLIDE 4: SDS – Interaction & Sequence Workflows
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, LIGHT_BG)
    add_header(s4, "Software Design Specification (SDS)  |  Behavioral Workflows", "System Interaction & Sequence Workflows", 4)

    flow_step_w = Inches(2.78)
    flow_step_h = Inches(3.2)
    flow_top = Inches(1.5)

    steps = [
        ("1. Client Request", "User / Policymaker", [
            "User browses Leaflet map and clicks target district or adjusts feature sliders in Policy Simulator.",
            "Asynchronous HTTP fetch request dispatched to Flask API endpoint (/api/predict or /api/districts).",
            "Session validation and request payload validation against configured schema."
        ], ACCENT_BLUE),
        ("2. Backend Routing", "Flask Application Tier", [
            "Controller validates inputs, checks session permissions, and queries SQLite database.",
            "Extracts canonical district telemetry or receives live slider overrides.",
            "Packages aligned feature vector matching feature_columns.json order."
        ], ACCENT_PURPLE),
        ("3. Pipeline Execution", "ML & XAI Engine", [
            "Loads frozen Scikit-learn Pipeline (best_model.pkl).",
            "Executes training-fold median imputation and 3.0×IQR clipping.",
            "Generates continuous deprivation score [0, 1] and tertile label (Low/Med/High).",
            "Executes TreeSHAP to calculate additive feature contributions."
        ], ACCENT_AMBER),
        ("4. Response & Render", "Client Presentation", [
            "JSON payload returned: predicted score, 80% empirical prediction interval, and SHAP vectors.",
            "Leaflet map dynamically color-codes district polygon.",
            "Chart.js generates waterfall attribution plot.",
            "Optional: One-click PDF policy dossier generation."
        ], ACCENT_EMERALD)
    ]

    for idx, (title, sub, bullets, col) in enumerate(steps):
        s_left = Inches(0.8) + idx * (flow_step_w + Inches(0.2))
        add_card(s4, s_left, flow_top, flow_step_w, flow_step_h, bg_color=WHITE, border_color=CARD_BORDER)
        bar = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, s_left, flow_top, flow_step_w, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = col
        bar.line.fill.background()

        tb = s4.shapes.add_textbox(s_left + Inches(0.15), flow_top + Inches(0.12), flow_step_w - Inches(0.3), flow_step_h - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.space_after = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.name = FONT_BODY
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = MUTED_TEXT
        p2.space_after = Pt(8)

        for b in bullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.name = FONT_BODY
            pb.font.size = Pt(9)
            pb.font.color.rgb = BODY_TEXT
            pb.space_after = Pt(4)

    b_card_w = Inches(5.75)
    b_card_h = Inches(2.0)
    b_top = Inches(4.9)

    create_bullet_card(s4, Inches(0.8), b_top, b_card_w, b_card_h,
        "System Activity Flow (Swimlane Summary)",
        [
            ("Exploration Activity", "Map Click → Retrieve Area Coordinates → Fetch Out-of-State Prediction → Load SHAP Breakdown."),
            ("What-If Simulation", "Parameter Slider Adjustment → Forward Inference Pipeline → Differential Score Computation → Impact Delta.")
        ]
    )

    create_bullet_card(s4, Inches(6.78), b_top, b_card_w, b_card_h,
        "User Roles & Access Control Workflows",
        [
            ("Policy Maker & Researcher", "Full access to choropleth GIS, district deep-dives, SHAP waterfall plots, and PDF dossier exports."),
            ("Public & Admin Roles", "Public: view-only high-level deprivation maps; Admin: model retraining trigger, audit log inspection, user management.")
        ]
    )

    set_speaker_notes(s4, """
Slide 4 illustrates our interaction sequence and activity workflows.

When a user interacts with the application—whether clicking a district polygon on the Leaflet map or testing hypothetical scenarios in our Policy Simulator:
1. The client dispatches an asynchronous HTTP request to Flask endpoints such as /api/predict or /api/districts/<id>.
2. The controller validates incoming payloads against feature_columns.json, checks authorization, and queries the database.
3. The request is routed to our singleton PredictionService. The feature vector passes through our frozen Scikit-learn pipeline, executing median imputation and IQR clipping. The Random Forest model outputs the predicted deprivation score [0.0 to 1.0] and tertile category.
4. Concurrently, the ExplainabilityService generates additive TreeSHAP values.
5. A structured JSON payload is returned containing the score, empirical 80% prediction interval ([-0.1363, +0.1437] from LOSO residuals), and feature attributions. The UI asynchronously renders choropleth shading, Chart.js waterfall graphs, or triggers automated PDF dossier compilation.

This modular workflow ensures low response latency (<50 ms for live forward inference) while maintaining full traceability.
""")

    # =========================================================================
    # SLIDE 5: Model & Methodology – Multimodal Telemetry & NFHS-5 Target
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, LIGHT_BG)
    add_header(s5, "Explanation of Model / Methodology  |  Data Ingestion & Ground Truth", "Multimodal Telemetry & NFHS-5 Deprivation Target", 5)

    left_w = Inches(6.0)
    right_w = Inches(5.5)
    top_pos = Inches(1.5)

    create_bullet_card(s5, Inches(0.8), top_pos, left_w, Inches(5.4),
        "Multimodal Earth Observation Telemetry",
        [
            ("Sentinel-2 MSI Surface Reflectance", "Multispectral optical imagery at 30m native resolution. Cloud-masked using Scene Classification Layer (SCL 4-7); extracts District Mean NDVI = (NIR - Red) / (NIR + Red)."),
            ("NOAA/NASA VIIRS DNB", "Day/Night Band monthly composites at 500m resolution. Measures nocturnal luminosity (nW·cm⁻²·sr⁻¹), proxying electrification, commercial density, and infrastructure."),
            ("JRC GHSL (GHS-BUILT-S)", "Global Human Settlement Layer at 100m resolution (epoch 2020). Measures built-up surface percentage across district polygons."),
            ("JRC GHSL (GHS-POP)", "Spatial human settlement density grids at 100m resolution (epoch 2020), providing spatial population density (people/km²)."),
            ("Temporal Window Synchronization", "All satellite telemetry strictly anchored to the NFHS-5 survey period (2019-11-01 to 2021-04-30), eliminating temporal misalignment artifacts.")
        ]
    )

    create_bullet_card(s5, Inches(7.0), top_pos, right_w, Inches(5.4),
        "NFHS-5 Ground Truth Deprivation Index",
        [
            ("5 Core Deprivation Dimensions", "Formulated from official district factsheets of National Family Health Survey (NFHS-5, 2019–21) conducted by MoHFW / IIPS:"),
            ("1. Sanitation Deficit", "Percentage of households without access to improved sanitation facilities."),
            ("2. Energy Poverty", "Percentage of households lacking access to electricity."),
            ("3. Cooking Fuel Deprivation", "Percentage of households not using clean cooking fuel (LPG/biogas)."),
            ("4. Child Stunting", "Prevalence of stunting (height-for-age < -2 SD) in children under 5."),
            ("5. Maternal Anaemia", "Prevalence of anaemia among pregnant women aged 15–49 (< 11.0 g/dl)."),
            ("Composite Index Formulation", "Each raw indicator is min-max normalized across surveyed districts; composite score is the unweighted arithmetic mean: Deprivation = (1/5) Σ I_k^norm ∈ [0, 1]."),
            ("Sensitivity Validation", "Principal Component Analysis (PCA) reveals PC1 explains 49% variance and maintains Spearman rank correlation ρ = 0.996 with equal-weight index.")
        ]
    )

    set_speaker_notes(s5, """
Moving to Section 2: Model and Methodology on Slide 5.

Here we explain our data ingestion and ground truth formulation:
Ground Truth Target: We constructed a continuous deprivation score from official factsheets of the National Family Health Survey (NFHS-5, 2019-21), conducted by MoHFW and IIPS. We selected 5 indicators capturing multi-dimensional living standards:
1. Household sanitation deficit
2. Lack of electricity access
3. Lack of clean cooking fuel
4. Child stunting prevalence (< -2 SD height-for-age)
5. Maternal anaemia prevalence (< 11.0 g/dl)
Each indicator is min-max normalized, and the overall deprivation score is their unweighted arithmetic mean in [0, 1]. 
To validate this index against weighting bias, we performed PCA across all surveyed districts: PC1 explains 49% of total variance and has a 0.996 Spearman rank correlation with our equal-weight score. Leaving out any single indicator maintains rank correlations between 0.932 and 0.988, proving the composite is exceptionally stable.

Remote Sensing Inputs: We ingests 4 independent sensor streams:
- Sentinel-2 optical imagery at 30m resolution (SCL cloud-masked) for NDVI vegetation vigor.
- NOAA/NASA VIIRS nighttime radiance at 500m resolution for nocturnal luminosity.
- JRC GHSL built-up area and population density grids at 100m resolution (epoch 2020).
All data is strictly synchronized to the survey window (Nov 2019 to Apr 2021).
""")

    # =========================================================================
    # SLIDE 6: Model & Methodology – Algorithmic Pipeline & Feature Engineering
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, LIGHT_BG)
    add_header(s6, "Explanation of Model / Methodology  |  Feature Pipeline & Selection", "Algorithmic Pipeline & Feature Engineering Design", 6)

    add_stat_box(s6, Inches(0.8), Inches(1.5), Inches(2.78), Inches(1.1), "4", "Core Satellite Features", "Zero multicollinearity", ACCENT_BLUE)
    add_stat_box(s6, Inches(3.78), Inches(1.5), Inches(2.78), Inches(1.1), "564", "Evaluated Districts", "31 Indian States / UTs", ACCENT_TEAL)
    add_stat_box(s6, Inches(6.76), Inches(1.5), Inches(2.78), Inches(1.1), "3.0×IQR", "Outlier Clipping", "Fold-isolated thresholds", ACCENT_AMBER)
    add_stat_box(s6, Inches(9.74), Inches(1.5), Inches(2.78), Inches(1.1), "0.996", "Ground Truth Stability", "PCA PC1 rank correlation", ACCENT_EMERALD)

    card_top = Inches(2.8)
    card_h = Inches(4.1)

    create_bullet_card(s6, Inches(0.8), card_top, Inches(5.75), card_h,
        "Scikit-Learn Pipeline Architecture",
        [
            ("End-to-End Pipeline Packaging", "All data transformations, imputation, clipping, and estimators encapsulated in sklearn.pipeline.Pipeline to prevent data snooping."),
            ("Training-Fold Median Imputation", "Missing values imputed strictly using median statistics calculated on the training partition of each CV fold."),
            ("Fold-Contained Outlier Clipping", "Interquartile range [Q1 - 3×IQR, Q3 + 3×IQR] calculated exclusively on training partitions; validation districts clipped passively."),
            ("Deterministic Feature Order", "Feature vectors strictly aligned to models/feature_columns.json ('nightlight_mean', 'ndvi_mean', 'builtup_pct', 'population_density')."),
            ("Uncertainty Estimation", "Empirical prediction intervals from LOSO residuals: 80% [-0.136, +0.144] and 90% [-0.162, +0.190] around predicted score.")
        ]
    )

    create_bullet_card(s6, Inches(6.78), card_top, Inches(5.75), card_h,
        "Scientific Feature Selection & Ablation Rationale",
        [
            ("Why 4 Satellite Features?", "Extensive ablation proved 4 remote sensing features yield optimal spatial generalization without overfitting to localized sensor artifacts."),
            ("Ablation of Deep Vision (VGG-16)", "Exploratory CNN deep visual features (VGG-16) produced poor standalone R² (0.0702) and degraded tabular models (R² dropped from 0.3273 to 0.2879) due to domain shifts on satellite tiles; intentionally excluded."),
            ("Ablation of OSM Accessibility", "OpenStreetMap road/hospital distances exhibited extreme collinearity (|r| ≈ 0.96) and marginal R² change (±0.003), while lacking nationwide auditability; excluded to preserve scientific rigor."),
            ("Parsimonious Generalization", "Final model relies on verified, publicly replicable earth observation datasets with complete nationwide coverage.")
        ]
    )

    set_speaker_notes(s6, """
Slide 6 covers our algorithmic pipeline architecture and feature selection rationale.

Our feature pipeline is encapsulated entirely inside Scikit-learn Pipeline objects.
To prevent any data leakage:
- Imputation uses training-fold medians only.
- Outlier filtering applies a 3.0×IQR threshold computed exclusively on the training fold.
- Features are strictly ordered: 'nightlight_mean', 'ndvi_mean', 'builtup_pct', 'population_density'.

Now, an important technical decision we made: in early exploratory work, we evaluated deep visual features from a pretrained VGG-16 CNN and OpenStreetMap infrastructure accessibility metrics.
However, empirical ablation revealed two critical findings:
1. Deep visual features alone had an out-of-state R² of only 0.0702, and adding them to tabular models actually degraded R² from 0.3273 down to 0.2879 due to domain shift on 30m resolution satellite tiles.
2. OSM road, school, and hospital distance features suffered from extreme collinearity (|r| ≈ 0.96) and contributed virtually zero performance gain (ΔR² ≈ 0.003), while introducing unverified crowd-sourced data quality issues.

Therefore, following Occam's Razor and scientific best practices, we selected the parsimonious 4-feature satellite model, ensuring maximum nationwide auditability and stability.
""")

    # =========================================================================
    # SLIDE 7: Model & Methodology – Spatial Leakage Prevention & Validation
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, LIGHT_BG)
    add_header(s7, "Explanation of Model / Methodology  |  Spatial Validation Architecture", "Spatial Leakage Mitigation: Tobler's First Law & LOSO", 7)

    create_bullet_card(s7, Inches(0.8), Inches(1.5), Inches(5.75), Inches(5.4),
        "The Spatial Autocorrelation Hazard",
        [
            ("Tobler's First Law of Geography", "'Everything is related to everything else, but near things are more related than distant things.' Adjacent districts share climate, topography, infrastructure, and state policies."),
            ("The Flaw of Standard Random CV", "Randomly assigning districts to train/test folds places neighboring districts across both splits. The model memorizes regional patterns, producing artificially inflated metrics."),
            ("Quantifying the Inflation (+37.7%)", "Standard Random 5-Fold CV yields an artificially high R² of 0.4628 and MAE of 0.0779. This is spurious spatial leakage, NOT genuine model generalization."),
            ("Published Literature Failure", "Many published poverty mapping studies report inflated metrics due to random splits; GeoPoverty AI enforces strict spatial partitioning to guarantee peer-reviewed integrity.")
        ]
    )

    create_bullet_card(s7, Inches(6.78), Inches(1.5), Inches(5.75), Inches(5.4),
        "Rigorous Spatial Validation Architecture",
        [
            ("Primary: Leave-One-State-Out (LOSO)", "31 Iterations: In each fold, all districts belonging to an entire administrative state are held out. The model is trained on the other 30 states and evaluated on the unseen state."),
            ("Genuine Generalization Test", "Simulates real-world deployment where the model predicts deprivation for un-surveyed regions or future census boundaries without local ground-truth hints."),
            ("Displayed Scores Integrity", "Every district score displayed in the GeoPoverty AI web application is an out-of-state prediction from this honest LOSO process (True R² = 0.3360, MAE = 0.0883)."),
            ("Secondary: Repeated GroupKFold", "5-fold cross-validation grouped by state_name, repeated 5 times with distinct random seeds (RepeatedGroupKFold), reporting mean ± std."),
            ("Leakage-Free Guarantee", "Guarantees zero geographic, feature, or target leakage across all evaluation benchmarks.")
        ]
    )

    set_speaker_notes(s7, """
Slide 7 addresses one of the most critical methodological contributions of our work: Spatial Leakage Prevention.

In geospatial machine learning, standard random train/test splits are deeply flawed due to Tobler's First Law of Geography: 'Near things are more related than distant things'. Neighboring districts share similar climate, topography, infrastructure, and administrative schemes. 
If district A is in the training set and neighboring district B is in the test set, the model memorizes local geography rather than learning generalizable socioeconomic signatures.

When we ran standard random 5-fold cross-validation, the model produced an R² of 0.4628. However, this is an illusion of accuracy—an inflation of +37.7% due to spatial leakage.

To establish genuine generalization, we engineered a rigorous Leave-One-State-Out (LOSO) cross-validation framework across 31 administrative states:
- In each of the 31 iterations, all districts in an entire state are held out.
- The model trains on 30 states and predicts on the held-out state, simulating a real-world scenario where remote sensing is deployed over un-surveyed regions.
Every single district prediction in our system is an out-of-state estimate from this honest LOSO scheme.
""")

    # =========================================================================
    # SLIDE 8: Model & Methodology – Framework Implementation & Core Modules
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, LIGHT_BG)
    add_header(s8, "Explanation of Model / Methodology  |  Implementation Status", "Framework Implementation & Core Modules Built to Date", 8)

    m_w = Inches(3.68)
    m_h = Inches(2.55)
    row1_top = Inches(1.5)
    row2_top = Inches(4.3)

    modules = [
        ("Module 1: Ingestion & Matching", "preprocessing/name_keys.py", [
            "Fuzzy state-contained district matcher linking GAUL 2015 polygons to NFHS-5 survey.",
            "564 districts matched across 31 states with zero cross-state entity errors.",
            "Built-in temporal alignment to 2019-2021 survey."
        ], ACCENT_BLUE),
        ("Module 2: ML Inference Pipeline", "training/pipelines.py", [
            "Trained & serialized Random Forest, XGBoost, GBDT, and Ridge regressors.",
            "Pipeline-contained median imputation and 3.0×IQR outlier clipping.",
            "Outputs continuous score [0, 1] & tertile category."
        ], ACCENT_PURPLE),
        ("Module 3: Game-Theoretic XAI", "explainability/shap_explain.py", [
            "TreeSHAP integration computing exact Shapley feature attributions.",
            "Local waterfall plots explaining which indicators drove predictions up or down.",
            "Precomputed district caching in SQLite for instant retrieval."
        ], ACCENT_AMBER),
        ("Module 4: Spatial DBSCAN Engine", "backend/services/hotspot_service.py", [
            "Haversine metric density-based spatial clustering (150 km radius, min=3 districts).",
            "Identifies contiguous regional poverty hot zones across state borders.",
            "Interactive cluster visualization on Leaflet map."
        ], ACCENT_TEAL),
        ("Module 5: Decision Support Web App", "backend/app.py & api/routes.py", [
            "Flask 3.0.3 REST backend with SQLite 3 relational database (sqlite3 / schema.sql).",
            "Session-based RBAC authentication and full audit logging.",
            "Endpoints for live predictions and district lookups."
        ], ACCENT_EMERALD),
        ("Module 6: GIS UI & Policy Simulator", "frontend/ (HTML5 / Leaflet)", [
            "Interactive choropleth map with search, state filtering, and legend.",
            "What-If Policy Intervention Simulator testing infrastructure adjustments.",
            "One-click automated PDF policy dossier generation."
        ], ACCENT_ROSE)
    ]

    for idx, (m_title, m_file, m_bullets, m_col) in enumerate(modules):
        r_top = row1_top if idx < 3 else row2_top
        c_left = Inches(0.8) + (idx % 3) * (m_w + Inches(0.35))
        add_card(s8, c_left, r_top, m_w, m_h, bg_color=WHITE, border_color=CARD_BORDER)
        bar = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, c_left, r_top, m_w, Inches(0.05))
        bar.fill.solid()
        bar.fill.fore_color.rgb = m_col
        bar.line.fill.background()

        tb = s8.shapes.add_textbox(c_left + Inches(0.18), r_top + Inches(0.1), m_w - Inches(0.36), m_h - Inches(0.18))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = m_title
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(11.5)
        p1.font.bold = True
        p1.font.color.rgb = DARK_TEXT

        p2 = tf.add_paragraph()
        p2.text = m_file
        p2.font.name = FONT_BODY
        p2.font.size = Pt(8.5)
        p2.font.bold = True
        p2.font.color.rgb = m_col
        p2.space_after = Pt(4)

        for b in m_bullets:
            pb = tf.add_paragraph()
            pb.text = "• " + b
            pb.font.name = FONT_BODY
            pb.font.size = Pt(8.5)
            pb.font.color.rgb = BODY_TEXT
            pb.space_after = Pt(2)

    set_speaker_notes(s8, """
Slide 8 summarizes our framework implementation and the 6 core software modules built to date:
1. Data Harmonization Engine: Robust string-normalization and state-contained matching linking 564 districts.
2. Spatial ML Inference Pipeline: Scikit-learn pipelines with serialized model artifacts (best_model.pkl).
3. Game-Theoretic Explainability Engine: TreeSHAP implementation calculating exact local Shapley values precomputed in SQLite.
4. Spatial DBSCAN Analytics Engine: Density-based clustering using Haversine distance with a 150 km search radius, successfully finding contiguous poverty pockets regardless of administrative borders.
5. Decision-Support Backend: Flask 3.0.3, SQLite 3 relational database, RBAC authentication, and audit logs.
6. GIS Client & Policy Simulator: Leaflet.js choropleth visualization, interactive What-If scenario simulations, and automated ReportLab PDF dossier generation.

All 6 core modules are implemented, tested, and integrated into our functional prototype.
""")

    # =========================================================================
    # SLIDE 9: Performance Metrics – Benchmark Comparison & Model Selection
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, LIGHT_BG)
    add_header(s9, "Performance Metrics & Evaluation  |  Benchmark Comparison", "Quantitative Model Evaluation & Benchmark Results", 9)

    loso_headers = ["Model", "R²", "MAE", "RMSE", "Spearman ρ", "Cat. Acc", "Top-20% Rec."]
    loso_data = [
        ["Dummy (Mean)", "-0.0565", "0.1142", "0.1371", "-0.6972", "31.9%", "0.9%"],
        ["Ridge", "0.2700", "0.0914", "0.1140", "0.5542", "48.6%", "32.7%"],
        ["RandomForest ★", "0.3360", "0.0883", "0.1087", "0.5745", "54.6%", "46.0%"],
        ["GradientBoosting", "0.2951", "0.0912", "0.1120", "0.5539", "52.7%", "45.1%"],
        ["XGBoost", "0.3074", "0.0905", "0.1110", "0.5643", "52.7%", "46.9%"]
    ]
    t_widths = [Inches(1.8), Inches(0.85), Inches(0.85), Inches(0.85), Inches(1.1), Inches(0.95), Inches(1.1)]
    create_table(s9, Inches(0.8), Inches(1.5), Inches(7.5), Inches(2.3), loso_headers, loso_data, t_widths)

    chart_path = "models/figures/model_comparison.png"
    if os.path.exists(chart_path):
        s9.shapes.add_picture(chart_path, Inches(0.8), Inches(3.95), width=Inches(7.5), height=Inches(3.0))

    create_bullet_card(s9, Inches(8.5), Inches(1.5), Inches(4.03), Inches(5.45),
        "Model Selection & Key Findings",
        [
            ("Random Forest Outperforms All", "Selected as primary production model: achieves highest R² (0.3360), lowest MAE (0.0883), and lowest RMSE (0.1087) on strict out-of-state holdout."),
            ("22.7% Error Reduction", "Outperforms Dummy baseline by -22.7% MAE, confirming strong predictive signal from earth observation telemetry alone."),
            ("High Rank Correlation (ρ = 0.5745)", "Demonstrates strong monotonic ranking ability, critical for targeting priority districts for development aid."),
            ("Top-20% Recall (46.0%)", "Successfully identifies nearly half of India's most acutely deprived districts purely from satellite telemetry without any field surveys."),
            ("Tree Ensembles vs. Linear", "RandomForest and XGBoost capture non-linear vegetation-luminosity interactions that linear Ridge regression misses (R² 0.336 vs 0.270).")
        ]
    )

    set_speaker_notes(s9, """
Slide 9 presents our primary quantitative evaluation: the Leave-One-State-Out (LOSO) benchmark across 564 districts in 31 states.

As shown in the table and chart:
1. The Dummy Regressor (predicting the training mean of other states) yields an R² of -0.0565 and MAE of 0.1142.
2. Ridge Regression achieves an R² of 0.2700, showing clear predictive ability.
3. RandomForest Regressor clearly emerges as the champion model:
   - Out-of-state R² of 0.3360
   - MAE of 0.0883 (a 22.7% reduction in error vs. baseline)
   - RMSE of 0.1087
   - Spearman rank correlation of 0.5745
   - 3-class Categorical Accuracy of 54.6%
   - Top-20% Deprivation Recall of 46.0%
4. XGBoost and GradientBoosting also perform strongly (R² 0.3074 and 0.2951), but Random Forest's variance reduction across geographically diverse states makes it the most reliable.

Capturing 46% of the most deprived districts purely from orbital sensors demonstrates remarkable decision-support utility for resource allocation.
""")

    # =========================================================================
    # SLIDE 10: Performance Metrics – Feature Ablation & Spatial Leakage
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, LIGHT_BG)
    add_header(s10, "Performance Metrics & Evaluation  |  Ablation & Spatial Integrity", "Feature Ablation Study & Spatial Leakage Analysis", 10)

    abl_headers = ["Feature Configuration", "Features", "R²", "MAE", "RMSE", "Spearman ρ", "Top-20% Rec."]
    abl_data = [
        ["All 4 Features (Full)", "4", "0.3360", "0.0883", "0.1087", "0.5745", "46.0%"],
        ["Without Nightlight (VIIRS)", "3", "0.2894", "0.0911", "0.1125", "0.5430", "38.0%"],
        ["Without Built-Up (GHSL)", "3", "0.2529", "0.0939", "0.1153", "0.4978", "39.8%"],
        ["Without Population (GHSL)", "3", "0.1854", "0.0976", "0.1204", "0.4188", "28.3%"],
        ["Without Vegetation (NDVI)", "3", "0.1567", "0.0988", "0.1225", "0.4032", "37.2%"]
    ]
    abl_widths = [Inches(2.2), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.8), Inches(1.1), Inches(1.1)]
    create_table(s10, Inches(0.8), Inches(1.5), Inches(7.5), Inches(2.3), abl_headers, abl_data, abl_widths)

    f_chart_path = "models/figures/feature_group_importance.png"
    if os.path.exists(f_chart_path):
        s10.shapes.add_picture(f_chart_path, Inches(0.8), Inches(3.95), width=Inches(7.5), height=Inches(3.0))

    create_bullet_card(s10, Inches(8.5), Inches(1.5), Inches(4.03), Inches(5.45),
        "Ablation Insights & Leakage Proof",
        [
            ("Vegetation (NDVI) is #1 Driver", "Omitting Sentinel-2 NDVI causes the catastrophic collapse of R² from 0.3360 to 0.1567 (-53.4% drop), proving agricultural health is the strongest predictor of rural deprivation."),
            ("Population Density (#2 Driver)", "Removing GHSL population density cuts R² to 0.1854 (-44.8%), confirming rural settlement dispersion correlates heavily with multidimensional poverty."),
            ("Spatial Leakage Empirical Proof", "Random 5-Fold CV yields R² = 0.4628 vs LOSO R² = 0.3360. That +37.7% inflation proves the peril of spatial autocorrelation and validates our honest evaluation."),
            ("Residual Error Analysis", "Highest prediction errors occur in topographically distinct regions (Goa, Tripura, Meghalaya, Jharkhand) where dense forests mask settlement patterns.")
        ]
    )

    set_speaker_notes(s10, """
Slide 10 presents our systematic feature ablation study and spatial error breakdown.

By training models while selectively omitting one satellite feature at a time under Leave-One-State-Out evaluation:
- Omitting Sentinel-2 NDVI causes R² to plummet from 0.3360 to 0.1567—a 53.4% drop in predictive power. This proves that vegetation index, reflecting rural agricultural viability and green cover, is the single most vital predictor.
- Omitting GHSL population density cuts R² to 0.1854 (a 44.8% drop), showing that sparse settlement patterns strongly correlate with deprivation.
- Dropping built-up area and nighttime lights reduces R² to 0.2529 and 0.2894 respectively.

This establishes that all four features provide non-redundant, complementary signals.
On the right, we also show where residual errors are concentrated: predominantly northeastern and forested states (Goa, Tripura, Meghalaya, Jharkhand). In these regions, high canopy cover shields rural settlements from optical sensors, and micro-topography creates localized variance—providing clear avenues for our future spatial refinement.
""")

    # =========================================================================
    # SLIDE 11: Performance Metrics – Qualitative Evaluation & System Validation
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, LIGHT_BG)
    add_header(s11, "Performance Metrics & Evaluation  |  Qualitative Validation", "Qualitative Validation: XAI, Policy Simulator & GIS UI", 11)

    card_w11 = Inches(5.75)
    card_h11 = Inches(2.6)

    # Card 1 (Top Left): Interactive Map Screenshot
    c1 = add_card(s11, Inches(0.8), Inches(1.5), card_w11, card_h11, bg_color=WHITE, border_color=CARD_BORDER)
    img_map = "documentation/screenshots/02_interactive_map.png"
    if os.path.exists(img_map):
        s11.shapes.add_picture(img_map, Inches(0.9), Inches(1.6), width=Inches(3.2), height=Inches(2.4))
    tb1 = s11.shapes.add_textbox(Inches(4.2), Inches(1.6), Inches(2.25), Inches(2.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    p = tf1.paragraphs[0]
    p.text = "Interactive Choropleth GIS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT
    p.space_after = Pt(4)
    for txt in ["Choropleth map of 564 districts color-coded by tertile.", "Real-time state and category filtering.", "Hover metadata displaying exact predicted scores and uncertainty bounds."]:
        pb = tf1.add_paragraph()
        pb.text = "• " + txt
        pb.font.size = Pt(8.5)
        pb.font.color.rgb = BODY_TEXT
        pb.space_after = Pt(2)

    # Card 2 (Top Right): Policy Simulator Screenshot
    c2 = add_card(s11, Inches(6.78), Inches(1.5), card_w11, card_h11, bg_color=WHITE, border_color=CARD_BORDER)
    img_sim = "documentation/screenshots/03_whatif_simulator.png"
    if os.path.exists(img_sim):
        s11.shapes.add_picture(img_sim, Inches(6.88), Inches(1.6), width=Inches(3.2), height=Inches(2.4))
    tb2 = s11.shapes.add_textbox(Inches(10.18), Inches(1.6), Inches(2.25), Inches(2.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
    p = tf2.paragraphs[0]
    p.text = "What-If Policy Simulator"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT
    p.space_after = Pt(4)
    for txt in ["Simulates infrastructure interventions (e.g. rural electrification).", "Computes forward inference delta in real time.", "Aids targeted developmental fund allocations without live surveys."]:
        pb = tf2.add_paragraph()
        pb.text = "• " + txt
        pb.font.size = Pt(8.5)
        pb.font.color.rgb = BODY_TEXT
        pb.space_after = Pt(2)

    # Card 3 (Bottom Left): XAI Deep Dive Screenshot
    c3 = add_card(s11, Inches(0.8), Inches(4.3), card_w11, card_h11, bg_color=WHITE, border_color=CARD_BORDER)
    img_xai = "documentation/screenshots/04_district_deepdive_xai.png"
    if os.path.exists(img_xai):
        s11.shapes.add_picture(img_xai, Inches(0.9), Inches(4.4), width=Inches(3.2), height=Inches(2.4))
    tb3 = s11.shapes.add_textbox(Inches(4.2), Inches(4.4), Inches(2.25), Inches(2.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = 0
    p = tf3.paragraphs[0]
    p.text = "TreeSHAP Local Attributions"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT
    p.space_after = Pt(4)
    for txt in ["Additive feature explanations for every individual district.", "Explains why a district was rated High or Low deprivation.", "Validates domain socioeconomic intuition with transparent AI."]:
        pb = tf3.add_paragraph()
        pb.text = "• " + txt
        pb.font.size = Pt(8.5)
        pb.font.color.rgb = BODY_TEXT
        pb.space_after = Pt(2)

    # Card 4 (Bottom Right): Hotspot Detection & PDF Dossiers
    c4 = add_card(s11, Inches(6.78), Inches(4.3), card_w11, card_h11, bg_color=WHITE, border_color=CARD_BORDER)
    img_home = "documentation/screenshots/01_home_dashboard.png"
    if os.path.exists(img_home):
        s11.shapes.add_picture(img_home, Inches(6.88), Inches(4.4), width=Inches(3.2), height=Inches(2.4))
    tb4 = s11.shapes.add_textbox(Inches(10.18), Inches(4.4), Inches(2.25), Inches(2.4))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = tf4.margin_top = tf4.margin_right = tf4.margin_bottom = 0
    p = tf4.paragraphs[0]
    p.text = "Spatial Clusters & Executive Dossier"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT
    p.space_after = Pt(4)
    for txt in ["Haversine DBSCAN clusters contiguous regional deprivation pockets.", "Discovers trans-boundary deprivation corridors.", "Automated server-side PDF dossier export for field deployment."]:
        pb = tf4.add_paragraph()
        pb.text = "• " + txt
        pb.font.size = Pt(8.5)
        pb.font.color.rgb = BODY_TEXT
        pb.space_after = Pt(2)

    set_speaker_notes(s11, """
Slide 11 highlights our qualitative evaluation and decision-support capabilities:
1. Interactive Choropleth Map: Displays 564 districts classified into Low, Medium, and High deprivation tertiles, enabling state-level filtering, spatial queries, and dynamic hover overlays showing empirical 80% prediction intervals.
2. What-If Policy Intervention Simulator: Allows policymakers to adjust parameters (such as simulating a rural electrification drive or urban settlement expansion) and observe forward model predictions and score changes in real-time.
3. TreeSHAP Local Attributions: Provides transparent explanations. For instance, in East Godavari or Anantapur, SHAP shows exactly how high vegetation index and low nighttime luminosity push deprivation scores up or down, corroborating known economic geography.
4. Hotspot Detection & PDF Reporting: Haversine DBSCAN isolates continuous regional clusters (e.g. the tribal belt in Central India) across administrative state borders, and one-click PDF dossiers provide field officers with publication-ready reports.
""")

    # =========================================================================
    # SLIDE 12: Summary, Project Status & Roadmap for Final Review
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, DARK_BG)

    tb12 = s12.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(0.9))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    tf12.margin_left = tf12.margin_top = tf12.margin_right = tf12.margin_bottom = 0
    p_cat12 = tf12.paragraphs[0]
    p_cat12.text = "SUMMARY & ROADMAP  |  2ND REVIEW DELIVERABLES".upper()
    p_cat12.font.name = FONT_HEADING
    p_cat12.font.size = Pt(10)
    p_cat12.font.bold = True
    p_cat12.font.color.rgb = ACCENT_BLUE
    p_cat12.space_after = Pt(2)

    p_t12 = tf12.add_paragraph()
    p_t12.text = "Project Milestones, Deliverables & Roadmap for Final Review"
    p_t12.font.name = FONT_HEADING
    p_t12.font.size = Pt(22)
    p_t12.font.bold = True
    p_t12.font.color.rgb = WHITE

    c_left12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.75), Inches(4.5))
    c_left12.fill.solid()
    c_left12.fill.fore_color.rgb = DARK_CARD
    c_left12.line.color.rgb = DARK_BORDER
    bar_l12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.75), Inches(0.06))
    bar_l12.fill.solid()
    bar_l12.fill.fore_color.rgb = ACCENT_EMERALD
    bar_l12.line.fill.background()

    tb_dl = s12.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.25), Inches(4.1))
    tfdl = tb_dl.text_frame
    tfdl.word_wrap = True
    tfdl.margin_left = tfdl.margin_top = tfdl.margin_right = tfdl.margin_bottom = 0

    p = tfdl.paragraphs[0]
    p.text = "Completed Work & 2nd Review Deliverables"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_EMERALD
    p.space_after = Pt(10)

    delivered_items = [
        ("Complete SDS Documentation", "Architectural decomposition, 9-table relational database schema, and detailed UML workflows fully formalized."),
        ("Leakage-Free Spatial ML Engine", "Leave-One-State-Out (LOSO) 31-fold cross-validation pipeline trained, benchmarked, and frozen (RandomForest R² = 0.3360)."),
        ("XAI & Spatial Clustering Engines", "TreeSHAP additive attributions and 150 km Haversine DBSCAN hotspot detection fully operational."),
        ("Full-Stack Decision Support System", "Flask 3.0.3 backend, Leaflet interactive choropleth GIS, Policy Simulator, and automated PDF dossier export active."),
        ("Scientifically Audited Dataset", "564 Indian districts matched with zero cross-state errors; sensitivity-verified equal-weighted NFHS-5 ground truth.")
    ]
    for lead, body in delivered_items:
        pi = tfdl.add_paragraph()
        r1 = pi.add_run()
        r1.text = "✓ " + lead + ": "
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = WHITE
        r2 = pi.add_run()
        r2.text = body
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = LIGHT_MUTED
        pi.space_after = Pt(6)

    c_right12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.78), Inches(1.5), Inches(5.75), Inches(4.5))
    c_right12.fill.solid()
    c_right12.fill.fore_color.rgb = DARK_CARD
    c_right12.line.color.rgb = DARK_BORDER
    bar_r12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.78), Inches(1.5), Inches(5.75), Inches(0.06))
    bar_r12.fill.solid()
    bar_r12.fill.fore_color.rgb = ACCENT_BLUE
    bar_r12.line.fill.background()

    tb_dr = s12.shapes.add_textbox(Inches(7.03), Inches(1.7), Inches(5.25), Inches(4.1))
    tfdr = tb_dr.text_frame
    tfdr.word_wrap = True
    tfdr.margin_left = tfdr.margin_top = tfdr.margin_right = tfdr.margin_bottom = 0

    p = tfdr.paragraphs[0]
    p.text = "Planned Enhancements for Final Review"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)

    future_items = [
        ("Sub-District Spatial Disaggregation", "Investigate downscaling model predictions to tehsil/block level using 10m Sentinel-2 bands and high-resolution building footprints."),
        ("Dynamic Earth Engine Tile Server", "Integrate direct Google Earth Engine API tile serving to stream live false-color infrared and nightlight raster overlays directly onto the Leaflet map."),
        ("Multi-Year Longitudinal Drift Analysis", "Evaluate model temporal transferability across successive survey waves to track deprivation dynamics over time."),
        ("Cloud Deployment & Containerization", "Docker containerization and deployment to an institutional server with NGINX reverse proxy and automated backup schedules."),
        ("Comprehensive Final Project Dissertation", "Drafting complete project report and publishing empirical findings for academic review.")
    ]
    for lead, body in future_items:
        pi = tfdr.add_paragraph()
        r1 = pi.add_run()
        r1.text = "→ " + lead + ": "
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = WHITE
        r2 = pi.add_run()
        r2.text = body
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = LIGHT_MUTED
        pi.space_after = Pt(6)

    q_box = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.75))
    q_box.fill.solid()
    q_box.fill.fore_color.rgb = DARK_CARD
    q_box.line.color.rgb = ACCENT_AMBER
    qtf = q_box.text_frame
    qtf.margin_top = Inches(0.12)
    qp = qtf.paragraphs[0]
    qp.alignment = PP_ALIGN.CENTER
    r_qa = qp.add_run()
    r_qa.text = "Thank You! Questions and Feedback from the Panel are Welcome."
    r_qa.font.name = FONT_HEADING
    r_qa.font.size = Pt(14)
    r_qa.font.bold = True
    r_qa.font.color.rgb = ACCENT_AMBER

    set_speaker_notes(s12, """
In conclusion: for this 2nd Review milestone, we have accomplished all targeted deliverables:
- We completed the detailed Software Design Specification (SDS).
- We solved spatial entity reconciliation across 564 districts in 31 states.
- We formulated and sensitivity-validated the continuous NFHS-5 deprivation index.
- We proved the necessity of spatial cross-validation, demonstrating how random K-Fold inflates R² by +37.7%, and established an honest Leave-One-State-Out benchmark (RandomForest R² = 0.3360, MAE = 0.0883, Top-20% recall = 46.0%).
- We integrated TreeSHAP explainability, Haversine DBSCAN spatial clustering, and an interactive GIS decision-support portal.

For the Final Review phase, our roadmap includes sub-district spatial disaggregation, dynamic Earth Engine raster tile streaming, longitudinal drift analysis, and completing the comprehensive MCA dissertation.

Thank you for your valuable time and attention. I am now open to questions, feedback, and suggestions from the esteemed panel.
""")

    out_path = "GeoPovertyAI_2nd_Review_Presentation.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to: {out_path}")

if __name__ == "__main__":
    build_deck()
