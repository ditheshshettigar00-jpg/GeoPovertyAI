# Module 2: Google Earth Engine Setup

This module gets REAL satellite, nighttime-light, NDVI, and built-up data
flowing — replacing the synthetic demo data from Module 1.

**You must do the account/authentication steps yourself** (I cannot do this
from here — it needs your own Google identity), but everything after that
is just running the provided scripts.

---

## Step 1: Get Earth Engine access

1. Go to https://earthengine.google.com/ and sign in with a Google account.
2. Click "Sign Up" / "Get Started" and register for non-commercial use
   (free for students/research). This may take a few minutes to a day to
   be approved, depending on current Google policy — do this first, before
   anything else in this module, since it can be a blocking step.
3. Once approved, go to https://console.cloud.google.com/ and create a new
   Google Cloud project (or reuse an existing one). Note the **Project ID**
   — you'll need it below. The Earth Engine API should auto-enable for this
   project once you've registered it for Earth Engine at
   https://code.earthengine.google.com/register.

## Step 2: Install required packages

Already listed in `requirements.txt` from Module 1 (`earthengine-api`,
`geemap`). If you haven't already:

```bash
source venv/bin/activate
pip install earthengine-api geemap
```

## Step 3: Authenticate (one-time, per machine)

```bash
earthengine authenticate
```

This opens a browser window, asks you to sign in with the same Google
account, and grants a token stored locally (`~/.config/earthengine/`
on Linux/Mac, `%APPDATA%\earthengine` on Windows). You only need to do
this once per machine.

## Step 4: Verify it works

```bash
python3 -c "
import ee
ee.Initialize(project='YOUR_PROJECT_ID')
print(ee.Image('NASA/NASADEM_HGT/001').get('title').getInfo())
"
```

Replace `YOUR_PROJECT_ID` with the Cloud project ID from Step 1. If this
prints something without an error, authentication is working.

## Step 5: Set your project ID in this codebase

Open `geospatial/gee_config.py` (created below) and put your project ID in
the `EE_PROJECT_ID` variable, or set it as an environment variable:

```bash
export GEE_PROJECT_ID="your-project-id-here"
```

---

## What this module does NOT require

- No paid tier, no billing account (Earth Engine's non-commercial tier is
  free).
- No GPU.
- No downloading huge raw satellite files by hand — Earth Engine does the
  filtering/compositing on Google's servers; you only download the small,
  already-processed result (a district-sized image patch or a single mean
  value).

## If GEE approval is delayed or you want to keep working in the meantime

Everything built in Module 1 (demo data, database, and every module we
build after this one — fusion, training, SHAP, API, dashboard) already
works against the synthetic demo dataset. You can keep developing and
demoing the rest of the system, and swap in real data the moment your GEE
access is approved by re-running the scripts in this module.
