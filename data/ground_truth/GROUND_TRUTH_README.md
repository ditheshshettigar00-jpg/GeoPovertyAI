# Ground Truth Data (real socioeconomic target)

This is the one piece of Module 2 that is **not** Earth Engine — the
deprivation *target* has to come from an actual survey/census, not a
satellite. This note tells you exactly what's realistically available and
how to bring it into this pipeline.

## What is realistically accessible

- **Census of India, 2011** (most recent full census at the time of
  writing): district-level indicator tables (e.g., housing quality,
  household amenities, literacy) are published at
  https://censusindia.gov.in/ — you'll need to browse to the relevant
  table series (e.g., House Listing and Housing Census tables) and
  download district-level CSV/Excel files yourself; exact table numbers
  and layouts change over time on that site, so I'm not going to guess a
  specific deep link here — search the site's data portal for
  "District Census Handbook" or "Houselisting and Housing Data" for your
  chosen state.
- **NFHS (National Family Health Survey)** district-level fact sheets
  are published via the Ministry of Health & Family Welfare / IIPS, and
  summarized indicators are also mirrored on the DHS Program site
  (https://dhsprogram.com/). District-level *fact sheets* (PDF/summary
  tables) are publicly accessible; household-level microdata with precise
  coordinates generally requires a formal DHS Program data request and is
  usually released with location displacement for privacy, as discussed
  in the earlier design document.

## What target to actually build

For an MCA-project-realistic version:
1. Pick **one or a small number of established indicators** from a
   Census District Handbook or NFHS district fact sheet that plausibly
   reflect deprivation — e.g., % of households without access to
   improved sanitation, % without electricity, or a similar published
   composite.
2. If you want a **single combined score** rather than one raw indicator,
   the defensible approach is to min-max normalize 2-4 such published
   indicators to a 0-1 scale and average them — and document this
   combination method explicitly in your report as your own scoring
   choice, not as an official government index (unless you're using an
   already-published composite index as-is).
3. Save the result as a CSV with **exactly** these columns, so it drops
   straight into the fusion pipeline (Module 4):

```
area_id,area_name,state_name,deprivation_score,data_source,data_year
1,<district name matching your GAUL area_name>,<state>,0.42,"Census2011_Housing",2011
```

Place this file at `data/ground_truth/real_ground_truth.csv`.

**Important matching detail:** district names in Census/NFHS tables and in
the GAUL boundaries used in `fetch_district_boundaries.py` don't always
match character-for-character (e.g., spelling, "and" vs "&", spacing).
Module 4's fusion script will need a name-matching/cleanup step — flag any
districts that fail to match rather than silently dropping or
mismatching them.

## What I will NOT do

I won't fabricate a ready-made CSV of "real" district deprivation scores,
invent specific table numbers, or guess at numbers that look plausible —
that would risk you presenting fabricated data as real research findings,
which the project's own constraints explicitly rule out. This step
genuinely requires you to go pull the actual published table for your
chosen state/districts.
