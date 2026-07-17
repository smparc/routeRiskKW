# routeRIskKW — KW Traffic Collision Risk Predictor

A machine learning project that predicts high-risk areas for traffic collisions in the Kitchener-Waterloo region. The system builds geographically balanced street zones from OSM data, assigns historical collision records to those zones, generates synthetic non-collision records to balance the dataset, trains classification models, and visualizes risk as an interactive heatmap.

**Team:** Matthew Park · Jonathan Rethish · Benjamin Liu · Hayden Azan · Heeseung Oh

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Pipeline Overview](#pipeline-overview)
5. [File Reference](#file-reference)
6. [Full Run Instructions](#full-run-instructions)
7. [Artifacts Reference](#artifacts-reference)
8. [Model Details](#model-details)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Notes and Known Limitations](#notes-and-known-limitations)

---

## Project Overview

Traditional traffic safety approaches rely on historical crash data alone to identify black spots, meaning interventions only happen after lives are lost. This project takes a proactive approach — building a model that predicts the probability of a collision occurring at a location given its current road conditions, time of day, weather, and other contextual features.

The Kitchener-Waterloo region is divided into approximately 50 geographically balanced zones derived from the public drivable street network. Each zone becomes a unit of prediction. A balanced training dataset is constructed by pairing real collision records (positive samples) with synthetically generated non-collision records (negative samples), and classification models are trained to distinguish between the two.

---

## Repository Structure

```
MSE-446-Project/
│
├── kw_heatmap/
│   ├── __init__.py
│   └── pipeline.py                  # Core zone-building and collision assignment logic
│
├── sklearn-env/                     # Python virtual environment (not committed)
│
├── artifacts/                       # Generated outputs (not committed)
│   ├── collisions_cleaned.csv
│   ├── collisions_unmappable.csv
│   ├── collisions_with_negatives.csv
│   ├── zones.geojson
│   ├── zone_metrics.csv
│   ├── build_summary.json
│   ├── collision_model.pkl
│   ├── model_features.pkl
│   └── model_summary.json
│
├── cache/                           # GIS cache (not committed)
│   ├── kw_boundary.geojson
│   └── kw_streets.geojson
│
├── app.py                           # Streamlit heatmap app
├── build_kw_heatmap.py              # Entry point for zone/collision pipeline
├── data_processing.py               # Feature cleaning and OHE encoding
├── negative_sampling.py             # Synthetic non-collision record generation
├── train_model.py                   # Model training and evaluation
│
├── Traffic_Collisions_280340447332117481.csv   # Raw source data
├── Traffic_Collisions_Updated.csv             # Fully processed ML-ready dataset
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** City of Kitchener Open Data — [Traffic Collisions](https://data.waterloo.ca/datasets/KitchenerGIS::traffic-collisions/about)

**Coverage:** Kitchener-Waterloo region, January 2015 – July 2022

**Raw size:** 8,928 records, 42 columns

**Mappable records** (valid coordinates): 8,598

**Key features used:**

| Column | Description |
|---|---|
| `ACCIDENTDATE` | Date and time of the collision |
| `ACCIDENT_WEEKDAY` | Day of the week |
| `ACCIDENTLOCATION` | Location type (intersection, non-intersection, etc.) |
| `LIGHT` | Lighting condition at time of collision |
| `ROADJURISDICTION` | Road authority (municipal, regional, etc.) |
| `TRAFFICCONTROL` | Type of traffic control present |
| `TRAFFICCONTROLCONDITION` | Operational state of the traffic control |
| `ENVIRONMENTCONDITION1` | Weather/environment condition |
| `XMLIMPORTNOTES` | Street name and location description |
| `LONGITUDE` / `LATITUDE` | Coordinates (used for zone assignment, dropped for model) |

---

## Pipeline Overview

```
Raw CSV
   │
   ▼
build_kw_heatmap.py          ← downloads OSM streets, builds ~50 street-bounded zones,
   │                            assigns each collision to a zone
   │
   ▼
artifacts/collisions_cleaned.csv
   │
   ▼
negative_sampling.py          ← generates synthetic non-collision records in raw format,
   │                            adds CRASH column (1 = collision, 0 = no collision)
   │
   ▼
artifacts/collisions_with_negatives.csv
   │
   ▼
data_processing.py            ← drops irrelevant columns, applies one-hot encoding
   │                            to all categorical features
   │
   ▼
Traffic_Collisions_Updated.csv
   │
   ▼
train_model.py                ← 80/20 stratified split, trains Random Forest +
   │                            Gradient Boosting, saves best model by ROC-AUC
   │
   ▼
artifacts/collision_model.pkl
   │
   ▼
app.py                        ← Streamlit app, loads zones.geojson + zone_metrics.csv,
                                 renders interactive risk heatmap
```

---

## File Reference

### `kw_heatmap/pipeline.py`

The core data pipeline. Contains all logic for:

- **Boundary fetching** — geocodes the municipal boundaries of Kitchener and Waterloo via OSM and dissolves them into a single study area polygon.
- **Street network fetching** — downloads the drivable street network from OSM using `osmnx`, filters out private/service roads, and clips it to the study area.
- **Block construction** — polygonizes the street network to create street-bounded city blocks, filters out blocks smaller than 250 m² and those outside the study area.
- **Zone building** — groups blocks into approximately 50 geographically balanced zones using a graph-based growing algorithm that seeds from block centroids and rebalances until zone areas are within ±50% of the median.
- **Collision assignment** — spatial joins each collision record to its containing zone. Records that fall just outside a zone boundary (within 150 m) are assigned to the nearest zone.
- **Metrics computation** — computes per-zone collision counts, collision density (per km²), and average collisions per year.
- **Validation** — checks row counts, date ranges, zone count, contiguity of zone assignments, and interior zone area balance before writing outputs.

### `build_kw_heatmap.py`

Entry point that orchestrates `kw_heatmap/pipeline.py`. Accepts CLI flags and writes all outputs to `artifacts/`. GIS boundary and street data are cached in `cache/` to avoid re-downloading on subsequent runs.

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--raw-csv` | `Traffic_Collisions_280340447332117481.csv` | Path to raw collision CSV |
| `--output-dir` | `artifacts` | Directory for generated outputs |
| `--cache-dir` | `cache` | Directory for cached GIS data |
| `--target-zones` | `50` | Target number of zones to build |
| `--refresh-cache` | `False` | Re-download OSM boundary and street data |

### `negative_sampling.py`

Generates synthetic non-collision records to address the class imbalance problem — the raw dataset contains only collisions, which would bias a model toward predicting everything as high-risk.

**How it works:**

- Reads `artifacts/collisions_cleaned.csv` (post zone assignment)
- For each zone, allocates negative samples proportionally to the number of real collisions in that zone — zones with more collisions get more negatives, reflecting higher traffic exposure
- Each negative row is assigned a random timestamp within the dataset's date range (2015–2022)
- Categorical features (`LIGHT`, `TRAFFICCONTROL`, etc.) are sampled from the empirical frequency distribution of real collisions within that zone, so generated rows reflect the actual road characteristics of their area
- Street notes (`XMLIMPORTNOTES`) are randomly drawn from real records in the same zone
- Adds a `CRASH` column: `1` for real collisions, `0` for synthetic non-collisions
- Falls back to global (non-zone-proportional) sampling if `collisions_cleaned.csv` is not yet available — re-run after `build_kw_heatmap.py` to get full zone-proportional behavior
- Output is saved to `artifacts/collisions_with_negatives.csv` in the same raw categorical format as the source data, ready for `data_processing.py` to encode

**Default ratio:** 1:1 (balanced). Configurable via the `ratio` parameter in `generate_negative_samples()`.

### `data_processing.py`

Cleans and encodes the combined collision + negative sample dataset for model training.

**What it does:**

- Reads `artifacts/collisions_with_negatives.csv`
- Drops columns that are irrelevant to prediction (coordinates, IDs, timestamps, legacy fields)
- Drops any row where any column contains `"Other"` (inconsistently reported catch-all values)
- Normalises `TRAFFICCONTROLCONDITION`: maps `"unknown"` → `"Not Applicable"`
- Applies one-hot encoding (OHE) via `sklearn.preprocessing.OneHotEncoder` to all 7 categorical feature columns
- Preserves the `CRASH` column (label) through the encoding step
- Saves the fully processed dataset to `Traffic_Collisions_Updated.csv`

**Columns dropped:**

`OBJECTID`, `ACCIDENTNUM`, `ACCIDENT_YEAR`, `ACCIDENT_MONTH`, `ACCIDENT_DAY`, `ACCIDENT_HOUR`, `ACCIDENT_MINUTE`, `ACCIDENT_SECOND`, `XCOORD`, `YCOORD`, `LONGITUDE`, `LATITUDE`, `COLLISIONTYPE`, `CLASSIFICATIONOFACCIDENT`, `IMPACTLOCATION`, `INITIALDIRECTIONOFTRAVELONE`, `INITIALDIRECTIONOFTRAVELTWO`, `INITIALIMPACTTYPE`, `INTTRAFFICCONTROL`, `LIGHTFORREPORT`, `THRULANENO`, `NORTHBOUNDDISOBEYCOUNT`, `SOUTHBOUNDDISOBEYCOUNT`, `PEDESTRIANINVOLVED`, `CYCLISTINVOLVED`, `MOTORCYCLISTINVOLVED`, `ENVIRONMENTCONDITION2`, `SELFREPORTED`, `LASTEDITEDDATE`, `CREATE_BY`, `CREATE_DATE`, `x`, `y`, `source_row`, `year`, `has_valid_coords`, `distance_to_zone_m`, `XMLIMPORTNOTES`

### `train_model.py`

Trains and evaluates two classification models on the processed dataset.

**What it does:**

1. Loads `Traffic_Collisions_Updated.csv`
2. Drops `ACCIDENTDATE` and `zone_id` (non-numeric / identifier columns)
3. Performs an 80/20 stratified train/test split
4. Trains a `RandomForestClassifier` and a `GradientBoostingClassifier` in sequence
5. Reports accuracy, ROC-AUC, classification report, and confusion matrix for each
6. Prints the top 15 feature importances from the Random Forest
7. Saves the best-performing model (by ROC-AUC) and its feature list to `artifacts/`

### `app.py`

Streamlit web application that visualizes collision risk across the KW region as an interactive heatmap. Loads `artifacts/zones.geojson` for zone boundaries and `artifacts/zone_metrics.csv` for per-zone collision statistics. Collisions with zero or missing coordinates are excluded from the map and reported separately in the UI.

---

## Full Run Instructions

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

### Step 1 — Build zone map and assign collisions

```bash
python build_kw_heatmap.py
```

Or with flags:

```bash
python build_kw_heatmap.py --target-zones 50 --refresh-cache
```

> The first run downloads the KW street network from OSM and may take a few minutes. Subsequent runs reuse files in `cache/`.

### Step 2 — Generate negative samples

```bash
python negative_sampling.py
```

### Step 3 — Preprocess and encode features

Update the first line of `data_processing.py` to read from the combined file:

```python
df = pd.read_csv("artifacts/collisions_with_negatives.csv")
```

Then run:

```bash
python data_processing.py
```

### Step 4 — Train the model

```bash
python train_model.py
```

### Step 5 — Launch the heatmap

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

---

## Artifacts Reference

| File | Produced by | Description |
|---|---|---|
| `artifacts/collisions_cleaned.csv` | `build_kw_heatmap.py` | Collision records with zone assignments |
| `artifacts/collisions_unmappable.csv` | `build_kw_heatmap.py` | Records excluded due to missing/zero coordinates |
| `artifacts/zones.geojson` | `build_kw_heatmap.py` | Zone boundary polygons for the heatmap |
| `artifacts/zone_metrics.csv` | `build_kw_heatmap.py` | Per-zone collision counts and density |
| `artifacts/build_summary.json` | `build_kw_heatmap.py` | Build validation and summary statistics |
| `artifacts/collisions_with_negatives.csv` | `negative_sampling.py` | Combined real + synthetic records with `CRASH` label |
| `Traffic_Collisions_Updated.csv` | `data_processing.py` | Fully OHE-encoded ML-ready dataset |
| `artifacts/collision_model.pkl` | `train_model.py` | Best trained model (by ROC-AUC) |
| `artifacts/model_features.pkl` | `train_model.py` | Feature list used by the saved model |
| `artifacts/model_summary.json` | `train_model.py` | Model evaluation summary |

---

## Model Details

**Models trained:** Random Forest, Gradient Boosting (via scikit-learn)

**Train/test split:** 80/20 stratified on `CRASH`

**Features:** All OHE-encoded categorical columns from `data_processing.py` (54 binary columns across 7 feature groups)

**Label:** `CRASH` — `1` = real collision, `0` = synthetic non-collision

**Selection criterion:** Best ROC-AUC on the held-out test set

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| ROC-AUC | Discrimination ability across all thresholds |
| Precision / Recall / F1 | Per-class performance breakdown |
| Confusion matrix | False positive / false negative breakdown |
| Feature importances | Which road conditions drive predictions (from Random Forest) |
| Precision@K / Recall@K | What fraction of future crashes fall in the top-K predicted risk zones |

---

## Notes and Known Limitations

- The heatmap uses the raw collision CSV as its source of truth because the processed CSV drops latitude and longitude. These are two separate data flows that share the same source file.
- Zone-proportional negative sampling requires `collisions_cleaned.csv` to be present (produced by `build_kw_heatmap.py`). If it is missing, `negative_sampling.py` falls back to global sampling automatically and prints a warning.
- `data_processing.py` drops any row containing `"Other"` in any column. Negative samples are generated without `"Other"` values to ensure none are filtered out during this step.
- The dataset covers 2015–2022. Timestamps for negative samples are drawn uniformly across this range. Time-of-day and seasonal frequency matching against real collision patterns is a potential future improvement.
- The first OSM build may take several minutes depending on network speed. Use `--refresh-cache` only when the street network or boundary data needs to be re-fetched.
