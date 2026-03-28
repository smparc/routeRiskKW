# KW Collision Zone Heatmap

## Components

- `build_kw_heatmap.py` / `kw_heatmap/pipeline.py`: zone boundary construction and collision assignment logic
- `app.py`: Streamlit heatmap visualization
- `data_processing.py`: cleans the raw collision CSV and applies one-hot encoding to categorical features
- `negative_sampling.py`: generates synthetic non-collision records and combines them with the real collision data for model training
- `train_model.py`: trains and evaluates Random Forest and Gradient Boosting classifiers

---

## Full pipeline — run in this order

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Build zone map and assign collisions to zones

```bash
python build_kw_heatmap.py
```

Optional flags:

```bash
python build_kw_heatmap.py --target-zones 50 --refresh-cache
```

Outputs to `artifacts/`:

- `collisions_cleaned.csv` — collision records with zone assignments
- `collisions_unmappable.csv` — records excluded due to missing/zero coordinates
- `zones.geojson` — zone boundary polygons
- `zone_metrics.csv` — per-zone collision counts and density
- `build_summary.json` — build validation summary

GIS inputs are cached in `cache/`. The first build downloads and polygonizes the OSM street network for Kitchener-Waterloo and may take a few minutes. Later builds reuse the cache.

### 3. Generate negative samples

```bash
python negative_sampling.py
```

Reads `artifacts/collisions_cleaned.csv` (produced in step 2) and generates synthetic non-collision records in the same raw format as the real data. Negatives are allocated proportionally per zone — zones with more real collisions receive more negatives, reflecting higher traffic exposure in those areas. Each negative row is assigned a plausible combination of categorical features (light condition, road type, traffic control, etc.) sampled from the empirical distribution of real collisions in that zone.

Adds a `CRASH` column to both real (`1`) and synthetic (`0`) rows, then saves the combined dataset.

Output:

- `artifacts/collisions_with_negatives.csv` — balanced dataset ready for preprocessing

### 4. Preprocess and encode features

```bash
python data_processing.py
```

Reads `artifacts/collisions_with_negatives.csv`, drops unnecessary columns, applies one-hot encoding to all categorical features, and saves the processed result.

> **Note:** make sure `data_processing.py` reads from `artifacts/collisions_with_negatives.csv` and that `CRASH` is not in the `columns_to_del` list so it passes through as the label column.

Output:

- `Traffic_Collisions_Updated.csv` — fully processed and encoded dataset

### 5. Train the model

```bash
python train_model.py
```

Loads `Traffic_Collisions_Updated.csv`, drops `ACCIDENTDATE` and `zone_id`, and runs an 80/20 stratified train/test split. Trains Random Forest and Gradient Boosting classifiers in sequence and reports accuracy, ROC-AUC, classification report, and confusion matrix for each. Also prints the top 15 feature importances from Random Forest.

Saves the best model (by ROC-AUC) to `artifacts/`:

- `collision_model.pkl` — trained model
- `model_features.pkl` — feature list
- `model_summary.json` — evaluation summary

### 6. Launch the heatmap app

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

---

## Notes

- The heatmap uses the raw collision CSV as the source of truth because the processed CSV drops latitude and longitude.
- Collisions with zero or missing coordinates are excluded from the map and reported in the UI.
- Zone boundaries are derived from public drivable OSM streets clipped to the municipal boundaries of Kitchener and Waterloo.
- Negative sampling falls back to global (non-zone-proportional) sampling if `collisions_cleaned.csv` is not yet available. Re-run `negative_sampling.py` after step 2 to get the full zone-proportional behavior.