# KW Collision Zone Heatmap - hooray it works

Heatmap go brrrrrr.....

## stuff I added

- `build_kw_heatmap.py`: makign the map u will see
- `app.py`: Streamlit app aka thing u see.
- `kw_heatmap/pipeline.py`: bunch of data process logic

## run like this

```bash
python -m pip install -r requirements.txt
```

```bash
python build_kw_heatmap.py
```

yes, some flags work, like:

```bash
python build_kw_heatmap.py --target-zones 50 --refresh-cache
```

Generated outputs shit out into:

- `artifacts/collisions_cleaned.csv`
- `artifacts/collisions_unmappable.csv`
- `artifacts/zones.geojson`
- `artifacts/zone_metrics.csv`
- `artifacts/build_summary.json`

GIS inputs appear in `cache/`.

Then start the Streamlit site after the build finishes:

```bash
streamlit run app.py
```

Should open automatically otherwise open
`http://localhost:8501`.

## model training

Run the training script after the heatmap pipeline has been built and `Traffic_Collisions_Updated.csv` is present:

```bash
python train_model.py
```

What it does:

1. Loads `Traffic_Collisions_Updated.csv`, drops `ACCIDENTDATE` and `zone_id`
2. Splits 80/20 stratified train/test
3. Trains `RandomForest` and `GradientBoosting` in sequence
4. Prints accuracy, ROC-AUC, classification report, and confusion matrix for each
5. Prints top 15 feature importances from Random Forest
6. Saves the best model (by ROC-AUC) to `artifacts/collision_model.pkl`, feature list to `artifacts/model_features.pkl`, and a summary JSON to `artifacts/model_summary.json`

## some things maybe to remember

- The map uses the raw collision CSV as the source of truth because the legacy
  processed CSV drops latitude and longitude.
- Collisions with zero coordinates are excluded from the map and reported in the
  UI.
- Zone boundaries are built from public drivable OSM streets plus the municipal
  boundary of Kitchener and Waterloo.
- The first build may take a few minutes because it downloads and polygonizes
  the street network; later builds can reuse the cached files in `cache/`.
