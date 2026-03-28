import pandas as pd
import numpy as np
from datetime import timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# INPUT: The cleaned CSV output by pipeline.py (post zone assignment).
#        If that file doesn't exist yet, set USE_RAW_FALLBACK = True and it
#        will run directly from the original raw CSV instead.
CLEANED_CSV   = "artifacts/collisions_cleaned.csv"
RAW_CSV       = "Traffic_Collisions_280340447332117481.csv"
OUTPUT_CSV    = "artifacts/collisions_with_negatives.csv"

# Ratio of negative samples to positive samples (1.0 = balanced 1:1)
NEGATIVE_RATIO = 1.0
RANDOM_STATE   = 42

# ---------------------------------------------------------------------------
# Categorical columns that data_processing.py OHE-encodes.
# Negatives need plausible string values for these — all other columns can
# be left as NaN since data_processing.py drops them anyway.
# "Other" is excluded because data_processing.py drops any row containing it.
# ---------------------------------------------------------------------------
CAT_COLS = [
    "ACCIDENT_WEEKDAY",
    "ACCIDENTLOCATION",
    "LIGHT",
    "ROADJURISDICTION",
    "TRAFFICCONTROL",
    "TRAFFICCONTROLCONDITION",
    "ENVIRONMENTCONDITION1",
]


def _empirical_values(series: pd.Series) -> tuple[list, list]:
    """
    Return (values, probabilities) from a Series, excluding 'Other'/'other'.
    Probabilities are normalised so they sum to 1.
    """
    counts = series.value_counts()
    counts = counts[~(counts.index.str.lower() == "other")]
    probs = counts / counts.sum()
    return probs.index.tolist(), probs.values.tolist()


def generate_negative_samples(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a DataFrame of synthetic non-collision rows in the same raw
    column format as df (pre-OHE categorical strings).

    Allocation strategy:
      - If zone_id is present: proportional per zone (busier zones get more
        negatives, reflecting higher traffic exposure).
      - If zone_id is absent: global sampling from the full dataset.
    """
    n_pos = len(df)
    n_neg = int(round(n_pos * NEGATIVE_RATIO))
    has_zones = "zone_id" in df.columns

    date_min = pd.to_datetime(df["ACCIDENTDATE"]).min()
    date_max = pd.to_datetime(df["ACCIDENTDATE"]).max()
    date_range_s = int((date_max - date_min).total_seconds())

    # Pre-compute empirical distributions for categorical cols (global)
    global_dists = {
        col: _empirical_values(df[col].dropna()) for col in CAT_COLS
    }

    def _make_block(source_df: pd.DataFrame, n: int,
                    zone_id=None, zone_area=None) -> pd.DataFrame:
        """Generate n negative rows drawn from source_df's distributions."""
        block = pd.DataFrame(index=range(n))

        # Random timestamps within dataset date range
        offsets = rng.integers(0, date_range_s + 1, size=n)
        block["ACCIDENTDATE"] = [
            date_min + timedelta(seconds=int(s)) for s in offsets
        ]

        # Categorical features sampled from empirical distribution
        for col in CAT_COLS:
            vals, probs = _empirical_values(source_df[col].dropna())
            if not vals:          # fallback to global if zone has no data
                vals, probs = global_dists[col]
            block[col] = rng.choice(vals, size=n, p=probs)

        # Street notes — randomly draw from real notes in the source pool
        notes = source_df["XMLIMPORTNOTES"].dropna().values
        if len(notes):
            block["XMLIMPORTNOTES"] = rng.choice(notes, size=n)

        # Zone metadata (only when zone_id is present)
        if zone_id is not None:
            block["zone_id"]       = zone_id
            block["zone_area_km2"] = zone_area

        return block

    # ------------------------------------------------------------------
    # Proportional per-zone allocation (preferred)
    # ------------------------------------------------------------------
    if has_zones:
        print("  zone_id found — using proportional per-zone negative sampling.")
        zone_counts    = df["zone_id"].value_counts()
        zone_neg_share = ((zone_counts / n_pos) * n_neg).round().astype(int)

        # Fix any rounding drift
        diff = n_neg - zone_neg_share.sum()
        if diff:
            zone_neg_share[zone_neg_share.idxmax()] += diff

        blocks = []
        for zone_id, n_zone_neg in zone_neg_share.items():
            if n_zone_neg <= 0:
                continue
            zone_df   = df[df["zone_id"] == zone_id]
            zone_area = float(zone_df["zone_area_km2"].iloc[0])
            blocks.append(_make_block(zone_df, n_zone_neg, zone_id, zone_area))
        negatives = pd.concat(blocks, ignore_index=True)

    # ------------------------------------------------------------------
    # Global fallback (zone map not yet merged into CSV)
    # ------------------------------------------------------------------
    else:
        print("  zone_id not found — using global negative sampling.")
        print("  Re-run after pipeline.py has been run to get per-zone sampling.")
        negatives = _make_block(df, n_neg)

    return negatives


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # ---- Load source data ----
    if os.path.exists(CLEANED_CSV):
        print(f"Loading {CLEANED_CSV} ...")
        df = pd.read_csv(CLEANED_CSV)
        print("  (Using pipeline-cleaned CSV with zone assignments)")
    else:
        print(f"{CLEANED_CSV} not found — falling back to raw CSV.")
        print(f"Loading {RAW_CSV} ...")
        df = pd.read_csv(RAW_CSV)
        print("  (Zone-proportional sampling unavailable until pipeline.py is run)")

    print(f"  Rows loaded: {len(df)}")

    # ---- Add CRASH column to positives ----
    df["CRASH"] = 1

    # ---- Generate negatives ----
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"\nGenerating {int(round(len(df) * NEGATIVE_RATIO))} negative samples ...")
    negatives = generate_negative_samples(df, rng)
    negatives["CRASH"] = 0

    # ---- Combine ----
    # Align columns so both DataFrames match, then shuffle
    combined = pd.concat([df, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ---- Save ----
    os.makedirs("artifacts", exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    n_pos = (combined["CRASH"] == 1).sum()
    n_neg = (combined["CRASH"] == 0).sum()
    print(f"\nDone.")
    print(f"  Collision rows  : {n_pos}")
    print(f"  No-collision rows: {n_neg}")
    print(f"  Total rows       : {len(combined)}")
    print(f"\nSaved to: {OUTPUT_CSV}")
    print("\nNext step: point data_processing.py at this file instead of")
    print("collisions_cleaned.csv, and make sure the CRASH column is not")
    print("in its columns_to_del list so it passes through the OHE step.")