"""
Pipeline orchestration module for the Léarslán ML scoring pipeline.

Provides the single public entry point `run_pipeline()` that loads data,
engineers features, trains GBM models, validates them, assigns labels,
clusters areas, detects anomalies, and persists all artifacts to disk.

This module is pure Python with no Streamlit dependency. Caching (e.g.,
@st.cache_data) will be applied at the integration layer when wired into
the Streamlit dashboard in a future session.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from ml.anomaly_detector import detect_anomalies
from ml.clustering import cluster_areas
from ml.feature_engineering import engineer_features
from ml.risk_model import assign_labels, train_gbm_models, validate_models

logger = logging.getLogger(__name__)

# Paths to pre-merged CSVs produced by ml/prepare_data.py
_DATA_DIR = Path(__file__).parent / "data"
_COUNTY_CSV = _DATA_DIR / "county_merged.csv"
_ED_CSV = _DATA_DIR / "ed_merged.csv"

# Columns with zero variance across all rows — dropped before feature engineering
_ZERO_VARIANCE_COLS = ["rent_growth_pct", "pct_a_rated", "pct_bcd_rated", "avg_speed_kph"]

# The 15-column feature matrix fed to GBM training
FEATURE_COLS = [
    "avg_monthly_rent",
    "avg_income",
    "employment_rate",
    "unemployment_rate",
    "traffic_volume",
    "congestion_delay_minutes",
    "ber_avg_score",
    "est_annual_energy_cost",
    "rental_yield",
    "affordability_index",
    "rent_to_income_pct",
    "commute_cost_monthly",
    "true_cost_index",
    "energy_tax",
    "commute_to_rent_ratio",
]


def run_pipeline(granularity: str = "ed") -> tuple[pd.DataFrame, dict]:
    """
    Execute the full ML scoring pipeline.

    This is a pure Python function with no Streamlit dependency. Caching
    (e.g., @st.cache_data) will be applied at the integration layer when
    wired into the Streamlit dashboard in a future session.

    Args:
        granularity: "county" (N=26) or "ed" (N=255)

    Returns:
        (scored_df, models_dict)
        - scored_df: DataFrame with all raw, engineered, score, label,
          cluster, and anomaly columns
        - models_dict: {"risk_score": GBM, "livability_score": GBM, ...}

    Raises:
        FileNotFoundError: if source CSV is missing
        ValueError: if ed_type column missing at ED granularity
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    if granularity == "county":
        csv_path = _COUNTY_CSV
    else:
        csv_path = _ED_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {csv_path}. "
            "Run ml/prepare_data.py first to generate the merged files."
        )

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %s rows × %s cols", csv_path.name, *df.shape)

    # ── 2. Drop zero-variance columns ─────────────────────────────────────────
    for col in _ZERO_VARIANCE_COLS:
        if col in df.columns:
            logger.warning("Dropping zero-variance column: %s", col)
            df = df.drop(columns=[col])

    # ── 3. Validate ed_type for ED granularity ────────────────────────────────
    if granularity == "ed" and "ed_type" not in df.columns:
        raise ValueError(
            "ed_type column missing from ED DataFrame — required for TOPSIS filtering"
        )

    # ── 4. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df)
    logger.info("Feature engineering complete. Columns: %d", len(df.columns))

    # ── 5. GBM training ───────────────────────────────────────────────────────
    # Only include feature cols that are actually present after engineering
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    scored_df, models_dict, feature_names, targets_dict = train_gbm_models(df, feature_cols)
    logger.info("GBM training complete. Models: %s", list(models_dict.keys()))

    # ── 6. Model validation ───────────────────────────────────────────────────
    validate_models(models_dict, scored_df, feature_names, targets_dict)

    # ── 7. Score labels ───────────────────────────────────────────────────────
    scored_df = assign_labels(scored_df)

    # ── 8. Clustering ─────────────────────────────────────────────────────────
    scored_df, kmeans_model, cluster_scaler = cluster_areas(scored_df, granularity)
    logger.info("Clustering complete.")

    # ── 9. Anomaly detection ──────────────────────────────────────────────────
    scored_df = detect_anomalies(scored_df, granularity)
    logger.info("Anomaly detection complete.")

    # ── 10. Persist artifacts ─────────────────────────────────────────────────
    save_artifacts(
        models=models_dict,
        scored_df=scored_df,
        feature_cols=feature_names,
        kmeans_model=kmeans_model,
        cluster_scaler=cluster_scaler,
    )

    return scored_df, models_dict


def save_artifacts(
    models: dict,
    scored_df: pd.DataFrame,
    feature_cols: list[str],
    kmeans_model,
    cluster_scaler,
    output_dir: str = "ml/models",
) -> None:
    """
    Save all trained models and scored data to disk.

    Creates output_dir if it doesn't exist. Writes:
      - {score_name}_gbm.joblib for each of the 4 GBM models
      - kmeans_model.joblib and cluster_scaler.joblib
      - scored_df.parquet  (type-preserving, compact)
      - feature_columns.json

    Args:
        models: dict mapping score name to trained GBM model
        scored_df: fully scored and annotated DataFrame
        feature_cols: list of feature column names used for training
        kmeans_model: fitted KMeans instance
        cluster_scaler: fitted StandardScaler used for clustering
        output_dir: directory to write artifacts (created if absent)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # GBM models
    for score_name, model in models.items():
        path = out / f"{score_name}_gbm.joblib"
        joblib.dump(model, path)
        logger.info("Saved %s", path)

    # KMeans + scaler
    joblib.dump(kmeans_model, out / "kmeans_model.joblib")
    joblib.dump(cluster_scaler, out / "cluster_scaler.joblib")
    logger.info("Saved kmeans_model.joblib and cluster_scaler.joblib")

    # Scored DataFrame as Parquet
    parquet_path = out / "scored_df.parquet"
    scored_df.to_parquet(parquet_path, index=False)
    logger.info("Saved scored_df.parquet (%d rows)", len(scored_df))

    # Feature column list as JSON
    json_path = out / "feature_columns.json"
    json_path.write_text(json.dumps(feature_cols, indent=2))
    logger.info("Saved feature_columns.json (%d features)", len(feature_cols))


def load_artifacts(
    output_dir: str = "ml/models",
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Load previously saved models and scored data from disk.

    This enables the future Streamlit integration to skip retraining by
    loading pre-computed models and scores.

    Args:
        output_dir: directory containing saved artifacts

    Returns:
        (scored_df, models_dict, feature_cols)

    Raises:
        FileNotFoundError: if output_dir or expected files are missing
    """
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found: {out}. "
            "Run run_pipeline() first to generate artifacts."
        )

    # Feature columns
    json_path = out / "feature_columns.json"
    feature_cols: list[str] = json.loads(json_path.read_text())

    # GBM models
    score_names = ["risk_score", "livability_score", "transport_score", "affordability_score"]
    models_dict = {}
    for score_name in score_names:
        path = out / f"{score_name}_gbm.joblib"
        models_dict[score_name] = joblib.load(path)

    # Scored DataFrame
    scored_df = pd.read_parquet(out / "scored_df.parquet")

    logger.info(
        "Loaded artifacts from %s: %d rows, %d models, %d features",
        out, len(scored_df), len(models_dict), len(feature_cols),
    )

    return scored_df, models_dict, feature_cols
