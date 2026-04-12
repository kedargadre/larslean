"""
run_and_test.py
---------------
Self-contained test suite + pipeline runner for the Léarslán ML pipeline.
No pytest or hypothesis required — runs with plain Python.

Usage:
    python ml/run_and_test.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    marker = "✅" if condition else "❌"
    print(f"  {marker} {name}" + (f" — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Unit tests for feature_engineering
# ─────────────────────────────────────────────────────────────────────────────
section("1. Feature Engineering Tests")

from ml.feature_engineering import engineer_features, minmax_norm

# Test 1.1 — minmax_norm basic
s = pd.Series([0.0, 5.0, 10.0])
normed = minmax_norm(s)
check("minmax_norm: min=0, max=1", abs(normed.min()) < 1e-9 and abs(normed.max() - 1.0) < 1e-9)

# Test 1.2 — minmax_norm zero variance returns 0.5
s_const = pd.Series([3.5, 3.5, 3.5, 3.5])
check("minmax_norm: zero variance → 0.5", (minmax_norm(s_const) == 0.5).all())

# Test 1.3 — engineer_features preserves original columns
df_raw = pd.DataFrame({
    "avg_monthly_rent": [1000.0, 1500.0, 2000.0],
    "avg_income": [36000.0, 42000.0, 54000.0],
    "congestion_delay_minutes": [5.0, 10.0, 20.0],
    "est_annual_energy_cost": [2400.0, 2700.0, 3000.0],
})
df_eng = engineer_features(df_raw)
original_cols_preserved = all(c in df_eng.columns for c in df_raw.columns)
check("engineer_features: original columns preserved", original_cols_preserved)

# Test 1.4 — affordability_index formula
expected_ai = (df_raw["avg_income"] / 12) / df_raw["avg_monthly_rent"]
check("engineer_features: affordability_index formula correct",
      np.allclose(df_eng["affordability_index"], expected_ai))

# Test 1.5 — rent_to_income_pct formula
expected_rip = (df_raw["avg_monthly_rent"] * 12) / df_raw["avg_income"] * 100
check("engineer_features: rent_to_income_pct formula correct",
      np.allclose(df_eng["rent_to_income_pct"], expected_rip))

# Test 1.6 — commute_cost_monthly formula
expected_cc = df_raw["congestion_delay_minutes"] * 2 * 22
check("engineer_features: commute_cost_monthly formula correct",
      np.allclose(df_eng["commute_cost_monthly"], expected_cc))

# Test 1.7 — energy_tax formula
expected_et = df_raw["avg_monthly_rent"] + (df_raw["est_annual_energy_cost"] / 12)
check("engineer_features: energy_tax formula correct",
      np.allclose(df_eng["energy_tax"], expected_et))

# Test 1.8 — true_cost_index in [0, 100]
check("engineer_features: true_cost_index in [0,100]",
      (df_eng["true_cost_index"] >= 0).all() and (df_eng["true_cost_index"] <= 100).all())

# Test 1.9 — zero rent denominator guard (no NaN/Inf)
df_zero_rent = df_raw.copy()
df_zero_rent.loc[0, "avg_monthly_rent"] = 0.0
df_zero_eng = engineer_features(df_zero_rent)
check("engineer_features: zero rent → no NaN/Inf",
      not df_zero_eng[["affordability_index", "commute_to_rent_ratio"]].isnull().any().any() and
      not np.isinf(df_zero_eng[["affordability_index", "commute_to_rent_ratio"]].values).any())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GBM target formula tests
# ─────────────────────────────────────────────────────────────────────────────
section("2. GBM Target Formula Tests")

from ml.risk_model import _create_targets, assign_labels, train_gbm_models

# Build a minimal feature df (need all 15 feature cols)
FEATURE_COLS = [
    "avg_monthly_rent", "avg_income", "employment_rate", "unemployment_rate",
    "traffic_volume", "congestion_delay_minutes", "ber_avg_score",
    "est_annual_energy_cost", "rental_yield",
    "affordability_index", "rent_to_income_pct", "commute_cost_monthly",
    "true_cost_index", "energy_tax", "commute_to_rent_ratio",
]

df_base = pd.DataFrame({
    "avg_monthly_rent":        [800.0, 1500.0, 2500.0, 3000.0],
    "avg_income":              [32000.0, 42000.0, 55000.0, 60000.0],
    "employment_rate":         [0.65, 0.72, 0.78, 0.80],
    "unemployment_rate":       [0.35, 0.28, 0.22, 0.20],
    "traffic_volume":          [8000.0, 20000.0, 50000.0, 85000.0],
    "congestion_delay_minutes":[3.0, 8.0, 18.0, 28.0],
    "ber_avg_score":           [4.5, 4.0, 3.5, 3.2],
    "est_annual_energy_cost":  [3300.0, 2800.0, 2400.0, 2100.0],
    "rental_yield":            [5.5, 5.0, 4.5, 3.5],
})
df_base = engineer_features(df_base)

targets = _create_targets(df_base, FEATURE_COLS)

check("_create_targets: returns 4 scores",
      set(targets.keys()) == {"risk_score", "livability_score", "transport_score", "affordability_score"})

for score_name, series in targets.items():
    in_range = (series >= 0).all() and (series <= 100).all()
    check(f"_create_targets: {score_name} in [0,100]", in_range,
          f"min={series.min():.1f} max={series.max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Score label tests
# ─────────────────────────────────────────────────────────────────────────────
section("3. Score Label Classification Tests")

df_labels_test = df_base.copy()
df_labels_test["risk_score"]         = pd.Series([10.0, 50.0, 80.0, 33.0])
df_labels_test["affordability_score"]= pd.Series([10.0, 50.0, 80.0, 66.0])
df_labels_test["livability_score"]   = pd.Series([10.0, 50.0, 80.0, 33.0])
df_labels_test["transport_score"]    = pd.Series([10.0, 50.0, 80.0, 66.0])

df_labeled = assign_labels(df_labels_test)

check("assign_labels: risk Low/Medium/High",
      list(df_labeled["risk_label"]) == ["Low", "Medium", "High", "Low"])
check("assign_labels: affordability Expensive/Moderate/Affordable",
      list(df_labeled["affordability_label"]) == ["Expensive", "Moderate", "Affordable", "Moderate"])
check("assign_labels: livability Poor/Fair/Good",
      list(df_labeled["livability_label"]) == ["Poor", "Fair", "Good", "Poor"])
check("assign_labels: transport Isolated/Moderate/Well-Connected",
      list(df_labeled["transport_label"]) == ["Isolated", "Moderate", "Well-Connected", "Moderate"])
check("assign_labels: risk_trend Decreasing/Stable/Increasing",
      list(df_labeled["risk_trend"]) == ["Decreasing", "Stable", "Increasing", "Stable"])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Anomaly detector tests
# ─────────────────────────────────────────────────────────────────────────────
section("4. Anomaly Detector Tests")

from ml.anomaly_detector import detect_anomalies

df_anom_test = df_base.copy()
df_anom_test["risk_score"]         = [20.0, 30.0, 60.0, 80.0]
df_anom_test["affordability_score"]= [80.0, 60.0, 40.0, 15.0]

df_anom = detect_anomalies(df_anom_test, "county")
check("detect_anomalies: anomaly_flag column present", "anomaly_flag" in df_anom.columns)
check("detect_anomalies: anomaly_severity column present", "anomaly_severity" in df_anom.columns)
check("detect_anomalies: flag values only 1 or -1",
      set(df_anom["anomaly_flag"].unique()).issubset({1, -1}))
check("detect_anomalies: normal rows have severity 'none'",
      (df_anom[df_anom["anomaly_flag"] == 1]["anomaly_severity"] == "none").all())
check("detect_anomalies: anomalous rows have valid severity",
      df_anom[df_anom["anomaly_flag"] == -1]["anomaly_severity"].isin(["high","medium","low"]).all())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TOPSIS recommender tests
# ─────────────────────────────────────────────────────────────────────────────
section("5. TOPSIS Recommender Tests")

from ml.recommender import topsis_rank

# Build a minimal ED-level scored df
df_topsis = df_base.copy()
df_topsis["risk_score"]         = [20.0, 40.0, 60.0, 80.0]
df_topsis["livability_score"]   = [80.0, 60.0, 40.0, 20.0]
df_topsis["transport_score"]    = [70.0, 55.0, 45.0, 30.0]
df_topsis["affordability_score"]= [85.0, 60.0, 40.0, 15.0]
df_topsis["ed_type"]            = ["town", "suburban", "urban_core", "urban_core"]

# Test ValueError on county data (no ed_type)
try:
    topsis_rank(df_base, {"max_rent_budget": 5000, "selected_area_types": ["town"],
                           "slider_affordability": 70, "slider_quality": 50,
                           "slider_transport": 40, "slider_energy": 30,
                           "slider_jobs": 50, "slider_stability": 40})
    check("topsis_rank: raises ValueError without ed_type", False, "No error raised")
except ValueError as e:
    check("topsis_rank: raises ValueError without ed_type", True, str(e)[:60])

profile = {
    "max_rent_budget": 2000.0,
    "selected_area_types": ["town", "suburban", "urban_core"],
    "slider_affordability": 70, "slider_quality": 50,
    "slider_transport": 40, "slider_energy": 30,
    "slider_jobs": 50, "slider_stability": 40,
}
ranked = topsis_rank(df_topsis, profile)
check("topsis_rank: match_score column present", "match_score" in ranked.columns)
check("topsis_rank: hard filter applied (rent ≤ budget)",
      (ranked["avg_monthly_rent"] <= profile["max_rent_budget"]).all())
check("topsis_rank: sorted descending by match_score",
      ranked["match_score"].is_monotonic_decreasing)
check("topsis_rank: match_score in [0,100]",
      (ranked["match_score"] >= 0).all() and (ranked["match_score"] <= 100).all())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ARIMA forecasting tests
# ─────────────────────────────────────────────────────────────────────────────
section("6. ARIMA Forecasting Tests")

from ml.forecasting import forecast_metric

# Normal 12-month series
rent_series = pd.Series([1200, 1220, 1240, 1250, 1260, 1280,
                          1290, 1310, 1320, 1340, 1360, 1380], dtype=float)
result = forecast_metric(rent_series, n_periods=6)
check("forecast_metric: returns forecast/lower_ci/upper_ci/method",
      all(k in result for k in ["forecast", "lower_ci", "upper_ci", "method"]))
check("forecast_metric: forecast length = n_periods", len(result["forecast"]) == 6)
check("forecast_metric: lower_ci length = n_periods", len(result["lower_ci"]) == 6)
check("forecast_metric: upper_ci length = n_periods", len(result["upper_ci"]) == 6)
check("forecast_metric: method is arima or linear", result["method"] in ["arima", "linear"])

# Short series → linear fallback
short_series = pd.Series([1000.0, 1050.0])
result_short = forecast_metric(short_series, n_periods=3)
check("forecast_metric: short series → linear fallback", result_short["method"] == "linear")
check("forecast_metric: linear fallback length correct", len(result_short["forecast"]) == 3)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Full pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
section("7. Full Pipeline Execution (ED granularity)")
print("  (Note: first run includes UMAP/numba JIT compilation — may take 30–90s)")

from ml.pipeline import run_pipeline

t0 = time.time()
try:
    scored_df, models = run_pipeline(granularity="ed")
    elapsed = time.time() - t0
    pipeline_ok = True
except Exception as e:
    traceback.print_exc()
    elapsed = time.time() - t0
    pipeline_ok = False
    scored_df, models = None, None

check("run_pipeline: completes without error", pipeline_ok)
if pipeline_ok:
    check(f"run_pipeline: completes in <5s (actual: {elapsed:.1f}s)", elapsed < 5.0)
    check("run_pipeline: returns 255 ED rows", len(scored_df) == 255)
    check("run_pipeline: 4 GBM models returned", len(models) == 4)

    required_cols = [
        "risk_score", "livability_score", "transport_score", "affordability_score",
        "risk_label", "affordability_label", "livability_label", "transport_label",
        "risk_trend", "cluster", "cluster_category", "umap_x", "umap_y",
        "anomaly_flag", "anomaly_severity",
        "affordability_index", "rent_to_income_pct", "commute_cost_monthly",
        "true_cost_index", "energy_tax", "commute_to_rent_ratio",
    ]
    missing = [c for c in required_cols if c not in scored_df.columns]
    check("run_pipeline: all required output columns present",
          len(missing) == 0, f"Missing: {missing}" if missing else "")

    for score in ["risk_score", "livability_score", "transport_score", "affordability_score"]:
        in_range = (scored_df[score] >= 0).all() and (scored_df[score] <= 100).all()
        check(f"run_pipeline: {score} in [0,100]", in_range,
              f"min={scored_df[score].min():.1f} max={scored_df[score].max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Model performance summary
# ─────────────────────────────────────────────────────────────────────────────
if pipeline_ok:
    section("8. Model Performance Summary")

    from scipy.stats import spearmanr
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [c for c in [
        "avg_monthly_rent", "avg_income", "employment_rate", "unemployment_rate",
        "traffic_volume", "congestion_delay_minutes", "ber_avg_score",
        "est_annual_energy_cost", "rental_yield",
        "affordability_index", "rent_to_income_pct", "commute_cost_monthly",
        "true_cost_index", "energy_tax", "commute_to_rent_ratio",
    ] if c in scored_df.columns]

    X = scored_df[feature_cols].fillna(0)

    # Recompute targets for validation metrics
    scaler = MinMaxScaler()
    scaled_arr = scaler.fit_transform(X)
    scaled = pd.DataFrame(scaled_arr, columns=feature_cols, index=scored_df.index)

    def col(name):
        return scaled[name] if name in scaled.columns else pd.Series(0.0, index=scored_df.index)

    targets_recomputed = {
        "risk_score": (
            col("rental_yield")*20 + col("congestion_delay_minutes")*15 +
            col("rent_to_income_pct")*15 + (1-col("employment_rate"))*15 +
            col("ber_avg_score")*10 + col("est_annual_energy_cost")*10 +
            col("true_cost_index")*15
        ).clip(0, 100),
        "livability_score": (
            col("employment_rate")*20 + col("avg_income")*15 +
            (1-col("ber_avg_score"))*15 + (1-col("avg_monthly_rent"))*15 +
            (1-col("congestion_delay_minutes"))*10 + col("affordability_index")*25
        ).clip(0, 100),
        "transport_score": (
            (1-col("congestion_delay_minutes"))*40 + col("traffic_volume")*30 +
            col("employment_rate")*15 + (1-col("ber_avg_score"))*15
        ).clip(0, 100),
        "affordability_score": (
            col("affordability_index")*35 + (1-col("rent_to_income_pct"))*30 +
            (1-col("avg_monthly_rent"))*20 + (1-col("est_annual_energy_cost"))*15
        ).clip(0, 100),
    }

    print(f"\n  {'Model':<22} {'R²':>6}  {'Spearman ρ':>10}  {'Score Range':>14}  {'Mean':>6}  {'Top Feature'}")
    print(f"  {'-'*22} {'-'*6}  {'-'*10}  {'-'*14}  {'-'*6}  {'-'*20}")

    for score_name, model in models.items():
        y_true = targets_recomputed[score_name]
        y_pred = scored_df[score_name]

        r2 = model.score(X, y_true)
        rho, _ = spearmanr(y_true, y_pred)

        score_min = y_pred.min()
        score_max = y_pred.max()
        score_mean = y_pred.mean()

        importances = model.feature_importances_
        top_idx = np.argmax(importances)
        top_feat = feature_cols[top_idx]

        r2_flag = " ⚠️ overfit" if r2 > 0.99 else (" ⚠️ diverge" if r2 < 0.90 else "")
        rho_flag = " ⚠️" if rho < 0.95 else ""

        print(f"  {score_name:<22} {r2:>6.4f}  {rho:>10.4f}  "
              f"{score_min:>5.1f}–{score_max:<5.1f}   {score_mean:>5.1f}  {top_feat}{r2_flag}{rho_flag}")

    # Score distribution
    print(f"\n  Score Distribution (ED level, N=255):")
    print(f"  {'Score':<22} {'Low/Poor/Isolated':>18} {'Mid':>6} {'High/Good/Connected':>20}")
    print(f"  {'-'*22} {'-'*18} {'-'*6} {'-'*20}")
    label_map = {
        "risk_score":         ("risk_label",         "Low",      "Medium",   "High"),
        "livability_score":   ("livability_label",    "Poor",     "Fair",     "Good"),
        "transport_score":    ("transport_label",     "Isolated", "Moderate", "Well-Connected"),
        "affordability_score":("affordability_label", "Expensive","Moderate", "Affordable"),
    }
    for score, (label_col, l1, l2, l3) in label_map.items():
        counts = scored_df[label_col].value_counts()
        c1 = counts.get(l1, 0)
        c2 = counts.get(l2, 0)
        c3 = counts.get(l3, 0)
        print(f"  {score:<22} {f'{l1}: {c1}':>18} {f'{l2}: {c2}':>6} {f'{l3}: {c3}':>20}")

    # Cluster distribution
    print(f"\n  Cluster Distribution:")
    cluster_counts = scored_df["cluster_category"].value_counts()
    for cat, cnt in cluster_counts.items():
        print(f"    {cat:<30} {cnt:>3} EDs")

    # Anomaly summary
    n_anomalies = (scored_df["anomaly_flag"] == -1).sum()
    sev_counts = scored_df[scored_df["anomaly_flag"] == -1]["anomaly_severity"].value_counts()
    print(f"\n  Anomalies detected: {n_anomalies} / 255 EDs")
    for sev, cnt in sev_counts.items():
        print(f"    {sev:<10} {cnt}")

    # Artifacts check
    print(f"\n  Artifacts saved to ml/models/:")
    models_dir = Path("ml/models")
    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<40} {size_kb:>7.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("Test Summary")
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
total  = len(results)
print(f"\n  {passed}/{total} tests passed", end="")
if failed:
    print(f"  |  {failed} FAILED:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"    ❌ {name}" + (f" — {detail}" if detail else ""))
else:
    print("  — all green ✅")

sys.exit(0 if failed == 0 else 1)
