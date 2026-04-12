"""
Clustering module for the Léarslán ML scoring pipeline.

Clusters areas using KMeans + UMAP and assigns human-readable archetype labels.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_CLUSTER_FEATURES = [
    "avg_monthly_rent",
    "risk_score",
    "livability_score",
    "affordability_score",
    "transport_score",
    "employment_rate",
]

_VALID_LABELS = {
    "Premium Urban",
    "Hidden Gems",
    "High Risk / Volatile",
    "Affordable & Safe",
    "Budget Rural",
    "Balanced Suburbs",
}


def _assign_cluster_category(
    cluster_id: int,
    cluster_stats: pd.DataFrame,
    dataset_avg_rent: float,
    dataset_avg_livability: float,
    dataset_avg_affordability: float,
) -> str:
    """
    Assign a human-readable archetype label to a cluster based on its centroid stats.
    Rules are evaluated in priority order against dataset-level averages.
    """
    row = cluster_stats.loc[cluster_id]
    avg_rent = row["avg_monthly_rent"]
    avg_livability = row["livability_score"]
    avg_risk = row["risk_score"]
    avg_affordability = row["affordability_score"]

    if avg_rent > dataset_avg_rent * 1.15:
        return "Premium Urban"
    if avg_livability > dataset_avg_livability and avg_rent < dataset_avg_rent:
        return "Hidden Gems"
    if avg_risk > 60:
        return "High Risk / Volatile"
    if avg_affordability > dataset_avg_affordability * 1.10 and avg_risk < 40:
        return "Affordable & Safe"
    if avg_rent < dataset_avg_rent * 0.85:
        return "Budget Rural"
    return "Balanced Suburbs"


def cluster_areas(
    df: pd.DataFrame,
    granularity: str,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Cluster areas using KMeans + UMAP and assign archetype labels.

    Args:
        df: Scored DataFrame containing the 6 clustering features.
        granularity: "county" → 4 clusters; "ed" → 7 clusters.

    Returns:
        (df_with_clusters, kmeans_model, scaler) — 3-tuple.
        New columns added: cluster, cluster_category, umap_x, umap_y.
    """
    out = df.copy()
    n_clusters = 4 if granularity == "county" else 7

    X = out[_CLUSTER_FEATURES].values
    N = len(X)

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    out["cluster"] = kmeans.fit_predict(X_scaled)

    # UMAP dimensionality reduction
    # Note: UMAP requires numba JIT compilation on first run (~2-5 min).
    # Set env var LEARSLAN_SKIP_UMAP=1 to use PCA instead for faster iteration.
    import os
    n_neighbors = min(15, N - 1)
    skip_umap = os.environ.get("LEARSLAN_SKIP_UMAP", "0") == "1"
    if not skip_umap:
        try:
            from umap import UMAP
            reducer = UMAP(
                n_components=2,
                min_dist=0.3,
                random_state=42,
                n_neighbors=n_neighbors,
                n_epochs=200,
                low_memory=False,
                verbose=False,
            )
            embedding = reducer.fit_transform(X_scaled)
        except Exception:
            logger.warning("UMAP failed, falling back to PCA for 2D projection.")
            skip_umap = True

    if skip_umap:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(X_scaled)
    out["umap_x"] = embedding[:, 0].astype(float)
    out["umap_y"] = embedding[:, 1].astype(float)

    # Compute per-cluster means for labelling heuristic
    cluster_stats = out.groupby("cluster")[_CLUSTER_FEATURES].mean()

    dataset_avg_rent = out["avg_monthly_rent"].mean()
    dataset_avg_livability = out["livability_score"].mean()
    dataset_avg_affordability = out["affordability_score"].mean()

    label_map = {
        cid: _assign_cluster_category(
            cid,
            cluster_stats,
            dataset_avg_rent,
            dataset_avg_livability,
            dataset_avg_affordability,
        )
        for cid in cluster_stats.index
    }

    out["cluster_category"] = out["cluster"].map(label_map).astype(str)

    return out, kmeans, scaler
