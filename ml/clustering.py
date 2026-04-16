"""Clustering: Find similar areas using UMAP and KMeans (county or ED level)."""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        import umap
    except ImportError:
        umap = None

logger = logging.getLogger(__name__)

def generate_clusters(scores_df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    """
    Cluster areas and reduce dimensionality for a 2D scatter plot.
    
    Automatically selects cluster count based on dataset size:
    - County level (≤30 rows): 4 clusters
    - ED level (>30 rows): 6-8 clusters
    
    Args:
        scores_df: The harmonized DataFrame with all area metrics.
        n_clusters: Number of K-Means clusters (auto-detected if None).
        
    Returns:
        DataFrame with additional 'cluster', 'umap_x', and 'umap_y' columns.
    """
    if scores_df is None or len(scores_df) < 4 or umap is None:
        logger.warning(f"Insufficient data or umap not installed. Clustering skipped.")
        return scores_df.copy()
    
    # Auto-detect cluster count
    if n_clusters is None:
        if len(scores_df) <= 30:
            n_clusters = 4
        elif len(scores_df) <= 100:
            n_clusters = 6
        else:
            n_clusters = 8
    
    n_clusters = min(n_clusters, len(scores_df) - 1)
        
    features = [
        "avg_monthly_rent",
        "risk_score",
        "livability_score",
        "affordability_score",
        "transport_score",
        "employment_rate"
    ]
    
    missing = [f for f in features if f not in scores_df.columns]
    if missing:
        logger.warning(f"Missing clustering features: {missing}")
        return scores_df.copy()
        
    df = scores_df.copy()
    X = df[features].fillna(df[features].mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    # UMAP — adjust parameters for dataset size
    n_neighbors = min(5, len(scores_df) - 1) if len(scores_df) <= 30 else min(15, len(scores_df) - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.3,
        n_components=2,
        random_state=42
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        embedding = reducer.fit_transform(X_scaled)
        
    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]
    
    # Cluster naming heuristics
    df["cluster_name"] = "Cluster " + df["cluster"].astype(str)
    
    cluster_profiles = {}
    for c in range(n_clusters):
        cluster_data = df[df["cluster"] == c]
        avg_rent = cluster_data["avg_monthly_rent"].mean()
        avg_risk = cluster_data["risk_score"].mean()
        avg_livability = cluster_data["livability_score"].mean()
        avg_afford = cluster_data["affordability_score"].mean()
        
        if avg_rent > df["avg_monthly_rent"].mean() * 1.15:
            name = "Premium Urban 🏙️"
        elif avg_livability > df["livability_score"].mean() and avg_rent < df["avg_monthly_rent"].mean():
            name = "Hidden Gems 💎"
        elif avg_risk > 60:
            name = "High Risk / Volatile ⚠️"
        elif avg_afford > df["affordability_score"].mean() * 1.1 and avg_risk < 40:
            name = "Affordable & Safe 🌿"
        elif avg_rent < df["avg_monthly_rent"].mean() * 0.85:
            name = "Budget Rural 🏡"
        else:
            name = "Balanced Suburbs 🏡"
            
        cluster_profiles[c] = name
        
    df["cluster_category"] = df["cluster"].map(cluster_profiles)
    
    return df
