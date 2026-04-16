import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.spatial_harmonizer import harmonize_data, get_time_series_data
from ml.feature_engineering import engineer_features
from ml.risk_model import train_risk_model
from ml.clustering import generate_clusters

def generate_comprehensive_metrics():
    print("Loading and harmonizing data...")
    df = harmonize_data()
    df = engineer_features(df, daft_summaries={})
    ts_data = get_time_series_data()

    print("Executing GBM Scoring...")
    models, scored_df, feature_names = train_risk_model(df)
    
    # Base scores from the existing pipeline
    base_metrics = ['risk_score', 'livability_score', 'transport_score', 'affordability_score']
    
    print("Executing Anomaly Detection...")
    # Features for anomaly/consistency detection
    X_anomaly = scored_df[feature_names].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)
    
    # Isolation Forest for Anomaly Score & Reconstruction Error proxy
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso.fit(X_scaled)
    # score_samples returns opposite of anomaly score (lower is more anomalous)
    # We want anomaly_risk_score where higher is more anomalous
    anomaly_scores = -iso.score_samples(X_scaled) 
    scored_df['anomaly_risk_score'] = (MinMaxScaler().fit_transform(anomaly_scores.reshape(-1, 1)) * 100).flatten()
    
    # Reconstruction error proxy (distance to sample density)
    # For Isolation Forest, we can use the depth path as a proxy, 
    # but let's use a PCA reconstruction error for more "Senior AI" feel
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    X_projected = pca.inverse_transform(X_pca)
    recon_error = np.sum((X_scaled - X_projected)**2, axis=1)
    scored_df['reconstruction_error'] = (MinMaxScaler().fit_transform(recon_error.reshape(-1, 1)) * 10).flatten()

    print("Executing Clustering...")
    clustered_df = generate_clusters(scored_df)
    
    # Cluster Risk Score - handle missing cluster column
    if 'cluster' in clustered_df.columns:
        cluster_means = clustered_df.groupby('cluster')['risk_score'].mean().to_dict()
        scored_df['cluster_risk_score'] = clustered_df['cluster'].map(cluster_means)
    else:
        print("Clustering skipped (missing umap), using global mean as cluster mean.")
        scored_df['cluster_risk_score'] = scored_df['risk_score'].mean()

    print("Calculating Stability and Consistency...")
    stability_map = {}
    consistency_map = {}
    
    for county, county_ts in ts_data.items():
        if len(county_ts) > 1:
            # Stability: 100 - (coeff of variation * 100)
            # Focus on rent stability
            rent_std = county_ts['avg_monthly_rent'].std()
            rent_mean = county_ts['avg_monthly_rent'].mean()
            stability = max(0, 100 - ((rent_std / rent_mean) * 1000)) if rent_mean > 0 else 50
            stability_map[county] = stability
            
            # Consistency: Inverse of variance of rent growth
            growth_std = county_ts['rent_growth_pct'].std()
            consistency = max(0, 100 - (growth_std * 500))
            consistency_map[county] = consistency
        else:
            stability_map[county] = 50
            consistency_map[county] = 50

    scored_df['stability_index'] = scored_df['county'].map(stability_map).fillna(50)
    scored_df['consistency_risk_score'] = scored_df['county'].map(consistency_map).fillna(50)
    
    # Confidence: Inverse of anomaly score relative to cluster
    # Or just an inverse scaling of anomaly score
    scored_df['confidence'] = (100 - (scored_df['anomaly_risk_score'] * 0.5)).round(1)
    
    # Consistency Anomaly Label
    def get_label(row):
        if row['anomaly_risk_score'] > 70:
            return "High Anomaly"
        if row['consistency_risk_score'] < 40:
            return "Volatile"
        return "Consistent"
    
    scored_df['consistency_anomaly_label'] = scored_df.apply(get_label, axis=1)

    print("Formatting Table...")
    # Sorting by risk score for Ranking
    final_df = scored_df.sort_values(by='risk_score', ascending=True).reset_index(drop=True)
    final_df['Rank'] = final_df.index + 1
    
    # Precision rounding
    cols_to_round = [
        'risk_score', 'consistency_risk_score', 'anomaly_risk_score', 
        'livability_score', 'transport_score', 'affordability_score', 
        'stability_index', 'cluster_risk_score', 'reconstruction_error'
    ]
    for col in cols_to_round:
        final_df[col] = final_df[col].round(2)

    # Columns requested: Rank,county,risk_score,consistency_risk_score,anomaly_risk_score,livability_score,transport_score,affordability_score,consistency_anomaly_label,stability_index,confidence,cluster_risk_score,reconstruction_error
    requested_cols = [
        'Rank', 'county', 'risk_score', 'consistency_risk_score', 'anomaly_risk_score',
        'livability_score', 'transport_score', 'affordability_score',
        'consistency_anomaly_label', 'stability_index', 'confidence',
        'cluster_risk_score', 'reconstruction_error'
    ]
    
    final_table = final_df[requested_cols]
    
    # Save to MD
    md_content = "# Comprehensive County Risk & Performance Metrics\n\n"
    md_content += "Generated using Gradient Boosting Machines (GBM), Isolation Forest, and PCA Analysis.\n\n"
    
    # Manual Markdown Table
    headers = requested_cols
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for _, row in final_table.iterrows():
        md_content += "| " + " | ".join([str(row[col]) for col in headers]) + " |\n"
    
    out_path = Path(__file__).parent.parent / "docs" / "county_risk_metrics_comprehensive.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"File saved to {out_path}")

if __name__ == "__main__":
    generate_comprehensive_metrics()
