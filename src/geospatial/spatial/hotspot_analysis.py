"""
Hotspot Analysis — Getis-Ord Gi* 
src/geospatial/spatial/hotspot_analysis.py

This script identifies statistically significant spatial clusters of historical 
accidents along road segments using the Getis-Ord Gi* statistic.

Output: data/processed/results/historical_hotspots.gpkg
"""

import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from esda.getisord import G_Local

DATA_CFG_PATH = "configs/data_sources.yaml"

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("  H-Spot Bangkok — Historical Hotspot Analysis (Gi*)")
    print("=" * 60)

    # 1. Load Configurations
    data_cfg = load_yaml(DATA_CFG_PATH)
    segments_path = data_cfg["road_segments"]["output"]
    features_dir = data_cfg["features"]["output_dir"]
    features_path = os.path.join(features_dir, "model_dataset.parquet")

    # 2. Load Data
    print(f"Loading road segments from {segments_path}...")
    # Load only necessary columns to save memory
    segments = gpd.read_file(segments_path, columns=['segment_id', 'geometry'])
    
    print(f"Loading accident counts from {features_path}...")
    features = pd.read_parquet(features_path, columns=['segment_id', 'acc_total'])

    # Merge geometry with accident counts
    gdf = segments.merge(features, on='segment_id', how='inner')
    print(f"Merged data: {len(gdf):,} segments ready for analysis.")

    # 3. Create Spatial Weights Matrix
    # We use the centroid of each segment to calculate distances
    print("\nCalculating segment centroids for spatial weights...")
    # Ensure we are working with a projected CRS for accurate distance (meters)
    gdf = gdf.to_crs(data_cfg["road_segments"]["projected_crs"])
    centroids = gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid

    print("Building K-Nearest Neighbors (KNN) weights matrix (k=8)...")
    # KNN guarantees every segment has exactly 8 neighbors, preventing "island" errors
    w = libpysal.weights.KNN.from_dataframe(centroids, k=8)
    w.transform = 'R'  # Row-standardize the weights

    # 4. Calculate Getis-Ord Gi*
    print("\nCalculating Getis-Ord Gi* statistic...")
    y = gdf['acc_total'].values
    
    # Calculate local G
    gi_star = G_Local(y, w, transform='R', star=True)

    # Append results back to our GeoDataFrame
    gdf['gi_zscore'] = gi_star.Zs
    gdf['gi_pvalue'] = gi_star.p_sim

    # 5. Classify Hotspots
    # Z-score > 1.96 and p-value < 0.05 indicates a statistically significant hotspot (95% confidence)
    print("\nClassifying segments...")
    gdf['is_hotspot'] = ((gdf['gi_zscore'] > 1.96) & (gdf['gi_pvalue'] < 0.05)).astype(int)
    
    # Optional: Identify coldspots (statistically significant safe zones)
    gdf['is_coldspot'] = ((gdf['gi_zscore'] < -1.96) & (gdf['gi_pvalue'] < 0.05)).astype(int)

    hotspot_count = gdf['is_hotspot'].sum()
    print(f"  → Found {hotspot_count:,} significant hotspot segments.")
    print(f"  → Found {gdf['is_coldspot'].sum():,} significant coldspot segments.")

    # 6. Save Output
    os.makedirs("data/processed/results", exist_ok=True)
    out_path = "data/processed/results/historical_hotspots.gpkg"
    print(f"\nSaving hotspot map layer to {out_path}...")
    
    # Drop rows without geometry if any exist and save
    gdf = gdf.dropna(subset=['geometry'])
    gdf.to_file(out_path, driver="GPKG")
    
    print("Done! 🎉")

if __name__ == "__main__":
    main()
