import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import pickle
import yaml
import os

# Set up local imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.xai_explainer import explain_segment
from src.modeling.narrator import generate_explanation

st.set_page_config(page_title="H-Spot Bangkok", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Configuration & Caching
# -----------------------------------------------------------------------------
DATA_CFG_PATH  = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"

@st.cache_resource
def load_config():
    with open(DATA_CFG_PATH) as f:
        data_cfg = yaml.safe_load(f)
    with open(MODEL_CFG_PATH) as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg

@st.cache_data
def load_risk_data(threshold=0.15):
    """Loads and merges geometries with risk scores. Filters by threshold to keep map fast."""
    
    scores = pd.read_parquet("data/processed/results/risk_scores_v2_xgboost.parquet")
    segments = gpd.read_file("data/processed/road_segments.gpkg", columns=['segment_id', 'geometry'])
    
    # Merge and transform to WGS84 for PyDeck
    gdf = segments.merge(scores, on="segment_id", how="inner")
    gdf = gdf.to_crs("EPSG:4326")
    
    # Aggressive filtering to prevent MessageSizeError
    gdf_filtered = gdf[gdf['risk_score'] >= threshold].copy()
    
    # Convert Linestring to coordinate lists for PyDeck PathLayer
    gdf_filtered['path'] = gdf_filtered['geometry'].apply(lambda geom: [[c[0], c[1]] for c in geom.coords])
    
    # Define colors based on risk
    def get_color(risk):
        if risk > 0.5: return [255, 0, 0, 255]      # Red
        if risk > 0.3: return [255, 165, 0, 200]    # Orange
        return [255, 255, 0, 150]                   # Yellow
        
    gdf_filtered['color'] = gdf_filtered['risk_score'].apply(get_color)
    # Drop geometry object to save memory/payload size
    return gdf_filtered.drop(columns=['geometry'])

@st.cache_data
def load_hotspots():
    """Loads historical hotspots."""
    if not os.path.exists("data/processed/results/historical_hotspots.gpkg"):
        return None
    hotspots = gpd.read_file("data/processed/results/historical_hotspots.gpkg")
    hotspots = hotspots.to_crs("EPSG:4326")
    hotspots['path'] = hotspots['geometry'].apply(lambda geom: [[c[0], c[1]] for c in geom.coords])
    hotspots['color'] = hotspots.apply(lambda _: [139, 0, 0, 255], axis=1) # Dark Red
    return hotspots

@st.cache_data
def load_xai_data():
    data_cfg, model_cfg = load_config()
    features_dir = data_cfg["features"]["output_dir"]
    df = pd.read_parquet(os.path.join(features_dir, "model_dataset.parquet"))
    
    model_path = "models/xgboost_v2_xgboost.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    features = [f for f in model_cfg["modeling"]["features_v2"] if f in df.columns]
    return df, model, features, model_cfg

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🚦 H-Spot Bangkok")
    st.sidebar.markdown("Urban Traffic Risk Assessment")
    
    mode = st.sidebar.radio("Navigation", [
        "1. Predictive Risk Map",
        "2. Historical Hotspots",
        "3. Explainable AI (XAI)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Map Filters")
    risk_threshold = st.sidebar.slider(
        "Min Risk Threshold (%)", 
        min_value=5, max_value=90, value=20, step=5
    ) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System")
    if st.sidebar.button("Clear App Cache"):
        st.cache_data.clear()
        st.rerun()
    
    data_cfg, model_cfg = load_config()

    if mode == "1. Predictive Risk Map":
        st.header("🔮 Predictive Risk Map")
        st.markdown(f"Showing segments with predicted risk ≥ {risk_threshold*100:.0f}%.")
        
        with st.spinner("Filtering and loading Map Data..."):
            gdf = load_risk_data(risk_threshold)
            
            if gdf.empty:
                st.warning(f"No segments found with risk ≥ {risk_threshold*100:.0f}%. Try lowering the threshold.")
                return
            
            st.caption(f"Currently displaying {len(gdf):,} segments.")
            
            # Setup PyDeck Layer
            layer = pdk.Layer(
                "PathLayer",
                gdf,
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=2,
                get_path="path",
                get_width=5,
            )
            
            # View state centered on Bangkok
            view_state = pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=11, pitch=45)
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Segment ID: {segment_id}\nRisk Score: {risk_pct}%"}
            )
            
            st.pydeck_chart(r)

    elif mode == "2. Historical Hotspots":
        st.header("🔥 Historical Hotspots (Getis-Ord Gi*)")
        st.markdown("Statistically significant spatial clusters of historical accidents.")
        
        with st.spinner("Loading Hotspot Data..."):
            hotspots = load_hotspots()
            if hotspots is None:
                st.warning("No hotspot data found. Please run the hotspot analysis script first.")
                return
                
            layer = pdk.Layer(
                "PathLayer",
                hotspots,
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=3,
                get_path="path",
                get_width=5,
            )
            view_state = pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=11, pitch=45)
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Segment ID: {segment_id}\nTotal Historical Accidents: {acc_total}\nZ-Score: {gi_zscore}"}
            )
            st.pydeck_chart(r)

    elif mode == "3. Explainable AI (XAI)":
        st.header("🧠 Explainable AI (XAI)")
        st.markdown("Enter a Road Segment ID to understand *why* it is considered risky.")
        
        df, model, features, model_cfg = load_xai_data()
        
        segment_id = st.number_input("Enter Segment ID", min_value=0, value=103470, step=1)
        
        if st.button("Analyze Risk"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    risk_score, top_factors = explain_segment(segment_id, df, model, features, top_k=5)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Predicted Risk Score", f"{risk_score*100:.1f}%")
                        st.markdown("### Top Factors Driving Risk")
                        
                        # Prepare data for chart
                        plot_df = pd.DataFrame(list(top_factors.items()), columns=['Feature', 'Impact'])
                        plot_df['Direction'] = plot_df['Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
                        plot_df['Absolute Impact'] = plot_df['Impact'].abs()
                        plot_df = plot_df.sort_values('Absolute Impact', ascending=True)
                        
                        st.bar_chart(plot_df.set_index('Feature')['Impact'])
                        
                    with col2:
                        st.markdown("### AI Narrative Explanation")
                        should_narrate = model_cfg.get("explanation", {}).get("enable_narrative", False)
                        
                        if should_narrate:
                            if os.environ.get("GEMINI_API_KEY"):
                                with st.spinner("Asking Gemini..."):
                                    llm_model = model_cfg.get("explanation", {}).get("llm_model", "gemini-1.5-flash")
                                    narrative = generate_explanation(segment_id, risk_score, top_factors, llm_model)
                                    if narrative:
                                        st.info(narrative)
                                    else:
                                        st.error("Failed to generate narrative. Check terminal for errors.")
                            else:
                                st.warning("GEMINI_API_KEY not found in environment. Narrative generation skipped. Check your .env file.")
                        else:
                            st.info("Narrative generation is disabled in configs/model_params.yaml.")
                except ValueError as e:
                    st.error(str(e))

if __name__ == "__main__":
    main()
