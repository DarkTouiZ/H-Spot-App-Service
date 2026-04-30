"""
Production-Level Training Script
src/models/train.py

A configuration-driven training engine that:
1. Loads features and parameters from YAML.
2. Trains a model (XGBoost or Random Forest).
3. Logs metrics and artifacts to MLFlow.
4. Saves the final model.
"""

import os
import yaml
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Try to import MLFlow (Optional)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Config Paths
DATA_CFG_PATH = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    # 1. Load Configurations
    data_cfg = load_yaml(DATA_CFG_PATH)
    model_cfg = load_yaml(MODEL_CFG_PATH)
    
    m_params = model_cfg["modeling"]
    h_params = m_params["hyperparameters"]
    
    output_dir = data_cfg["features"]["output_dir"]
    data_path = os.path.join(output_dir, "model_dataset.parquet")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run pipeline first.")

    # 2. Load and Filter Data
    print(f"Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    
    features = m_params["features"]
    target = m_params["target"]
    
    X = df[features]
    y = df[target]
    
    # 3. Train/Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=h_params.get("random_state", 42), stratify=y
    )
    
    # 4. Initialize Model based on Config
    model_type = m_params.get("model_type", "xgboost")
    
    # Calculate class weight ratio for imbalance
    ratio = (y == 0).sum() / (y == 1).sum()
    
    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=h_params.get("n_estimators", 100),
            learning_rate=h_params.get("learning_rate", 0.1),
            max_depth=h_params.get("max_depth", 6),
            scale_pos_weight=ratio,
            random_state=h_params.get("random_state", 42),
            use_label_encoder=False,
            eval_metric="logloss"
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=h_params.get("n_estimators", 100),
            max_depth=h_params.get("max_depth", 10),
            class_weight="balanced",
            random_state=h_params.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 5. Training and MLFlow Logging
    if HAS_MLFLOW:
        mlflow.set_experiment(m_params.get("experiment_name", "Default_Exp"))
        with mlflow.start_run():
            print(f"Starting MLFlow Run: {model_type}")
            
            # Log Parameters
            mlflow.log_params(h_params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("num_features", len(features))
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Metrics
            auc = roc_auc_score(y_test, y_prob)
            mlflow.log_metric("auc_roc", auc)
            
            print(f"Model AUC: {auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Log Model Artifact
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Save local copy as well
            model_save_path = os.path.join("models", f"{model_type}_latest.pkl")
            os.makedirs("models", exist_ok=True)
            with open(model_save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Model also saved locally to {model_save_path}")
            
    else:
        # Standard training without MLFlow
        print(f"Training {model_type} (MLFlow not installed)...")
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"Model AUC: {auc:.4f}")
        
        model_save_path = os.path.join("models", f"{model_type}_latest.pkl")
        os.makedirs("models", exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
