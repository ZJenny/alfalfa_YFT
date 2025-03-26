import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import xgboost as xgb
import joblib
from scipy.stats import randint, uniform
from datetime import datetime
import json

from var import folder_results_csv, folder_saved_models, folder_png, folder_results

# Create directories if they don't exist
for folder in [folder_saved_models, folder_results, folder_results_csv, folder_png]:
    os.makedirs(folder, exist_ok=True)


def train_test_ensemble(data_path, train_experiments, test_experiments, author_name="your_name"):
    """
    Train and evaluate an ensemble model that combines FM-like, XGBoost, and Random Forest
    for alfalfa yield prediction

    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the data
    train_experiments : list
        List of experiment names to use for training
    test_experiments : list
        List of experiment names to use for testing
    author_name : str
        Author name to include in saved files

    Returns:
    --------
    model : object
        Trained ensemble model
    results : dict
        Dictionary containing performance metrics and predictions
    """
    # Generate timestamp for file versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    data = pd.read_csv(data_path)

    # 1. Data Preparation
    # ------------------

    # Convert timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["month"] = data["Timestamp"].dt.month
    data["year_numeric"] = data["Timestamp"].dt.year

    # Filter to only rows with yield data
    data_with_yield = data[data["Dry yield (ton/ha)"].notna()].copy()
    print(f"Working with {len(data_with_yield)} rows that have yield data")

    # Remove negative yield values (likely errors)
    data_with_yield = data_with_yield[data_with_yield["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)

    # 2. Feature Engineering
    # ---------------------

    # Create seasonal features using sine and cosine transformations
    data_with_yield["sin_day"] = np.sin(2 * np.pi * data_with_yield["Day of Year"] / 365)
    data_with_yield["cos_day"] = np.cos(2 * np.pi * data_with_yield["Day of Year"] / 365)

    # Location and irrigation type indicators
    data_with_yield["is_bushland"] = data_with_yield["location"].str.contains("Bushland", case=False, na=False).astype(int)
    data_with_yield["is_reno"] = data_with_yield["location"].str.contains("Reno", case=False, na=False).astype(int)
    data_with_yield["is_fallon"] = data_with_yield["location"].str.contains("Fallon", case=False, na=False).astype(int)
    data_with_yield["is_drip"] = data_with_yield["experiment_info"].str.contains("Drip", case=False, na=False).astype(int)
    data_with_yield["is_linear"] = data_with_yield["experiment_info"].str.contains("Linear", case=False, na=False).astype(int)
    data_with_yield["is_pivot"] = data_with_yield["experiment_info"].str.contains("Pivot", case=False, na=False).astype(int)
    data_with_yield["is_lysimeter"] = data_with_yield["experiment_info"].str.contains("Lysimeter", case=False, na=False).astype(int)

    # Extract year from experiment info
    def extract_year(exp):
        parts = exp.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None

    data_with_yield["exp_year"] = data_with_yield["experiment_info"].apply(extract_year)

    # 3. Train-Test Split
    # ------------------
    print("Creating train/test split using specified experiments...")

    train_data = data_with_yield[data_with_yield["experiment_info"].isin(train_experiments)].copy()
    test_data = data_with_yield[data_with_yield["experiment_info"].isin(test_experiments)].copy()

    print(f"Train set: {len(train_data)} samples from {len(train_data['experiment_info'].unique())} experiments")
    print(f"Test set: {len(test_data)} samples from {len(test_data['experiment_info'].unique())} experiments")

    # Print experiment breakdown
    print("\nTrain experiments:")
    for exp in train_data["experiment_info"].unique():
        count = len(train_data[train_data["experiment_info"] == exp])
        print(f"  - {exp}: {count} samples")

    print("\nTest experiments:")
    for exp in test_data["experiment_info"].unique():
        count = len(test_data[test_data["experiment_info"] == exp])
        print(f"  - {exp}: {count} samples")

    # 4. Feature Processing
    # -------------------

    # Identify categorical and numerical columns
    categorical_cols = ["Alfalfa variety"]

    # Basic numeric columns we'll definitely use
    numeric_cols = ["Day of Year", "sin_day", "cos_day", "is_bushland", "is_reno", "is_fallon", "is_drip", "is_linear", "is_pivot", "is_lysimeter", "exp_year", "month"]

    # Add additional numerical features if they exist and have less than 70% missing values
    candidate_cols = [
        "Fall Dormancy*",
        "Winterhardiness**",
        "Average Air Temperature (Deg C)",
        "Total Precipitation (mm)",
        "Total Solar Radiation (kW-hr/m2)",
        "Average Wind Speed (m/s)",
        "Relative Humidity (%)",
        "Irrig. amount (in)",
        "Ave. SWD (%)",
    ]

    for col in candidate_cols:
        if col in train_data.columns and train_data[col].isna().mean() < 0.7:
            numeric_cols.append(col)

    # Check which columns exist in both train and test
    existing_categorical_cols = []
    for col in categorical_cols:
        if col in train_data.columns and col in test_data.columns:
            existing_categorical_cols.append(col)

    existing_numeric_cols = []
    for col in numeric_cols:
        if col in train_data.columns and col in test_data.columns:
            existing_numeric_cols.append(col)

    # Create preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    # For irrigation features, use 0 imputation; for others, use mean
    irrig_cols = [col for col in existing_numeric_cols if "Irrig" in col]
    other_numeric_cols = [col for col in existing_numeric_cols if col not in irrig_cols]

    # Create column transformer with different preprocessing for different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, existing_categorical_cols),
            ("num", SimpleImputer(strategy="mean"), other_numeric_cols),
            ("irrig", SimpleImputer(strategy="constant", fill_value=0), irrig_cols),
        ]
    )

    # 5. Process train and test data
    # ----------------------------

    # Extract X (features) and y (target)
    X_train = train_data[existing_categorical_cols + existing_numeric_cols]
    y_train = train_data["Dry yield (ton/ha)"]

    X_test = test_data[existing_categorical_cols + existing_numeric_cols]
    y_test = test_data["Dry yield (ton/ha)"]

    # Apply preprocessing
    print("Preprocessing features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # 6. Add interaction terms to mimic FM for Elastic Net
    # ------------------------------------------------
    from sklearn.preprocessing import PolynomialFeatures

    print("Generating polynomial features to mimic FM interactions...")
    # If there are too many features, limit interactions to avoid memory issues
    if X_train_scaled.shape[1] > 100:
        print(f"Warning: Large number of features ({X_train_scaled.shape[1]}). Limiting to top 50 to avoid memory issues.")
        # Use Random Forest for quick feature importance ranking
        feature_selector = RandomForestRegressor(n_estimators=50, random_state=42)
        feature_selector.fit(X_train_scaled, y_train)

        # Get indices of top 50 features
        top_indices = np.argsort(feature_selector.feature_importances_)[-50:]

        # Use only top features for polynomial expansion
        X_train_for_poly = X_train_scaled[:, top_indices]
        X_test_for_poly = X_test_scaled[:, top_indices]

        # Create polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features_train = poly.fit_transform(X_train_for_poly)
        poly_features_test = poly.transform(X_test_for_poly)

        # Combine original features with polynomial features
        X_train_poly = np.hstack([X_train_scaled, poly_features_train])
        X_test_poly = np.hstack([X_test_scaled, poly_features_test])
    else:
        # If feature count is reasonable, create all pairwise interactions
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

    print(f"Feature space expanded from {X_train_scaled.shape[1]} to {X_train_poly.shape[1]} features with interactions")

    # 7. Train Base Models
    # ------------------
    print("Training base models...")

    # 7.1 Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # 7.2 XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    # 7.3 ElasticNet (FM-like with feature interactions)
    print("Training ElasticNet (FM-like model)...")
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    en.fit(X_train_poly, y_train)

    # 8. Create Stacking Ensemble
    # -------------------------
    print("Creating stacking ensemble...")

    # Define base models for stacking
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("xgb", xgb.XGBRegressor(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ]

    # Try different meta-models (final estimators)
    meta_models = {"ridge": Ridge(alpha=1.0, random_state=42), "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), "gbm": GradientBoostingRegressor(n_estimators=100, random_state=42)}

    best_score = -np.inf
    best_meta_model = None

    # Find the best meta-model through cross-validation
    for name, meta_model in meta_models.items():
        stack = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

        # Evaluate using cross-validation
        cv_scores = cross_val_score(stack, X_train_scaled, y_train, cv=5, scoring="r2")
        mean_score = np.mean(cv_scores)

        print(f"Meta-model {name}: Mean R² = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_meta_model = name

    print(f"Best meta-model: {best_meta_model} with R² = {best_score:.4f}")

    # Create final stacking ensemble with the best meta-model
    final_stack = StackingRegressor(estimators=base_models, final_estimator=meta_models[best_meta_model], cv=5, n_jobs=-1)

    # Train the final ensemble
    print("Training final ensemble model...")
    final_stack.fit(X_train_scaled, y_train)

    # 9. Train the FM-enhanced ensemble
    # -------------------------------
    print("Creating enhanced ensemble with FM-like features...")

    # Create predictions from the base models
    rf_train_preds = rf.predict(X_train_scaled).reshape(-1, 1)
    xgb_train_preds = xgb_model.predict(X_train_scaled).reshape(-1, 1)
    en_train_preds = en.predict(X_train_poly).reshape(-1, 1)

    rf_test_preds = rf.predict(X_test_scaled).reshape(-1, 1)
    xgb_test_preds = xgb_model.predict(X_test_scaled).reshape(-1, 1)
    en_test_preds = en.predict(X_test_poly).reshape(-1, 1)

    # Combine base model predictions with original features for a richer meta-model
    X_train_meta = np.hstack([X_train_scaled, rf_train_preds, xgb_train_preds, en_train_preds])
    X_test_meta = np.hstack([X_test_scaled, rf_test_preds, xgb_test_preds, en_test_preds])

    # Train the final meta-model on this enhanced feature set
    meta_final = meta_models[best_meta_model]
    meta_final.fit(X_train_meta, y_train)

    # 10. Model Evaluation
    # -----------------
    print("Evaluating models...")

    # Make predictions with individual models
    rf_preds = rf.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_scaled)
    en_preds = en.predict(X_test_poly)

    # Make predictions with ensemble models
    stack_preds = final_stack.predict(X_test_scaled)
    enhanced_preds = meta_final.predict(X_test_meta)

    # Calculate metrics for all models
    models = {"Random Forest": rf_preds, "XGBoost": xgb_preds, "ElasticNet (FM-like)": en_preds, "Stacking Ensemble": stack_preds, "Enhanced Ensemble": enhanced_preds}

    # Create a metrics dataframe to save
    metrics_data = []

    print("\nModel Performance Comparison:")
    for name, preds in models.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        metrics_data.append({"Model": name, "RMSE": rmse, "R2": r2, "MAE": mae})

        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")

    # Save metrics comparison to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_ensemble_model_comparison_{timestamp}.csv"), index=False)

    # 11. Visualizations and Analysis for the Best Model
    # ----------------------------------------------

    # Use the enhanced ensemble as our final model
    final_preds = enhanced_preds

    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, final_preds, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.xlabel("Actual Yield (ton/ha)")
    plt.ylabel("Predicted Yield (ton/ha)")
    plt.title("Actual vs Predicted Alfalfa Yield (Ensemble Model)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder_png, f"{author_name}_ensemble_actual_vs_predicted_{timestamp}.png"))
    plt.close()

    # Calculate residuals
    residuals = y_test - final_preds

    # Create a dataframe for analysis
    error_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": final_preds,
            "Residual": residuals,
            "Experiment": test_data["experiment_info"].values,
            "Year": test_data["year_numeric"].values,
            "Month": test_data["month"].values,
        }
    )

    # Plot residuals by experiment
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Experiment", y="Residual", data=error_df)
    plt.xticks(rotation=90)
    plt.title("Prediction Residuals by Experiment (Ensemble Model)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_ensemble_residuals_by_experiment_{timestamp}.png"))
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(12, 6))
    model_names = list(models.keys())
    r2_scores = [r2_score(y_test, models[name]) for name in model_names]

    sns.barplot(x=model_names, y=r2_scores)
    plt.title("Model Comparison (R² Score)")
    plt.ylabel("R² Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_ensemble_model_comparison_{timestamp}.png"))
    plt.close()

    # Experiment-specific metrics
    print("\nExperiment-specific performance (Enhanced Ensemble):")
    exp_metrics = {}

    for exp in error_df["Experiment"].unique():
        exp_data = error_df[error_df["Experiment"] == exp]
        exp_rmse = np.sqrt(mean_squared_error(exp_data["Actual"], exp_data["Predicted"]))
        exp_r2 = r2_score(exp_data["Actual"], exp_data["Predicted"])
        exp_mae = mean_absolute_error(exp_data["Actual"], exp_data["Predicted"])

        exp_metrics[exp] = {"RMSE": exp_rmse, "R2": exp_r2, "MAE": exp_mae, "Samples": len(exp_data)}

        print(f"  - {exp}:")
        print(f"    * RMSE: {exp_rmse:.4f}")
        print(f"    * R²: {exp_r2:.4f}")
        print(f"    * MAE: {exp_mae:.4f}")
        print(f"    * Samples: {len(exp_data)}")

    # Save experiment-specific metrics
    exp_metrics_df = pd.DataFrame.from_dict(exp_metrics, orient="index")
    exp_metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_ensemble_experiment_metrics_{timestamp}.csv"))

    # Save detailed results
    results_df = pd.DataFrame(
        {"Experiment": test_data["experiment_info"].values, "Actual_Yield": y_test.values, "Predicted_Yield": final_preds, "Error": residuals, "Percent_Error": (residuals / y_test.values) * 100}
    )

    results_df.to_csv(os.path.join(folder_results_csv, f"{author_name}_ensemble_prediction_results_{timestamp}.csv"), index=False)

    # Save all models and components
    model_package = {
        "rf": rf,
        "xgb": xgb_model,
        "en": en,
        "stack": final_stack,
        "enhanced": meta_final,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "poly": poly,
        "best_meta_model": best_meta_model,
        "features": existing_categorical_cols + existing_numeric_cols,
        "timestamp": timestamp,
    }

    # Save the ensemble package
    ensemble_path = os.path.join(folder_saved_models, f"{author_name}_ensemble_alfalfa_yield_models_{timestamp}.joblib")
    joblib.dump(model_package, ensemble_path)

    # Save ensemble configuration as JSON (exclude the actual models which are saved in joblib)
    ensemble_config = {
        "base_models": ["RandomForest", "XGBoost", "ElasticNet (FM-like)"],
        "meta_model": best_meta_model,
        "features_count": len(existing_categorical_cols + existing_numeric_cols),
        "performance": {"enhanced_ensemble": {"rmse": np.sqrt(mean_squared_error(y_test, enhanced_preds)), "r2": r2_score(y_test, enhanced_preds), "mae": mean_absolute_error(y_test, enhanced_preds)}},
        "timestamp": timestamp,
    }

    # Save model configuration as JSON
    with open(os.path.join(folder_saved_models, f"{author_name}_ensemble_config_{timestamp}.json"), "w") as f:
        json.dump(ensemble_config, f, indent=4)

    print("\nEnsemble analysis complete!")
    print("Check the following files:")
    print(f"- {author_name}_ensemble_actual_vs_predicted_{timestamp}.png - Plot of predictions vs actual values")
    print(f"- {author_name}_ensemble_residuals_by_experiment_{timestamp}.png - Box plot of residuals by experiment")
    print(f"- {author_name}_ensemble_model_comparison_{timestamp}.png - Bar chart comparing different models")
    print(f"- {author_name}_ensemble_prediction_results_{timestamp}.csv - Detailed prediction results with errors")
    print(f"- {author_name}_ensemble_alfalfa_yield_models_{timestamp}.joblib - Saved model package for future use")
    print(f"- {author_name}_ensemble_config_{timestamp}.json - Ensemble configuration details")

    # Return the enhanced ensemble and results
    results = {
        "model_comparison": {name: {"RMSE": np.sqrt(mean_squared_error(y_test, preds)), "R2": r2_score(y_test, preds), "MAE": mean_absolute_error(y_test, preds)} for name, preds in models.items()},
        "experiment_metrics": exp_metrics,
        "test_predictions": final_preds,
        "timestamp": timestamp,
    }

    return model_package, results


# Function to make predictions with the ensemble
def predict_with_ensemble(model_package, new_data):
    """
    Make predictions using the trained ensemble model

    Parameters:
    -----------
    model_package : dict
        Dictionary containing all trained models and preprocessing components
    new_data : DataFrame
        New data to make predictions on

    Returns:
    --------
    predictions : array
        Predicted yield values
    """
    # Extract components
    rf = model_package["rf"]
    xgb_model = model_package["xgb"]
    en = model_package["en"]
    meta_final = model_package["enhanced"]
    preprocessor = model_package["preprocessor"]
    scaler = model_package["scaler"]
    poly = model_package["poly"]

    # Preprocess new data
    X_processed = preprocessor.transform(new_data)
    X_scaled = scaler.transform(X_processed)

    # Generate base model predictions
    rf_preds = rf.predict(X_scaled).reshape(-1, 1)
    xgb_preds = xgb_model.predict(X_scaled).reshape(-1, 1)

    # For ElasticNet, need to create polynomial features
    if X_scaled.shape[1] > 100:
        # This assumes the same feature selection logic was used during training
        feature_selector = RandomForestRegressor(n_estimators=50, random_state=42)
        feature_selector.fit(X_scaled, np.zeros(X_scaled.shape[0]))  # Just to get feature importance
        top_indices = np.argsort(feature_selector.feature_importances_)[-50:]
        X_for_poly = X_scaled[:, top_indices]
        poly_features = poly.transform(X_for_poly)
        X_poly = np.hstack([X_scaled, poly_features])
    else:
        X_poly = poly.transform(X_scaled)

    en_preds = en.predict(X_poly).reshape(-1, 1)

    # Combine features for meta-model
    X_meta = np.hstack([X_scaled, rf_preds, xgb_preds, en_preds])

    # Make final prediction with meta-model
    final_preds = meta_final.predict(X_meta)

    return final_preds


# Example usage:
if __name__ == "__main__":
    # Define train and test experiments
    train_experiments = [
        "BushlandCenterPivot_2022",
        "BushlandLysimeters_1996",
        "BushlandLysimeters_1997",
        "BushlandLysimeters_1998",
        "Fallon_1973_Final",
        "Fallon_1974_Final",
        "Fallon_1975_Final",
        "Fallon_1976_Final",
        "Fallon_1977_Final",
        "Fallon_1981_Final",
        "RenoDripIrrigation_2021",
        "RenoDripIrrigation_2022",
        "RenoLinearMoveIrrigation_2023",
        "Fallon_1978_Final",
        "BushlandLysimeters_1999",
        "Fallon_1982",
    ]

    test_experiments = ["BushlandCenterPivot_2023", "RenoDripIrrigation_2023"]

    # Train and evaluate ensemble model with your name
    model_package, results = train_test_ensemble("Merge_Allcombine_Data_With_Location_Year.csv", train_experiments, test_experiments, author_name="Jenny")  # Replace with your actual name
