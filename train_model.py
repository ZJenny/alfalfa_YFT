import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
from datetime import datetime
import json
import time

# -------------------------------------------------------------------
#  Import the preprocessing function from our other file
# -------------------------------------------------------------------
from preprocess_features import prepare_data


def run_model_training(X_train, y_train, X_test, y_test, categorical_cols, numeric_cols, test_data, author_name="your_name"):
    """
    Takes pre-processed data and runs the XGBoost GridSearchCV,
    evaluation, plotting, and model saving.
    """
    
    print("\n--- STARTING MODEL TRAINING & EVALUATION ---")

    # Generate timestamp for file versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------------------------
    #  Original Script Logic (from step 6. Feature Processing)
    # -------------------------------------------------------------------

    # Create preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    # For irrigation features, use 0 imputation; for others, use KNN imputation
    irrig_cols = [col for col in numeric_cols if "Irrig" in col]
    other_numeric_cols = [col for col in numeric_cols if col not in irrig_cols]

    # Create column transformer with different preprocessing for different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", KNNImputer(n_neighbors=5), other_numeric_cols),
            ("irrig", SimpleImputer(strategy="constant", fill_value=0), irrig_cols),
        ]
    )

    # 7. Process train and test data
    # Apply preprocessing
    print("Preprocessing features (fitting ColumnTransformer)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # 8. Train XGBoost with GridSearchCV
    print("Training XGBoost with GridSearchCV...")

    # Set up parameter grid for XGBoost
    param_grid_xgb = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7, 9],
                      "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}

    # Initialize XGBoost model
    xgb_base = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Set up GridSearchCV
    grid_search_xgb = GridSearchCV(estimator=xgb_base, param_grid=param_grid_xgb, cv=5, n_jobs=1,
                                   scoring="neg_mean_squared_error", verbose=1)

    # Fit GridSearchCV to find the best model
    grid_search_xgb.fit(X_train_scaled, y_train)

    # Get the best parameters and model
    best_params_xgb = grid_search_xgb.best_params_
    print(f"Best XGBoost parameters: {best_params_xgb}")

    # Use the best model from GridSearchCV
    xgb_model = grid_search_xgb.best_estimator_

    # 9. Model Evaluation
    print("Evaluating XGBoost model...")

    # Make predictions
    xgb_preds = xgb_model.predict(X_test_scaled)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    r2 = r2_score(y_test, xgb_preds)
    mae = mean_absolute_error(y_test, xgb_preds)

    print(f"XGBoost Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({"Model": ["XGBoost"], "RMSE": [rmse], "R2": [r2], "MAE": [mae]})
    metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_xgboost_model_metrics_{timestamp}.csv"), index=False)

    # 10. Visualizations and Analysis
    print("Generating plots and saving results...")
    
    # Get feature importance
    feature_importance = xgb_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Save feature importance CSV
    importance_df.to_csv(os.path.join(folder_results, f"{author_name}_xgboost_feature_importance_{timestamp}.csv"),
                         index=False)

    # Enhanced Feature Importance Plot
    plt.figure(figsize=(14, 10))
    n_features = min(20, len(importance_df))
    top_features = importance_df.head(n_features)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_features))
    bars = plt.barh(range(n_features), top_features['Importance'], color=colors, height=0.7)
    plt.yticks(range(n_features), top_features['Feature'], fontsize=12)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, i, f"{width:.3f}", va='center', fontsize=10)
    plt.xlabel('Importance', fontsize=14)
    plt.title('Top 20 Feature Importance', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_feature_importance_{timestamp}.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # Enhanced plot of actual vs predicted scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_test, xgb_preds, c=range(len(y_test)), cmap='viridis',
                          alpha=0.8, s=80, edgecolor='w', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Data Point Index', fontsize=12)
    min_val = min(min(y_test), min(xgb_preds))
    max_val = max(max(y_test), max(xgb_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
             alpha=0.7, label='Perfect Prediction')
    z = np.polyfit(y_test, xgb_preds, 1)
    p = np.poly1d(z)
    r2_plot = np.corrcoef(y_test, xgb_preds)[0, 1] ** 2
    plt.plot(np.sort(y_test), p(np.sort(y_test)), 'r-', linewidth=2,
             alpha=0.7, label=f'Regression Line (R² = {r2_plot:.2f})')
    plt.xlabel('Actual Yield (ton/ha)', fontsize=14)
    plt.ylabel('Predicted Yield (ton/ha)', fontsize=14)
    plt.title('Actual vs. Predicted Alfalfa Yield', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    buffer = 0.5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    x_pos = min_val + (max_val - min_val) * 0.75
    y_pos = min_val + (max_val - min_val) * 0.15
    plt.text(x_pos, y_pos, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_xgboost_actual_vs_predicted_{timestamp}.png"), dpi=300)
    plt.close()

    # Enhanced time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, 'o-', color='#3366CC', linewidth=2,
             markersize=7, label='True Yield', alpha=0.8)
    plt.plot(range(len(xgb_preds)), xgb_preds, 'x--', color='#FF9933', linewidth=2,
             markersize=7, label='Predicted Yield', alpha=0.8)
    plt.fill_between(range(len(y_test)), y_test, alpha=0.1, color='#3366CC')
    plt.xlabel('Data Point Index', fontsize=14)
    plt.ylabel('Yield (ton/ha)', fontsize=14)
    plt.title('True vs. Predicted Alfalfa Yield Across Data Points', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_yield_comparison_{timestamp}.png"), dpi=300)
    plt.close()

    # Calculate residuals
    residuals = y_test - xgb_preds

    # Create a dataframe for analysis
    error_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": xgb_preds,
            "Residual": residuals,
            # Use the 'test_data' object we passed in
            "Experiment": test_data["experiment_info"].values,
            "Year": test_data["exp_year"].values if "exp_year" in test_data.columns else None,
            "Month": test_data["month"].values if "month" in test_data.columns else None,
        }
    )

    # Enhanced residuals by experiment plot
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x="Experiment", y="Residual", data=error_df, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Prediction Residuals by Experiment (XGBoost Model)", fontsize=16)
    plt.xlabel("Experiment", fontsize=14)
    plt.ylabel("Residual", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_xgboost_residuals_by_experiment_{timestamp}.png"), dpi=300)
    plt.close()

    # Experiment-specific metrics
    print("\nExperiment-specific performance:")
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

    exp_metrics_df = pd.DataFrame.from_dict(exp_metrics, orient="index")
    exp_metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_xgboost_experiment_metrics_{timestamp}.csv"))

    # Save detailed results
    results_df = pd.DataFrame(
        {"Experiment": test_data["experiment_info"].values, "Actual_Yield": y_test.values, "Predicted_Yield": xgb_preds,
         "Error": residuals, "Percent_Error": (residuals / y_test.values) * 100}
    )
    results_df.to_csv(os.path.join(folder_results_csv, f"{author_name}_xgboost_prediction_results_{timestamp}.csv"),
                      index=False)

    # Save model and components
    model_package = {
        "xgb": xgb_model,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "timestamp": timestamp,
    }
    model_path = os.path.join(folder_saved_models, f"{author_name}_xgboost_alf_yield_model_{timestamp}.joblib")
    joblib.dump(model_package, model_path)

    # Save model configuration as JSON
    model_config = {
        "model_type": "XGBoost",
        "best_parameters": best_params_xgb,
        "features_count": X_train_scaled.shape[1],
        "performance": {"rmse": rmse, "r2": r2, "mae": mae},
        "timestamp": timestamp,
    }
    with open(os.path.join(folder_saved_models, f"{author_name}_xgboost_config_{timestamp}.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    print("\nXGBoost analysis complete!")
    print(f"- Model package saved to: {model_path}")
    print(f"- All plots and CSV results saved to 'plots/', 'results/', and 'results_csv/' folders.")

    results = {
        "model_performance": {"RMSE": rmse, "R2": r2, "MAE": mae},
        "experiment_metrics": exp_metrics,
        "test_predictions": xgb_preds,
        "timestamp": timestamp,
    }
    
    print("--- MODELING FINISHED ---")

    return model_package, results


# ===================================================================
#  SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":
    
    print("=============================================")
    print("=== STARTING ALFALFA YIELD MODEL PIPELINE ===")
    print("=============================================")
    
    start_time = time.time()
    
    # 1. Setup Environment
    
    # Create directories if they don't exist
    folder_results_csv = os.path.join(os.getcwd(), "results_csv")
    folder_saved_models = os.path.join(os.getcwd(), "saved_models")
    folder_png = os.path.join(os.getcwd(), "plots")
    folder_results = os.path.join(os.getcwd(), "results")
    
    for folder in [folder_saved_models, folder_results, folder_results_csv, folder_png]:
        os.makedirs(folder, exist_ok=True)

    # Set the number of threads for BLAS and OpenMP
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 2. Define Configuration
    DATA_FILE = "Merge_Allcombine_Data_With_Location_Year.csv"
    AUTHOR_NAME = "XGBoost_GridSearchCV"
    
    TRAIN_EXPERIMENTS = [
        "BushlandCenterPivot_2022", "BushlandLysimeters_1996", "BushlandLysimeters_1997",
        "BushlandLysimeters_1998", "Fallon_1973_Final", "Fallon_1974_Final",
        "Fallon_1975_Final", "Fallon_1976_Final", "Fallon_1977_Final",
        "Fallon_1981_Final", "RenoDripIrrigation_2021", "RenoDripIrrigation_2022",
        "BushlandCenterPivot_2023", "Fallon_1978_Final", "BushlandLysimeters_1999",
        "Fallon_1982",
    ]
    TEST_EXPERIMENTS = ["RenoLinearMoveIrrigation_2023", "RenoDripIrrigation_2023"]

    # 3. Run Preprocessing Step
    (
        X_train, y_train, 
        X_test, y_test, 
        categorical_cols, numeric_cols, 
        test_data_metadata
    ) = prepare_data(DATA_FILE, TRAIN_EXPERIMENTS, TEST_EXPERIMENTS)

    # 4. Run Model Training Step
    model_package, results = run_model_training(
        X_train, y_train, X_test, y_test, 
        categorical_cols, numeric_cols, 
        test_data_metadata, 
        author_name=AUTHOR_NAME
    )
    
    # 5. Final Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=============================================")
    print("===      PIPELINE RUN FINISHED            ===")
    print("=============================================")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("\nOverall Model Performance:")
    print(f"  RMSE: {results['model_performance']['RMSE']:.4f}")
    print(f"  R²:   {results['model_performance']['R2']:.4f}")
    print(f"  MAE:  {results['model_performance']['MAE']:.4f}")
    print("---------------------------------------------")
