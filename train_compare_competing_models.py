import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import json
import time

# --- Scikit-learn Imports ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression  # For MLR
from sklearn.ensemble import RandomForestRegressor # For RF

# --- Import the data preparation function from your other file ---
from preprocess_features import prepare_data

# -------------------------------------------------------------------
#  Define global paths (same as your train_model.py)
# -------------------------------------------------------------------
folder_results_csv = os.path.join(os.getcwd(), "results_csv")
folder_saved_models = os.path.join(os.getcwd(), "saved_models")
folder_png = os.path.join(os.getcwd(), "plots")
folder_results = os.path.join(os.getcwd(), "results")


def get_experiment_metrics(y_test, preds, test_data_metadata):
    """
    Helper function to calculate and print metrics for each experiment.
    """
    error_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": preds,
        "Residual": y_test - preds,
        "Experiment": test_data_metadata["experiment_info"].values,
    })

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
        
    return exp_metrics, error_df

def save_model_package(model, preprocessor, scaler, author_name, timestamp, best_params, metrics):
    """
    Helper function to save the model, preprocessors, and config files.
    """
    # 1. Save the model package
    model_package = {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "timestamp": timestamp,
    }
    model_path = os.path.join(folder_saved_models, f"{author_name}_alf_yield_model_{timestamp}.joblib")
    joblib.dump(model_package, model_path)

    # 2. Save model configuration as JSON
    model_config = {
        "model_type": author_name,
        "best_parameters": best_params,
        "features_count": scaler.n_features_in_,
        "performance": metrics,
        "timestamp": timestamp,
    }
    with open(os.path.join(folder_saved_models, f"{author_name}_config_{timestamp}.json"), "w") as f:
        json.dump(model_config, f, indent=4)
        
    print(f"\nModel package saved to: {model_path}")

# -------------------------------------------------------------------
#  MODEL 1: MULTIPLE LINEAR REGRESSION (MLR)
# -------------------------------------------------------------------

def run_linear_regression(X_train, y_train, X_test, y_test, test_data_metadata, preprocessor, scaler, author_name):
    """
    Trains and evaluates a simple Linear Regression model as a baseline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Initialize and train the model
    print("Training Linear Regression model...")
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train) # Uses the pre-scaled data
    
    # 2. Make predictions
    print("Evaluating Linear Regression model...")
    mlr_preds = mlr_model.predict(X_test)
    
    # 3. Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(y_test, mlr_preds))
    r2 = r2_score(y_test, mlr_preds)
    mae = mean_absolute_error(y_test, mlr_preds)
    
    print(f"Linear Regression Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    model_performance = {"Model": "LinearRegression", "RMSE": rmse, "R2": r2, "MAE": mae}

    # 4. Calculate experiment-specific metrics
    exp_metrics, error_df = get_experiment_metrics(y_test, mlr_preds, test_data_metadata)

    # 5. Save results and model
    # Save metrics to CSV
    metrics_df = pd.DataFrame([model_performance])
    metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_model_metrics_{timestamp}.csv"), index=False)
    
    exp_metrics_df = pd.DataFrame.from_dict(exp_metrics, orient="index")
    exp_metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_experiment_metrics_{timestamp}.csv"))
    
    results_df = error_df.rename(columns={"Actual": "Actual_Yield", "Predicted": "Predicted_Yield", "Residual": "Error"})
    results_df["Percent_Error"] = (results_df["Error"] / results_df["Actual_Yield"]) * 100
    results_df.to_csv(os.path.join(folder_results_csv, f"{author_name}_prediction_results_{timestamp}.csv"), index=False)

    # Save model package
    save_model_package(mlr_model, preprocessor, scaler, author_name, timestamp, best_params="N/A (Linear)", metrics=model_performance)

    print("--- Linear Regression analysis complete! ---")
    return {"model_performance": model_performance, "timestamp": timestamp}


# -------------------------------------------------------------------
#  MODEL 2: RANDOM FOREST (RF)
# -------------------------------------------------------------------

def run_random_forest(X_train, y_train, X_test, y_test, test_data_metadata, preprocessor, scaler, author_name):
    """
    Trains and evaluates a Random Forest model with GridSearchCV.
    This logic is copied from your train_model.py for a fair comparison.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Setup GridSearchCV for Random Forest
    print("Training Random Forest with GridSearchCV...")
    
    # Define a parameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None], # 'None' means nodes expand until all leaves are pure
        'max_features': ['sqrt', 'log2', 1.0], # 1.0 is equivalent to 'auto' in older versions
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest model
    rf_base = RandomForestRegressor(random_state=42)
    
    # Set up GridSearchCV
    # We use n_jobs=-1 because RF is CPU-based, and this will not conflict with a GPU.
    # This will speed up the grid search significantly.
    grid_search_rf = GridSearchCV(estimator=rf_base, param_grid=param_grid_rf, cv=5, n_jobs=-1,
                                  scoring="neg_mean_squared_error", verbose=1)
                                  
    # Fit GridSearchCV
    grid_search_rf.fit(X_train, y_train) # Uses the pre-scaled data
    
    # Get the best model
    best_params_rf = grid_search_rf.best_params_
    print(f"Best Random Forest parameters: {best_params_rf}")
    rf_model = grid_search_rf.best_estimator_

    # 2. Make predictions
    print("Evaluating Random Forest model...")
    rf_preds = rf_model.predict(X_test)
    
    # 3. Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    r2 = r2_score(y_test, rf_preds)
    mae = mean_absolute_error(y_test, rf_preds)
    
    print(f"Random Forest Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")

    model_performance = {"Model": "RandomForest", "RMSE": rmse, "R2": r2, "MAE": mae}

    # 4. Calculate experiment-specific metrics
    exp_metrics, error_df = get_experiment_metrics(y_test, rf_preds, test_data_metadata)

    # 5. Save results and model (including plots, same as your XGB script)
    print("Generating plots and saving results for Random Forest...")
    
    # Save metrics
    metrics_df = pd.DataFrame([model_performance])
    metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_model_metrics_{timestamp}.csv"), index=False)
    
    exp_metrics_df = pd.DataFrame.from_dict(exp_metrics, orient="index")
    exp_metrics_df.to_csv(os.path.join(folder_results, f"{author_name}_experiment_metrics_{timestamp}.csv"))
    
    results_df = error_df.rename(columns={"Actual": "Actual_Yield", "Predicted": "Predicted_Yield", "Residual": "Error"})
    results_df["Percent_Error"] = (results_df["Error"] / results_df["Actual_Yield"]) * 100
    results_df.to_csv(os.path.join(folder_results_csv, f"{author_name}_prediction_results_{timestamp}.csv"), index=False)

    # Save Feature Importance plot
    try:
        feature_importance = rf_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(14, 10))
        n_features = min(20, len(importance_df))
        top_features = importance_df.head(n_features)
        colors = plt.cm.viridis(np.linspace(0, 0.8, n_features))
        plt.barh(range(n_features), top_features['Importance'], color=colors, height=0.7)
        plt.yticks(range(n_features), top_features['Feature'], fontsize=12)
        plt.xlabel('Importance', fontsize=14)
        plt.title('Top 20 Feature Importance (Random Forest)', fontsize=16)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_png, f"{author_name}_feature_importance_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    # Save Actual vs. Predicted plot
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, rf_preds, c=range(len(y_test)), cmap='viridis', alpha=0.8, s=80, edgecolor='w', linewidth=0.5)
        min_val = min(min(y_test), min(rf_preds))
        max_val = max(max(y_test), max(rf_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
        plt.xlabel('Actual Yield (ton/ha)', fontsize=14)
        plt.ylabel('Predicted Yield (ton/ha)', fontsize=14)
        plt.title('Actual vs. Predicted Yield (Random Forest)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        buffer = 0.5
        plt.xlim(min_val - buffer, max_val + buffer)
        plt.ylim(min_val - buffer, max_val + buffer)
        plt.text(0.95, 0.05, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='bottom', horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(folder_png, f"{author_name}_actual_vs_predicted_{timestamp}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not generate actual vs predicted plot: {e}")

    # Save Residuals by Experiment plot
    try:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x="Experiment", y="Residual", data=error_df, palette="viridis")
        plt.xticks(rotation=90)
        plt.title("Prediction Residuals by Experiment (Random Forest)", fontsize=16)
        plt.xlabel("Experiment", fontsize=14)
        plt.ylabel("Residual", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_png, f"{author_name}_residuals_by_experiment_{timestamp}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not generate residuals plot: {e}")
        
    # 6. Save model package
    save_model_package(rf_model, preprocessor, scaler, author_name, timestamp, best_params_rf, model_performance)

    print("--- Random Forest analysis complete! ---")
    return {"model_performance": model_performance, "timestamp": timestamp}


# ===================================================================
#  MAIN SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":
    
    print("======================================================")
    print("=== STARTING ALFALFA YIELD (MLR & RF) COMPARISON ===")
    print("======================================================")
    
    start_time = time.time()
    
    # 1. Setup Environment
    # (Create directories)
    for folder in [folder_saved_models, folder_results, folder_results_csv, folder_png]:
        os.makedirs(folder, exist_ok=True)

    # (Set thread counts for numpy/BLAS)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 2. Define Configuration (Same as your other script)
    DATA_FILE = "Merge_Allcombine_Data_With_Location_Year.csv"
    
    TRAIN_EXPERIMENTS = [
        "BushlandCenterPivot_2022", "BushlandLysimeters_1996", "BushlandLysimeters_1997",
        "BushlandLysimeters_1998", "Fallon_1973_Final", "Fallon_1974_Final",
        "Fallon_1975_Final", "Fallon_1976_Final", "Fallon_1977_Final",
        "Fallon_1981_Final", "RenoDripIrrigation_2021", "RenoDripIrrigation_2022",
        "BushlandCenterPivot_2023", "Fallon_1978_Final", "BushlandLysimeters_1999",
        "Fallon_1982",
    ]
    TEST_EXPERIMENTS = ["RenoLinearMoveIrrigation_2023", "RenoDripIrrigation_2023"]

    # 3. Run Preprocessing Step (from preprocess_features.py)
    (
        X_train, y_train, 
        X_test, y_test, 
        categorical_cols, numeric_cols, 
        test_data_metadata
    ) = prepare_data(DATA_FILE, TRAIN_EXPERIMENTS, TEST_EXPERIMENTS)

    # 4. Define and Fit the *exact same* Preprocessing Pipeline
    # This is CRITICAL for a fair comparison. All models must
    # be trained on identical data.
    print("\n--- FITTING PREPROCESSING PIPELINE ---")
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    
    irrig_cols = [col for col in numeric_cols if "Irrig" in col]
    other_numeric_cols = [col for col in numeric_cols if col not in irrig_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", KNNImputer(n_neighbors=5), other_numeric_cols),
            ("irrig", SimpleImputer(strategy="constant", fill_value=0), irrig_cols),
        ]
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    print("--- PREPROCESSING PIPELINE FITTED ---")


    # 5. Run Model Training Steps
    all_results = []

    # Run MLR
    print("\n" + "="*50)
    print("  RUNNING MODEL 1: MULTIPLE LINEAR REGRESSION (BASELINE)")
    print("="*50)
    mlr_run_results = run_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        test_data_metadata, preprocessor, scaler, 
        author_name="MLR_Baseline"
    )
    all_results.append(mlr_run_results['model_performance'])

    # Run RF
    print("\n" + "="*50)
    print("  RUNNING MODEL 2: RANDOM FOREST (COMPARISON)")
    print("="*50)
    rf_run_results = run_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        test_data_metadata, preprocessor, scaler, 
        author_name="RandomForest_Comparison"
    )
    all_results.append(rf_run_results['model_performance'])

    # 6. Final Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print("===      COMPARISON PIPELINE FINISHED       ===")
    print("="*50)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame(all_results).set_index("Model")
    
    print("\n--- OVERALL MODEL PERFORMANCE SUMMARY ---")
    print(summary_df.to_markdown(floatfmt=".4f"))
    print("---------------------------------------------")
