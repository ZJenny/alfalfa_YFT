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


def calculate_rolling_features_with_logic(data, dates, window_size=180):
    """
    Calculates rolling features with different logic for BushlandLysimeters vs other experiments.
    Now includes median in addition to min, max, mean, and sum.

    Parameters:
    -----------
    data : DataFrame
        Input data with Timestamp, Plot, weather and irrigation data
    dates : array-like
        Array of dates to calculate features for
    window_size : int
        Number of days to look back for rolling window calculations

    Returns:
    --------
    DataFrame
        DataFrame with rolling features
    """
    features_list = []

    # generate time-window description
    window_label = f"({window_size}d)"

    # Split data into BushlandLysimeter and other experiment groups based on location
    bushland_data = data[data["location"].str.contains("BushlandLysimeters", case=False, na=False)]
    other_data = data[~data["location"].str.contains("BushlandLysimeters", case=False, na=False)]

    # Process BushlandLysimeter data
    for date in dates:
        end_date = date
        start_date = date - pd.Timedelta(days=window_size)
        year = end_date.year  # Extract year from end_date

        mask = (bushland_data["Timestamp"] >= start_date) & (bushland_data["Timestamp"] <= end_date)
        window_data = bushland_data.loc[mask]
        weather_data = window_data.drop_duplicates(subset=["Timestamp"])  # Define weather_data

        for plot in window_data["Plot"].unique():
            plot_data = window_data[window_data["Plot"] == plot]

            # Calculate statistical features for BushlandLysimeter
            def calculate_stats(column):
                return {
                    "min": column.min(),
                    "max": column.max(),
                    "mean": column.mean(),
                    "median": column.median(),  # Added median calculation
                    "sum": column.sum()
                }

            stats = {
                "Total Precipitation (mm)": calculate_stats(plot_data["Total Precipitation (mm)"]),
                "Average Air Temperature (Deg C)": calculate_stats(plot_data["Average Air Temperature (Deg C)"]),
                "Total Solar Radiation (kW-hr/m2)": calculate_stats(plot_data["Total Solar Radiation (kW-hr/m2)"]),
                "Average Wind Speed (m/s)": calculate_stats(plot_data["Average Wind Speed (m/s)"]),
                "Relative Humidity (%)": calculate_stats(plot_data["Relative Humidity (%)"]),
                "Irrig. amount (in)": calculate_stats(plot_data["Irrig. amount (in)"]),
                "Ave. SWD (%)": calculate_stats(plot_data["Ave. SWD (%)"]),
                "Plant height (cm)": calculate_stats(plot_data["Plant height (cm)"]),
            }

            features_list.append(
                {
                    "Timestamp": end_date,
                    "Plot": plot,
                    "experiment_info": f"BushlandLysimeters_{year}",
                    **{f"{feature} {window_label} {stat}": value for feature, stats_dict in stats.items() for
                       stat, value in stats_dict.items()},
                }
            )

    # Process other experiment_info groups
    for experiment in other_data["experiment_info"].unique():
        experiment_group = other_data[other_data["experiment_info"] == experiment]

        for date in dates:
            end_date = date
            start_date = date - pd.Timedelta(days=window_size)

            mask = (experiment_group["Timestamp"] >= start_date) & (experiment_group["Timestamp"] <= end_date)
            window_data = experiment_group.loc[mask]

            # Weather features calculation (shared across all plots in the same experiment)
            weather_data = window_data.drop_duplicates(subset=["Timestamp"])

            def calculate_stats(column):
                return {
                    "min": column.min(),
                    "max": column.max(),
                    "mean": column.mean(),
                    "median": column.median(),  # Added median calculation
                    "sum": column.sum()
                }

            stats = {
                "Total Precipitation (mm)": calculate_stats(weather_data["Total Precipitation (mm)"]),
                "Average Air Temperature (Deg C)": calculate_stats(weather_data["Average Air Temperature (Deg C)"]),
                "Total Solar Radiation (kW-hr/m2)": calculate_stats(weather_data["Total Solar Radiation (kW-hr/m2)"]),
                "Average Wind Speed (m/s)": calculate_stats(weather_data["Average Wind Speed (m/s)"]),
                "Relative Humidity (%)": calculate_stats(weather_data["Relative Humidity (%)"]),
            }

            for plot in window_data["Plot"].unique():
                plot_data = window_data[window_data["Plot"] == plot]

                # Calculate plot-specific features
                plot_stats = {
                    "Irrig. amount (in)": calculate_stats(plot_data["Irrig. amount (in)"]),
                    "Ave. SWD (%)": calculate_stats(plot_data["Ave. SWD (%)"]),
                    "Plant height (cm)": calculate_stats(plot_data["Plant height (cm)"]),
                }

                features_list.append(
                    {
                        "Timestamp": end_date,
                        "Plot": plot,
                        "experiment_info": experiment,
                        **{f"{feature} {window_label} {stat}": value for feature, stats_dict in
                           {**stats, **plot_stats}.items() for stat, value in stats_dict.items()},
                    }
                )

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(features_list)
    return features_df


def calculate_monthly_features_by_experiment(data):
    """
    Calculates monthly (Jan~May) weather and plot-level features in "wide" format.
    Now includes median in addition to min, max, mean, and sum.

    - BushlandLysimeters:
      *All* features (weather + plot) are computed per Plot.

    - Non-BushlandLysimeters:
      Weather features are computed at the *entire experiment_info* level (shared by all plots).
      Plot-level features are computed per Plot.
    """

    # 1) Ensure Timestamp is datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"])

    # 2) Keep only Jan~May
    data = data[data["Timestamp"].dt.month.isin([1, 2, 3, 4, 5])]

    # 3) Split into Bushland and non-Bushland
    bushland_data = data[data["location"].str.contains("BushlandLysimeters", case=False, na=False)]
    other_data = data[~data["location"].str.contains("BushlandLysimeters", case=False, na=False)]

    # Specify which columns are considered "weather" vs. "plot-specific"
    weather_cols = ["Total Precipitation (mm)", "Average Air Temperature (Deg C)", "Total Solar Radiation (kW-hr/m2)",
                    "Average Wind Speed (m/s)", "Relative Humidity (%)"]
    plot_cols = ["Irrig. amount (in)", "Ave. SWD (%)", "Plant height (cm)"]

    # Helper: compute min, max, mean, sum for a single column's data
    def calc_stats(series, month):
        return {
            f"{series.name} (M{month}) min": series.min(),
            f"{series.name} (M{month}) max": series.max(),
            f"{series.name} (M{month}) mean": series.mean(),
            f"{series.name} (M{month}) median": series.median(),  # Added median calculation
            f"{series.name} (M{month}) sum": series.sum(),
        }

    features_list = []  # Will accumulate final row dicts

    # --------------------------------------------------------------------------
    # A) BushlandLysimeters:
    #    Weather + plot features both computed at the Plot level
    # --------------------------------------------------------------------------
    for experiment in bushland_data["experiment_info"].unique():
        exp_group = bushland_data[bushland_data["experiment_info"] == experiment]

        # Each row in final result => (experiment_info, Plot)
        for plot_id in exp_group["Plot"].unique():
            plot_data = exp_group[exp_group["Plot"] == plot_id]

            # We'll build one row with M1..M5 stats for this plot
            row_dict = {"experiment_info": experiment, "Plot": plot_id}

            # Loop months 1..5 and gather stats in "wide" columns
            for month in [1, 2, 3, 4, 5]:
                month_data = plot_data[plot_data["Timestamp"].dt.month == month]

                # (1) Weather columns at plot-level
                for wcol in weather_cols:
                    if wcol in month_data.columns:
                        row_dict.update(calc_stats(month_data[wcol], month))

                # (2) Plot-specific columns
                for pcol in plot_cols:
                    if pcol in month_data.columns:
                        row_dict.update(calc_stats(month_data[pcol], month))

            # After collecting all 5 months into row_dict, append once
            features_list.append(row_dict)

    # --------------------------------------------------------------------------------
    # B) Non-BushlandLysimeters:
    #    - Weather features = experiment_info level (shared by all plots)
    #    - Plot features = per Plot
    # --------------------------------------------------------------------------------
    for experiment in other_data["experiment_info"].unique():
        exp_group = other_data[other_data["experiment_info"] == experiment]

        # First, compute "group-level" weather stats for all 5 months
        # We'll store them in a dict keyed by month.
        group_weather_stats = {m: {} for m in [1, 2, 3, 4, 5]}
        for month in [1, 2, 3, 4, 5]:
            month_data_whole = exp_group[exp_group["Timestamp"].dt.month == month]
            for wcol in weather_cols:
                if wcol in month_data_whole.columns:
                    group_weather_stats[month].update(calc_stats(month_data_whole[wcol], month))

        # Now, for each Plot, compute its plot-specific stats (M1..M5)
        # and combine with the group-level weather stats
        for plot_id in exp_group["Plot"].unique():
            plot_data = exp_group[exp_group["Plot"] == plot_id]

            row_dict = {"experiment_info": experiment, "Plot": plot_id}

            for month in [1, 2, 3, 4, 5]:
                # Add the group's weather stats for this month
                row_dict.update(group_weather_stats[month])

                # Add this plot's stats for this month
                month_data_plot = plot_data[plot_data["Timestamp"].dt.month == month]
                for pcol in plot_cols:
                    if pcol in month_data_plot.columns:
                        row_dict.update(calc_stats(month_data_plot[pcol], month))

            # One row for this Plot
            features_list.append(row_dict)

    # 6) Convert list of dicts to a DataFrame (wide format)
    monthly_features_df = pd.DataFrame(features_list)

    # 7) Optional: reorder columns. For instance, keep 'experiment_info' and 'Plot' at front.
    core_cols = ["experiment_info", "Plot"]
    other_cols = [c for c in monthly_features_df.columns if c not in core_cols]
    monthly_features_df = monthly_features_df[core_cols + other_cols]

    return monthly_features_df


def calculate_pre_harvest_features(data, harvest_dates):
    """
    Calculate features specifically for the 3 months prior to each harvest date.
    This function focuses on critical pre-harvest growth periods.
    Now includes median in addition to min, max, mean, and sum.

    Parameters:
    -----------
    data : DataFrame
        Input data with Timestamp, Plot, and measurement data
    harvest_dates : array-like
        Array of harvest dates to calculate features for

    Returns:
    --------
    DataFrame
        Features for 3 months prior to harvest
    """
    # Ensure Timestamp is datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    # List to collect features
    features_list = []

    # Define variable groups
    weather_vars = ["Total Precipitation (mm)", "Average Air Temperature (Deg C)",
                    "Total Solar Radiation (kW-hr/m2)", "Average Wind Speed (m/s)",
                    "Relative Humidity (%)"]
    management_vars = ["Irrig. amount (in)", "Ave. SWD (%)"]
    plant_vars = ["Plant height (cm)"]

    # Helper: compute min, max, mean, sum for a column
    def calc_stats(series, month_label):
        if series.empty or series.isna().all():
            return {}
        return {
            f"{series.name} ({month_label}) min": series.min(),
            f"{series.name} ({month_label}) max": series.max(),
            f"{series.name} ({month_label}) mean": series.mean(),
            f"{series.name} ({month_label}) median": series.median(),  # Added median calculation
            f"{series.name} ({month_label}) sum": series.sum(),
        }

    # Process each harvest date
    for harvest_date in harvest_dates:
        # Find plots with yield data for this date
        yield_rows = data[(data["Timestamp"] == harvest_date) &
                          (data["Dry yield (ton/ha)"].notna())]

        # Skip if no yield data for this date
        if yield_rows.empty:
            continue

        # Process each plot with yield data
        for _, yield_row in yield_rows.iterrows():
            experiment_info = yield_row["experiment_info"]
            plot = yield_row["Plot"]

            # Get data for this experiment and plot
            plot_data = data[(data["experiment_info"] == experiment_info) &
                             (data["Plot"] == plot)]

            # Create row dictionary
            row_dict = {
                "experiment_info": experiment_info,
                "Plot": plot,
                "Timestamp": harvest_date
            }

            # Calculate features for 3 months prior to harvest
            for month_offset in range(3):  # 0 = harvest month, 1 = month before, etc.
                target_month = harvest_date.month - month_offset
                target_year = harvest_date.year

                # Handle cases where we go to previous year
                if target_month <= 0:
                    target_month += 12
                    target_year -= 1

                # Filter data for this specific month
                month_data = plot_data[(plot_data["Timestamp"].dt.month == target_month) &
                                       (plot_data["Timestamp"].dt.year == target_year)]

                # Skip if no data for this month
                if month_data.empty:
                    continue

                # Label for this month (M-0, M-1, M-2)
                month_label = f"M-{month_offset}"

                # Calculate weather variables
                for var in weather_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))

                # Calculate management variables
                for var in management_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))

                # Calculate plant variables - these are especially important pre-harvest
                for var in plant_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))

            # Add completed row if we have more than just the basic fields
            if len(row_dict) > 3:
                features_list.append(row_dict)

    # Convert to DataFrame
    pre_harvest_df = pd.DataFrame(features_list)

    return pre_harvest_df


def train_test_xgboost(data_path, train_experiments, test_experiments, author_name="your_name"):
    """
    Train and evaluate an XGBoost model for alfalfa yield prediction with advanced feature engineering

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
        Trained XGBoost model
    results : dict
        Dictionary containing performance metrics and predictions
    """
    # Generate timestamp for file versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    data = pd.read_csv(data_path)

    # Drop specified columns if they exist
    columns_to_drop = ["Growth stage", "Leaf Area Index (LAI)"]
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 1. Basic Data Preparation
    # ------------------

    # Convert timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["month"] = data["Timestamp"].dt.month
    data["year_numeric"] = data["Timestamp"].dt.year

    # Filter to only rows with yield data
    data_with_yield = data[data["Dry yield (ton/ha)"].notna()].copy()
    print(f"Working with {len(data_with_yield)} rows that have yield data")

    # Remove negative yield values (likely errors)
    original_count = len(data_with_yield)
    data_with_yield = data_with_yield[data_with_yield["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)
    removed_count = original_count - len(data_with_yield)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with negative yield values")

    # 2. Basic Feature Engineering
    # ---------------------

    # Create seasonal features using sine and cosine transformations
    data["sin_day"] = np.sin(2 * np.pi * data["Day of Year"] / 365)
    data["cos_day"] = np.cos(2 * np.pi * data["Day of Year"] / 365)

    # Location and irrigation type indicators
    data["is_bushland"] = data["location"].str.contains("Bushland", case=False, na=False).astype(int)
    data["is_reno"] = data["location"].str.contains("Reno", case=False, na=False).astype(int)
    data["is_fallon"] = data["location"].str.contains("Fallon", case=False, na=False).astype(int)
    data["is_drip"] = data["experiment_info"].str.contains("Drip", case=False, na=False).astype(int)
    data["is_linear"] = data["experiment_info"].str.contains("Linear", case=False, na=False).astype(int)
    data["is_pivot"] = data["experiment_info"].str.contains("Pivot", case=False, na=False).astype(int)
    data["is_lysimeter"] = data["experiment_info"].str.contains("Lysimeter", case=False, na=False).astype(int)

    # Extract year from experiment info
    def extract_year(exp):
        parts = exp.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None

    data["exp_year"] = data["experiment_info"].apply(extract_year)

    # 3. Advanced Feature Engineering
    # -----------------------------------------------

    # Identify harvest dates from data
    harvest_dates = data_with_yield["Timestamp"].unique()
    harvest_dates = pd.to_datetime(harvest_dates)

    # Calculate rolling features with 180-day window
    print("Calculating rolling features with 180-day window...")
    rolling_features = calculate_rolling_features_with_logic(data, harvest_dates, window_size=180)

    # Calculate monthly features for January to May
    print("Calculating monthly features for January to May...")
    monthly_features = calculate_monthly_features_by_experiment(data)

    # Calculate 3-month pre-harvest features
    print("Calculating 3-month pre-harvest features...")
    pre_harvest_features = calculate_pre_harvest_features(data, harvest_dates)

    # Merge all features together
    print("Merging features...")
    # First, merge rolling features with original data's yield and crop characteristics
    merged_data = pd.merge(
        rolling_features,
        data[
            [
                "Timestamp",
                "Plot",
                "Dry yield (ton/ha)",
                "Fall Dormancy*",
                "Winterhardiness**",
                "Day of Year",
                "Alfalfa variety",
                "experiment_info",
                "sin_day",
                "cos_day",
                "is_bushland",
                "is_reno",
                "is_fallon",
                "is_drip",
                "is_linear",
                "is_pivot",
                "is_lysimeter",
                "exp_year",
            ]
        ],
        on=["Timestamp", "Plot", "experiment_info"],
        how="left",
    )

    # Then merge with monthly features
    merged_data = pd.merge(merged_data, monthly_features, on=["experiment_info", "Plot"], how="left",
                           suffixes=("", "_monthly"))

    # Merge with pre-harvest features
    if not pre_harvest_features.empty:
        merged_data = pd.merge(
            merged_data,
            pre_harvest_features,
            on=["Timestamp", "Plot", "experiment_info"],
            how="left",
            suffixes=("", "_pre_harvest")
        )

    # Drop rows where target variable is missing or negative
    print(f"Before filtering: {len(merged_data)} rows in merged dataset")
    merged_data = merged_data.dropna(subset=["Dry yield (ton/ha)"])

    # Double-check to ensure no negative yield values remain
    original_count = len(merged_data)
    merged_data = merged_data[merged_data["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)
    removed_count = original_count - len(merged_data)
    if removed_count > 0:
        print(f"Removed {removed_count} additional rows with negative yield values")

    print(f"After filtering: {len(merged_data)} rows with valid yield data remaining")

    # 4. Handle Missing Values
    # ------------------------

    # Identify columns that are completely missing
    fully_missing_columns = merged_data.columns[merged_data.isna().all()]
    merged_data = merged_data.drop(columns=fully_missing_columns)

    # Find columns containing "Irrig. amount" and fill with 0
    irrig_columns = [col for col in merged_data.columns if "Irrig. amount" in col]
    merged_data[irrig_columns] = merged_data[irrig_columns].fillna(0)

    # 5. Train-Test Split
    # ------------------
    print("Creating train/test split using specified experiments...")

    train_data = merged_data[merged_data["experiment_info"].isin(train_experiments)].copy()
    test_data = merged_data[merged_data["experiment_info"].isin(test_experiments)].copy()

    # Final verification for negative yield values
    if (train_data["Dry yield (ton/ha)"] < 0).any():
        print("WARNING: Negative yield values found in training data!")
        print(f"Removing {(train_data['Dry yield (ton/ha)'] < 0).sum()} negative yield values from training data")
        train_data = train_data[train_data["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)

    if (test_data["Dry yield (ton/ha)"] < 0).any():
        print("WARNING: Negative yield values found in test data!")
        print(f"Removing {(test_data['Dry yield (ton/ha)'] < 0).sum()} negative yield values from test data")
        test_data = test_data[test_data["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)

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

    # 6. Feature Processing
    # -------------------

    # Identify categorical and numerical columns
    categorical_cols = ["Alfalfa variety"]

    # Define columns to exclude from features (metadata/identifiers)
    exclude_cols = ["Timestamp", "Plot", "experiment_info", "Dry yield (ton/ha)"]

    # All numeric columns except for excluded ones
    numeric_cols = [col for col in merged_data.select_dtypes(include=["float64", "int64"]).columns if
                    col not in exclude_cols and col != "Dry yield (ton/ha)"]

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
    # ----------------------------

    # Extract X (features) and y (target)
    X_train = train_data[categorical_cols + numeric_cols]
    y_train = train_data["Dry yield (ton/ha)"]

    X_test = test_data[categorical_cols + numeric_cols]
    y_test = test_data["Dry yield (ton/ha)"]

    # Apply preprocessing
    print("Preprocessing features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # 8. Train XGBoost with GridSearchCV
    # ------------------
    print("Training XGBoost with GridSearchCV...")

    # Set up parameter grid for XGBoost
    param_grid_xgb = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7, 9],
                      "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}

    # Initialize XGBoost model
    xgb_base = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Set up GridSearchCV
    grid_search_xgb = GridSearchCV(estimator=xgb_base, param_grid=param_grid_xgb, cv=5, n_jobs=-1,
                                   scoring="neg_mean_squared_error", verbose=1)

    # Fit GridSearchCV to find the best model
    grid_search_xgb.fit(X_train_scaled, y_train)

    # Get the best parameters and model
    best_params_xgb = grid_search_xgb.best_params_
    print(f"Best XGBoost parameters: {best_params_xgb}")

    # Use the best model from GridSearchCV
    xgb_model = grid_search_xgb.best_estimator_

    # 9. Model Evaluation
    # -----------------
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
    # ----------------------------------------------

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
    # Select top 20 features for better visualization
    n_features = min(20, len(importance_df))
    top_features = importance_df.head(n_features)

    # Create horizontal bar chart with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_features))
    bars = plt.barh(range(n_features), top_features['Importance'], color=colors, height=0.7)
    plt.yticks(range(n_features), top_features['Feature'], fontsize=12)

    # Add values to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, i, f"{width:.3f}", va='center', fontsize=10)

    # Add styling
    plt.xlabel('Importance', fontsize=14)
    plt.title('Top 20 Feature Importance', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_feature_importance_{timestamp}.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # Enhanced plot of actual vs predicted scatter
    plt.figure(figsize=(10, 8))

    # Create scatter plot with attractive styling
    scatter = plt.scatter(y_test, xgb_preds, c=range(len(y_test)), cmap='viridis',
                          alpha=0.8, s=80, edgecolor='w', linewidth=0.5)
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Data Point Index', fontsize=12)

    # Add the perfect prediction line (diagonal)
    min_val = min(min(y_test), min(xgb_preds))
    max_val = max(max(y_test), max(xgb_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
             alpha=0.7, label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(y_test, xgb_preds, 1)
    p = np.poly1d(z)
    r2_plot = np.corrcoef(y_test, xgb_preds)[0, 1] ** 2
    plt.plot(np.sort(y_test), p(np.sort(y_test)), 'r-', linewidth=2,
             alpha=0.7, label=f'Regression Line (R² = {r2_plot:.2f})')

    # Improve styling
    plt.xlabel('Actual Yield (ton/ha)', fontsize=14)
    plt.ylabel('Predicted Yield (ton/ha)', fontsize=14)
    plt.title('Actual vs. Predicted Alfalfa Yield', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Make the plot area slightly larger than the data range
    buffer = 0.5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    # Add RMSE and MAE metrics - MOVED TO BOTTOM RIGHT
    # Calculate position (75% from left, 15% from bottom)
    x_pos = min_val + (max_val - min_val) * 0.75
    y_pos = min_val + (max_val - min_val) * 0.15

    plt.text(x_pos, y_pos, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(os.path.join(folder_png, f"{author_name}_xgboost_actual_vs_predicted_{timestamp}.png"), dpi=300)
    plt.close()

    # Enhanced time series plot
    plt.figure(figsize=(12, 6))

    # Use more attractive colors and styling
    plt.plot(range(len(y_test)), y_test, 'o-', color='#3366CC', linewidth=2,
             markersize=7, label='True Yield', alpha=0.8)
    plt.plot(range(len(xgb_preds)), xgb_preds, 'x--', color='#FF9933', linewidth=2,
             markersize=7, label='Predicted Yield', alpha=0.8)

    # Add a light blue background area
    plt.fill_between(range(len(y_test)), y_test, alpha=0.1, color='#3366CC')

    # Improve styling
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

    # Save experiment-specific metrics
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

    # Save the model package
    model_path = os.path.join(folder_saved_models, f"{author_name}_xgboost_alfalfa_yield_model_{timestamp}.joblib")
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
    print("Check the following files:")
    print(f"- {author_name}_xgboost_actual_vs_predicted_{timestamp}.png - Plot of predictions vs actual values")
    print(f"- {author_name}_xgboost_residuals_by_experiment_{timestamp}.png - Box plot of residuals by experiment")
    print(f"- {author_name}_xgboost_prediction_results_{timestamp}.csv - Detailed prediction results with errors")
    print(f"- {author_name}_xgboost_alfalfa_yield_model_{timestamp}.joblib - Saved model package for future use")
    print(f"- {author_name}_xgboost_config_{timestamp}.json - Model configuration details")

    # Return the model and results
    results = {
        "model_performance": {"RMSE": rmse, "R2": r2, "MAE": mae},
        "experiment_metrics": exp_metrics,
        "test_predictions": xgb_preds,
        "timestamp": timestamp,
    }

    return model_package, results


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
        "BushlandCenterPivot_2023",
        "Fallon_1978_Final",
        "BushlandLysimeters_1999",
        "Fallon_1982",
    ]

    test_experiments = ["RenoLinearMoveIrrigation_2023", "RenoDripIrrigation_2023"]

    # Train and evaluate XGBoost model with advanced feature engineering
    model_package, results = train_test_xgboost("Merge_Allcombine_Data_With_Location_Year.csv", train_experiments,
                                                test_experiments, author_name="XGBoost_GridSearchCV")
