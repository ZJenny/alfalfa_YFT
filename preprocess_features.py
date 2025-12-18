import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
#  HELPER FUNCTIONS (Copied directly from original script)
# -------------------------------------------------------------------

def calculate_rolling_features_with_logic(data, dates, window_size=180):
    """
    Calculates rolling features with different logic for BushlandLysimeters vs other experiments.
    Now includes median in addition to min, max, mean, and sum.
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

    # A) BushlandLysimeters:
    for experiment in bushland_data["experiment_info"].unique():
        exp_group = bushland_data[bushland_data["experiment_info"] == experiment]

        for plot_id in exp_group["Plot"].unique():
            plot_data = exp_group[exp_group["Plot"] == plot_id]
            row_dict = {"experiment_info": experiment, "Plot": plot_id}

            for month in [1, 2, 3, 4, 5]:
                month_data = plot_data[plot_data["Timestamp"].dt.month == month]
                for wcol in weather_cols:
                    if wcol in month_data.columns:
                        row_dict.update(calc_stats(month_data[wcol], month))
                for pcol in plot_cols:
                    if pcol in month_data.columns:
                        row_dict.update(calc_stats(month_data[pcol], month))
            features_list.append(row_dict)

    # B) Non-BushlandLysimeters:
    for experiment in other_data["experiment_info"].unique():
        exp_group = other_data[other_data["experiment_info"] == experiment]
        group_weather_stats = {m: {} for m in [1, 2, 3, 4, 5]}
        for month in [1, 2, 3, 4, 5]:
            month_data_whole = exp_group[exp_group["Timestamp"].dt.month == month]
            for wcol in weather_cols:
                if wcol in month_data_whole.columns:
                    group_weather_stats[month].update(calc_stats(month_data_whole[wcol], month))

        for plot_id in exp_group["Plot"].unique():
            plot_data = exp_group[exp_group["Plot"] == plot_id]
            row_dict = {"experiment_info": experiment, "Plot": plot_id}

            for month in [1, 2, 3, 4, 5]:
                row_dict.update(group_weather_stats[month])
                month_data_plot = plot_data[plot_data["Timestamp"].dt.month == month]
                for pcol in plot_cols:
                    if pcol in month_data_plot.columns:
                        row_dict.update(calc_stats(month_data_plot[pcol], month))
            features_list.append(row_dict)

    monthly_features_df = pd.DataFrame(features_list)
    core_cols = ["experiment_info", "Plot"]
    other_cols = [c for c in monthly_features_df.columns if c not in core_cols]
    monthly_features_df = monthly_features_df[core_cols + other_cols]

    return monthly_features_df


def calculate_pre_harvest_features(data, harvest_dates):
    """
    Calculate features specifically for the 3 months prior to each harvest date.
    """
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    features_list = []
    weather_vars = ["Total Precipitation (mm)", "Average Air Temperature (Deg C)",
                    "Total Solar Radiation (kW-hr/m2)", "Average Wind Speed (m/s)",
                    "Relative Humidity (%)"]
    management_vars = ["Irrig. amount (in)", "Ave. SWD (%)"]
    plant_vars = ["Plant height (cm)"]

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

    for harvest_date in harvest_dates:
        yield_rows = data[(data["Timestamp"] == harvest_date) &
                          (data["Dry yield (ton/ha)"].notna())]
        if yield_rows.empty:
            continue

        for _, yield_row in yield_rows.iterrows():
            experiment_info = yield_row["experiment_info"]
            plot = yield_row["Plot"]
            plot_data = data[(data["experiment_info"] == experiment_info) &
                             (data["Plot"] == plot)]
            row_dict = {
                "experiment_info": experiment_info,
                "Plot": plot,
                "Timestamp": harvest_date
            }

            for month_offset in range(3):
                target_month = harvest_date.month - month_offset
                target_year = harvest_date.year
                if target_month <= 0:
                    target_month += 12
                    target_year -= 1

                month_data = plot_data[(plot_data["Timestamp"].dt.month == target_month) &
                                       (plot_data["Timestamp"].dt.year == target_year)]
                if month_data.empty:
                    continue
                month_label = f"M-{month_offset}"

                for var in weather_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))
                for var in management_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))
                for var in plant_vars:
                    if var in month_data.columns:
                        row_dict.update(calc_stats(month_data[var], month_label))

            if len(row_dict) > 3:
                features_list.append(row_dict)

    pre_harvest_df = pd.DataFrame(features_list)
    return pre_harvest_df


# -------------------------------------------------------------------
#  MAIN PREPROCESSING FUNCTION
# -------------------------------------------------------------------

def prepare_data(data_path, train_experiments, test_experiments):
    """
    Loads data, performs all feature engineering, and splits into
    train/test sets exactly as in the original script.
    
    Returns:
    X_train, y_train, X_test, y_test, categorical_cols, numeric_cols, test_data
    """
    
    print("--- STARTING PREPROCESSING ---")
    
    # Generate timestamp for file versioning
    print("Loading data...")
    data = pd.read_csv(data_path)

    # Drop specified columns if they exist
    columns_to_drop = ["Growth stage", "Leaf Area Index (LAI)"]
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 1. Basic Data Preparation
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["month"] = data["Timestamp"].dt.month
    data["year_numeric"] = data["Timestamp"].dt.year

    data_with_yield = data[data["Dry yield (ton/ha)"].notna()].copy()
    print(f"Working with {len(data_with_yield)} rows that have yield data")

    original_count = len(data_with_yield)
    data_with_yield = data_with_yield[data_with_yield["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)
    removed_count = original_count - len(data_with_yield)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with negative yield values")

    # 2. Basic Feature Engineering
    data["sin_day"] = np.sin(2 * np.pi * data["Day of Year"] / 365)
    data["cos_day"] = np.cos(2 * np.pi * data["Day of Year"] / 365)
    data["is_bushland"] = data["location"].str.contains("Bushland", case=False, na=False).astype(int)
    data["is_reno"] = data["location"].str.contains("Reno", case=False, na=False).astype(int)
    data["is_fallon"] = data["location"].str.contains("Fallon", case=False, na=False).astype(int)
    data["is_drip"] = data["experiment_info"].str.contains("Drip", case=False, na=False).astype(int)
    data["is_linear"] = data["experiment_info"].str.contains("Linear", case=False, na=False).astype(int)
    data["is_pivot"] = data["experiment_info"].str.contains("Pivot", case=False, na=False).astype(int)
    data["is_lysimeter"] = data["experiment_info"].str.contains("Lysimeter", case=False, na=False).astype(int)

    def extract_year(exp):
        parts = exp.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None
    data["exp_year"] = data["experiment_info"].apply(extract_year)

    # 3. Advanced Feature Engineering
    harvest_dates = data_with_yield["Timestamp"].unique()
    harvest_dates = pd.to_datetime(harvest_dates)

    print("Calculating rolling features with 180-day window...")
    rolling_features = calculate_rolling_features_with_logic(data, harvest_dates, window_size=180)

    print("Calculating monthly features for January to May...")
    monthly_features = calculate_monthly_features_by_experiment(data)

    print("Calculating 3-month pre-harvest features...")
    pre_harvest_features = calculate_pre_harvest_features(data, harvest_dates)

    print("Merging features...")
    merged_data = pd.merge(
        rolling_features,
        data[
            [
                "Timestamp", "Plot", "Dry yield (ton/ha)", "Fall Dormancy*", "Winterhardiness**",
                "Day of Year", "Alfalfa variety", "experiment_info", "sin_day", "cos_day",
                "is_bushland", "is_reno", "is_fallon", "is_drip", "is_linear", "is_pivot",
                "is_lysimeter", "exp_year",
            ]
        ],
        on=["Timestamp", "Plot", "experiment_info"], how="left",
    )
    merged_data = pd.merge(merged_data, monthly_features, on=["experiment_info", "Plot"], how="left", suffixes=("", "_monthly"))
    if not pre_harvest_features.empty:
        merged_data = pd.merge(
            merged_data, pre_harvest_features, on=["Timestamp", "Plot", "experiment_info"],
            how="left", suffixes=("", "_pre_harvest")
        )

    print(f"Before filtering: {len(merged_data)} rows in merged dataset")
    merged_data = merged_data.dropna(subset=["Dry yield (ton/ha)"])

    original_count = len(merged_data)
    merged_data = merged_data[merged_data["Dry yield (ton/ha)"] >= 0].reset_index(drop=True)
    removed_count = original_count - len(merged_data)
    if removed_count > 0:
        print(f"Removed {removed_count} additional rows with negative yield values")
    print(f"After filtering: {len(merged_data)} rows with valid yield data remaining")

    # 4. Handle Missing Values
    fully_missing_columns = merged_data.columns[merged_data.isna().all()]
    merged_data = merged_data.drop(columns=fully_missing_columns)
    irrig_columns = [col for col in merged_data.columns if "Irrig. amount" in col]
    merged_data[irrig_columns] = merged_data[irrig_columns].fillna(0)

    # 5. Train-Test Split
    print("Creating train/test split using specified experiments...")
    train_data = merged_data[merged_data["experiment_info"].isin(train_experiments)].copy()
    test_data = merged_data[merged_data["experiment_info"].isin(test_experiments)].copy()

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

    print("\nTrain experiments:")
    for exp in train_data["experiment_info"].unique():
        count = len(train_data[train_data["experiment_info"] == exp])
        print(f"  - {exp}: {count} samples")
    print("\nTest experiments:")
    for exp in test_data["experiment_info"].unique():
        count = len(test_data[test_data["experiment_info"] == exp])
        print(f"  - {exp}: {count} samples")

    # 6. Feature Processing (Define feature lists)
    categorical_cols = ["Alfalfa variety"]
    exclude_cols = ["Timestamp", "Plot", "experiment_info", "Dry yield (ton/ha)"]
    numeric_cols = [col for col in merged_data.select_dtypes(include=["float64", "int64"]).columns if
                    col not in exclude_cols and col != "Dry yield (ton/ha)"]

    # 7. Process train and test data (Extract X and y)
    X_train = train_data[categorical_cols + numeric_cols]
    y_train = train_data["Dry yield (ton/ha)"]
    X_test = test_data[categorical_cols + numeric_cols]
    y_test = test_data["Dry yield (ton/ha)"]

    print("--- PREPROCESSING FINISHED ---")
    
    # Return all necessary objects for the next step
    return X_train, y_train, X_test, y_test, categorical_cols, numeric_cols, test_data
