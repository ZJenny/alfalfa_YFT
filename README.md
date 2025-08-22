# # Alfalfa Yield Prediction Project

This repository contains a collection of machine learning models designed to predict alfalfa yield based on environmental, agronomic, and meteorological data. The models are built to support precision agriculture and resource management in alfalfa production systems.

## Project Overview

The goal of this project is to develop accurate predictive models for alfalfa dry yield (ton/ha) using data from multiple experiment locations across different years. The models leverage various features including:

- Day of year and seasonal information
- Location indicators (Bushland, Reno, Fallon)
- Irrigation type (Drip, Linear, Pivot, Lysimeter)
- Weather variables (temperature, precipitation, solar radiation, etc.)
- Alfalfa variety information
- Soil water deficit measurements


## Usage Example

```python
from alfa_prediction_classified_fmrf import train_test_fm_alternative

# Define train and test experiments
train_experiments = [
    "BushlandCenterPivot_2022",
    "BushlandLysimeters_1996",
    # Additional training experiments...
]

test_experiments = [
    "BushlandCenterPivot_2023", 
    "RenoDripIrrigation_2023"
]

# Train and evaluate the model
best_model, results = train_test_fm_alternative(
    "Merge_Allcombine_Data_With_Location_Year.csv", 
    train_experiments, 
    test_experiments, 
    author_name="your_name"
)
```

## Model Outputs
### Test
All models generate:

1. Trained model files saved with timestamps
2. Visualizations including:
   - Actual vs. predicted plots
   - Feature importance charts
   - Residual analysis by experiment
3. Detailed performance metrics:
   - Overall RMSE, R², MAE
   - Experiment-specific metrics
   - Prediction results with error analysis

## Performance Comparison

The FM-like model (`alfa_prediction_classified_fmrf.py`) consistently outperforms other approaches, particularly for cross-experiment prediction where data distributions may differ between training and testing sets. The model's ability to capture interaction effects while maintaining good generalization properties makes it particularly well-suited for agricultural yield prediction where complex environmental relationships impact crop performance.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- xgboost (for XGBoost and ensemble models only)

## Future Improvements

- Integration with weather forecast data for forward prediction
- Additional feature engineering for irrigation efficiency metrics
- Time-series modeling to account for temporal dependencies
- Deployment pipeline for real-time prediction during growing season
