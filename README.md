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

## Models Implemented

The repository contains several machine learning approaches for yield prediction:

1. **FM-like Random Forest (alfa_prediction_classified_fmrf.py)** - ⭐ **Best Performing Model** ⭐
   - Combines Factorization Machine (FM) style interaction features with Random Forest or ElasticNet
   - Automatically selects the best performing approach between ElasticNet and Random Forest
   - Incorporates polynomial feature interactions to capture complex relationships

2. **Ensemble Model (alfa_prediction_classified_integrated.py)**
   - Stacking ensemble that combines FM-like, XGBoost, and Random Forest approaches
   - Uses meta-learning to optimize the combination of base learners

3. **Neural Network (alfa_prediction_classified_nn.py)**
   - Multi-layer perceptron with regularization techniques
   - Adaptively sizes the network based on available data volume

4. **Random Forest (alfa_prediction_classified_rf.py)**
   - Tree-based ensemble with hyperparameter optimization
   - Includes feature importance analysis

5. **Support Vector Machine (alfa_prediction_classified_svm.py)**
   - Optimized kernel selection and hyperparameter tuning
   - Two-phase tuning approach for more efficient parameter search

6. **XGBoost (alfa_prediction_classified_xgboost.py)**
   - Gradient boosting implementation with feature selection
   - Includes detailed error analysis by experiment

## Key Features of the Best Model (FM-like RF)

The `alfa_prediction_classified_fmrf.py` model achieves the best performance through several key innovations:

- **Feature Engineering**: Creates domain-specific features like seasonal transformations (sine/cosine of day of year) and location/irrigation type indicators
- **Interaction Terms**: Generates pairwise interaction features similar to Factorization Machines to capture complex relationships
- **Adaptive Complexity**: Automatically limits feature interactions for large feature spaces to prevent memory issues
- **Model Selection**: Dynamically selects between ElasticNet (for linear relationships with interactions) and Random Forest (for non-linear patterns)
- **Robust Preprocessing**: Handles missing values with context-aware strategies (e.g., zeros for missing irrigation amounts)
- **Detailed Evaluation**: Provides experiment-specific performance metrics and visualizations for error analysis

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
- tensorflow (for neural network model only)
- xgboost (for XGBoost and ensemble models only)

## Future Improvements

- Integration with weather forecast data for forward prediction
- Additional feature engineering for irrigation efficiency metrics
- Time-series modeling to account for temporal dependencies
- Deployment pipeline for real-time prediction during growing season
