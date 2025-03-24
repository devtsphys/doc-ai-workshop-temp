# Using scikit-learn for Time Series Forecasting
## A Comprehensive Guide to LinearModel and RandomForestRegressor

This guide demonstrates how to use scikit-learn's linear models and RandomForestRegressor for time series forecasting when you have features alongside your time series data.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Linear Models for Time Series](#linear-models-for-time-series)
- [Random Forest for Time Series](#random-forest-for-time-series)
- [Making Future Predictions](#making-future-predictions)
- [Performance Evaluation](#performance-evaluation)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Introduction

While specialized time series libraries exist, scikit-learn models can be effective for time series forecasting when you have:
- Feature-rich time series data
- Need for interpretable models
- Cases where traditional time series assumptions aren't met

This guide covers how to properly transform time series data for scikit-learn models and leverage LinearRegression and RandomForestRegressor for forecasting.

## Data Preparation

Time series data requires special preparation before using scikit-learn models:

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
```

### Step 2: Create Time Features

```python
def create_time_features(df, target_col):
    """
    Create time-based features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # Drop the original date column
    df.drop('date', axis=1, inplace=True)
    
    return df
```

### Step 3: Create Lag Features

```python
def add_lag_features(df, target_col, lag_list=[1, 7, 14, 30]):
    """
    Add lag features for the target column.
    """
    df = df.copy()
    for lag in lag_list:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df
```

### Step 4: Create Rolling Window Features

```python
def add_rolling_features(df, target_col, window_sizes=[7, 14, 30]):
    """
    Add rolling mean and std features for the target column.
    """
    df = df.copy()
    for window in window_sizes:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df
```

### Step 5: Prepare the Dataset

```python
def prepare_dataset(df, target_col, lag_list=[1, 7, 14, 30], window_sizes=[7, 14, 30],
                   test_size=0.2):
    """
    Prepare the dataset for modeling.
    """
    # Create features
    df = create_time_features(df, target_col)
    df = add_lag_features(df, target_col, lag_list)
    df = add_rolling_features(df, target_col, window_sizes)
    
    # Drop NA values caused by lagging and rolling operations
    df.dropna(inplace=True)
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Time series split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns
```

## Linear Models for Time Series

Linear models are simple yet effective for time series with features:

### Step 1: Train Linear Models

```python
def train_linear_models(X_train_scaled, y_train):
    """
    Train different linear models.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.1),
        'Lasso Regression': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
    }
    
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        
    return fitted_models
```

### Step 2: Evaluate Linear Models

```python
def evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Evaluate the performance of the models.
    """
    results = {}
    for name, model in models.items():
        # Training metrics
        y_pred_train = model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Testing metrics
        y_pred_test = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    return pd.DataFrame(results).T
```

### Step 3: Analyze Feature Importance (Linear Models)

```python
def analyze_linear_feature_importance(model, feature_names):
    """
    Analyze feature importance for linear models.
    """
    # Get coefficients
    coefficients = model.coef_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient values
    importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
    
    return importance_df
```

## Random Forest for Time Series

Random Forest can capture non-linear relationships in time series data:

### Step 1: Train Random Forest

```python
def train_random_forest(X_train_scaled, y_train):
    """
    Train a RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    return model
```

### Step 2: Analyze Feature Importance (Random Forest)

```python
def analyze_rf_feature_importance(model, feature_names):
    """
    Analyze feature importance for Random Forest model.
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df
```

### Step 3: Cross-Validation for Time Series

```python
def time_series_cv(X, y, model, n_splits=5):
    """
    Perform time series cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
        cv_scores.append(rmse)
    
    return cv_scores
```

## Making Future Predictions

To make future predictions with scikit-learn models:

### Step 1: Create Future Feature Data

```python
def create_future_features(last_date, periods, freq, last_known_values, feature_columns):
    """
    Create a DataFrame with future dates and features.
    
    Parameters:
    last_date : datetime
        Last date in the training data
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of the time series ('D', 'H', etc.)
    last_known_values : dict
        Dictionary with the last known values for lag features
    feature_columns : list
        List of feature column names
        
    Returns:
    DataFrame with future dates and features
    """
    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq[0].lower()),
                                 periods=periods, freq=freq)
    
    # Create DataFrame with future dates
    future_df = pd.DataFrame(index=future_dates)
    
    # Add time features
    future_df = create_time_features(future_df, target_col=None)
    
    # Initialize lag and rolling features with last known values
    for col in feature_columns:
        if col in future_df.columns:
            continue
        if col in last_known_values:
            future_df[col] = last_known_values[col]
        else:
            future_df[col] = np.nan
    
    return future_df
```

### Step 2: Make Recursive Multi-Step Predictions

```python
def forecast_future(model, scaler, last_date, periods, freq, df, 
                    target_col, feature_columns, lag_list=[1, 7, 14, 30]):
    """
    Make future predictions using the trained model.
    
    Parameters:
    model : trained model object
    scaler : fitted StandardScaler
    last_date : datetime
        Last date in the training data
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of the time series ('D', 'H', etc.)
    df : DataFrame
        Original DataFrame with historical data
    target_col : str
        Name of the target column
    feature_columns : list
        List of feature column names
    lag_list : list
        List of lag values used
        
    Returns:
    DataFrame with future predictions
    """
    # Get last known values for lag features
    last_known_values = {}
    for lag in lag_list:
        last_known_values[f'{target_col}_lag_{lag}'] = df[target_col].iloc[-lag]
    
    # Get last known values for rolling features
    window_sizes = [7, 14, 30]  # Adjust based on what was used in training
    for window in window_sizes:
        last_known_values[f'{target_col}_rolling_mean_{window}'] = df[target_col].iloc[-window:].mean()
        last_known_values[f'{target_col}_rolling_std_{window}'] = df[target_col].iloc[-window:].std()
    
    # Create future features DataFrame
    future_df = create_future_features(last_date, periods, freq, last_known_values, feature_columns)
    
    # Make recursive predictions
    predictions = []
    for i in range(periods):
        # Scale features for current step
        current_features = future_df.iloc[i:i+1]
        current_features_scaled = scaler.transform(current_features)
        
        # Make prediction
        prediction = model.predict(current_features_scaled)[0]
        predictions.append(prediction)
        
        # Update lag features for next step
        if i + 1 < periods:
            for lag in lag_list:
                if lag == 1:
                    future_df.iloc[i+1, future_df.columns.get_loc(f'{target_col}_lag_{lag}')] = prediction
                elif i + 1 >= lag:
                    future_df.iloc[i+1, future_df.columns.get_loc(f'{target_col}_lag_{lag}')] = predictions[i+1-lag]
                # Else keep the initial values
            
            # Update rolling features if needed
            for window in window_sizes:
                if i + 1 >= window:
                    recent_preds = predictions[-(window-1):] + [prediction]
                    future_df.iloc[i+1, future_df.columns.get_loc(f'{target_col}_rolling_mean_{window}')] = np.mean(recent_preds)
                    future_df.iloc[i+1, future_df.columns.get_loc(f'{target_col}_rolling_std_{window}')] = np.std(recent_preds)
    
    # Create result DataFrame
    forecast_df = pd.DataFrame({
        'ds': future_df.index,
        'yhat': predictions
    })
    
    return forecast_df
```

### Step 3: Visualize Predictions

```python
def plot_forecast(history_df, forecast_df, target_col, figsize=(12, 6)):
    """
    Plot historical data and forecast.
    
    Parameters:
    history_df : DataFrame
        Historical data
    forecast_df : DataFrame
        Forecast data with 'ds' (date) and 'yhat' (prediction) columns
    target_col : str
        Name of the target column
    figsize : tuple
        Figure size
        
    Returns:
    matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Plot historical data
    plt.plot(history_df.index, history_df[target_col], label='Historical Data', color='blue')
    
    # Plot forecast
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='red')
    
    # Add confidence intervals if available
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        plt.fill_between(forecast_df['ds'], 
                         forecast_df['yhat_lower'], 
                         forecast_df['yhat_upper'], 
                         color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Add vertical line separating history and forecast
    plt.axvline(x=history_df.index[-1], color='black', linestyle='--')
    
    plt.title('Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()
```

## Performance Evaluation

Evaluate your time series forecasting models:

### Step 1: Time Series Performance Metrics

```python
def calculate_metrics(y_true, y_pred):
    """
    Calculate common time series forecasting metrics.
    
    Parameters:
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    dict with metrics
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
    }
    
    return metrics
```

### Step 2: Residual Analysis

```python
def plot_residuals(y_true, y_pred, figsize=(12, 8)):
    """
    Plot residual analysis.
    
    Parameters:
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size
        
    Returns:
    matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals vs. Fitted
    axes[0, 0].scatter(y_pred, residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs. Fitted')
    
    # Distribution of Residuals
    axes[0, 1].hist(residuals, bins=20)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_title('Distribution of Residuals')
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Residuals over time
    axes[1, 1].plot(residuals)
    axes[1, 1].axhline(y=0, color='r', linestyle='-')
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals over Time')
    
    plt.tight_layout()
    return fig
```

## Advanced Techniques

Enhancing model performance:

### Step 1: Hyperparameter Tuning

```python
def tune_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Tune RandomForestRegressor hyperparameters.
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    # Create model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create randomized search
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # Fit random search
    rf_random.fit(X_train_scaled, y_train)
    
    # Best parameters
    best_params = rf_random.best_params_
    
    # Get best model
    best_model = rf_random.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return best_model, best_params, rmse
```

### Step 2: Feature Selection

```python
def select_features(X_train_scaled, y_train, X_test_scaled, feature_names, threshold=0.01):
    """
    Select features based on RandomForestRegressor importance.
    """
    # Train a RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Select features above threshold
    selected_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()
    
    # Get indices of selected features
    selected_indices = [list(feature_names).index(feature) for feature in selected_features]
    
    # Filter X_train and X_test
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    return X_train_selected, X_test_selected, selected_features
```

### Step 3: Model Stacking

```python
def stack_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Create a stacked model using LinearRegression and RandomForestRegressor.
    """
    # Create base models
    base_models = [
        ('lr', LinearRegression()),
        ('ridge', Ridge(alpha=0.1)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    
    # Create meta-learner
    meta_model = LinearRegression()
    
    # Create training predictions for meta-learner
    meta_X_train = np.zeros((X_train_scaled.shape[0], len(base_models)))
    
    # Use TimeSeriesSplit for out-of-fold predictions
    tscv = TimeSeriesSplit(n_splits=5)
    for i, (name, model) in enumerate(base_models):
        oof_preds = np.zeros(X_train_scaled.shape[0])
        
        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx] = model.predict(X_val_fold)
        
        meta_X_train[:, i] = oof_preds
    
    # Train meta-learner
    meta_model.fit(meta_X_train, y_train)
    
    # Train base models on full data
    base_models_fitted = []
    for name, model in base_models:
        model.fit(X_train_scaled, y_train)
        base_models_fitted.append((name, model))
    
    # Create test predictions
    meta_X_test = np.zeros((X_test_scaled.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models_fitted):
        meta_X_test[:, i] = model.predict(X_test_scaled)
    
    # Make final prediction
    final_preds = meta_model.predict(meta_X_test)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    
    return base_models_fitted, meta_model, rmse
```

## Troubleshooting

Common issues and how to fix them:

### Poor Model Performance

1. **Check Feature Importance**:
   - Use `analyze_linear_feature_importance` or `analyze_rf_feature_importance` to identify weak features
   - Remove features with very low importance

2. **Non-stationarity Issues**:
   ```python
   from statsmodels.tsa.stattools import adfuller
   
   def check_stationarity(timeseries):
       """
       Perform Augmented Dickey-Fuller test to check stationarity.
       """
       result = adfuller(timeseries)
       print(f'ADF Statistic: {result[0]}')
       print(f'p-value: {result[1]}')
       print('Critical Values:')
       for key, value in result[4].items():
           print(f'\t{key}: {value}')
       
       if result[1] <= 0.05:
           print("Stationary (reject H0)")
       else:
           print("Non-stationary (fail to reject H0)")
   ```

3. **Differencing for Non-stationary Data**:
   ```python
   def difference_series(df, column, interval=1):
       """
       Create differenced series.
       """
       df[f'{column}_diff_{interval}'] = df[column].diff(intervals=interval)
       return df
   ```

### Forecasting Errors

1. **Check for data leakage**:
   - Ensure proper time series split
   - Verify that future information isn't leaking into features

2. **Validate recursive forecasting**:
   ```python
   def validate_recursive_forecast(model, scaler, df, target_col, 
                                  feature_columns, test_start_idx, 
                                  horizon=30, lag_list=[1, 7, 14, 30]):
       """
       Validate recursive forecasting accuracy by using a section of known data.
       """
       # Create a copy of the dataframe up to test_start_idx
       train_df = df.iloc[:test_start_idx].copy()
       actual_values = df.iloc[test_start_idx:test_start_idx+horizon][target_col]
       
       # Make forecast
       last_date = train_df.index[-1]
       forecast_df = forecast_future(model, scaler, last_date, horizon, 
                                    df.index.freq, train_df, target_col, 
                                    feature_columns, lag_list)
       
       # Calculate metrics
       metrics = calculate_metrics(actual_values, forecast_df['yhat'])
       
       # Plot comparison
       plt.figure(figsize=(10, 6))
       plt.plot(actual_values.index, actual_values, label='Actual')
       plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
       plt.title('Recursive Forecast Validation')
       plt.legend()
       
       return metrics, plt.gcf()
   ```

## Complete Example

Here's a complete example showing how to use the framework:

```python
# Load example data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample time series data
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
np.random.seed(42)

# Create target variable with seasonality and trend
trend = np.linspace(0, 15, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 2, 365)

sales = trend + seasonality + noise
sales = np.maximum(sales, 0)  # Ensure non-negative

# Create external features
temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 3, 365)
is_weekend = [(date.dayofweek >= 5).astype(int) for date in dates]
is_holiday = np.zeros(365)
is_holiday[[1, 359, 360]] = 1  # New Year's, Christmas

# Create DataFrame
df = pd.DataFrame({
    'sales': sales,
    'temperature': temperature,
    'is_weekend': is_weekend,
    'is_holiday': is_holiday
}, index=dates)

# Prepare the dataset
target_col = 'sales'
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns = prepare_dataset(
    df, target_col=target_col, test_size=0.2
)

# Train linear models
linear_models = train_linear_models(X_train_scaled, y_train)
linear_results = evaluate_models(linear_models, X_train_scaled, y_train, X_test_scaled, y_test)
print("Linear Models Results:")
print(linear_results)

# Train Random Forest
rf_model = train_random_forest(X_train_scaled, y_train)
rf_importance = analyze_rf_feature_importance(rf_model, feature_columns)
print("\nRandom Forest Feature Importance:")
print(rf_importance.head(10))

# Make forecast with best model
best_model = rf_model  # Or choose the best from linear_models based on results
forecast_horizon = 30
last_date = df.index[-1]
forecast_df = forecast_future(best_model, scaler, last_date, forecast_horizon, 'D', 
                             df, target_col, feature_columns)

# Plot the forecast
_ = plot_forecast(df, forecast_df, target_col)
```

This guide provides a comprehensive framework for using scikit-learn models for time series forecasting with features. By properly preparing your data and leveraging the power of these models, you can achieve accurate forecasts even without specialized time series libraries.
