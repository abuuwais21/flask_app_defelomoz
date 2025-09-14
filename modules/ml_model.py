# -*- coding: utf-8 -*-
"""
Machine Learning Model Module for Waste Management Time Series Analysis
Handles model training, prediction, and forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

class MLModel:
    """Class for machine learning model operations"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = None
    
    def train_model(self, data, test_size=0.2, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Train a Random Forest model for time series prediction
        
        Args:
            data (pd.DataFrame): Time series data with Timestamp and Jumlah columns
            test_size (float): Proportion of data to use for testing
            lags (list): List of lag periods for features
            roll_windows (list): List of rolling window sizes for features
            
        Returns:
            tuple: (trained_model, metrics_dict)
        """
        from modules.time_series_analyzer import TimeSeriesAnalyzer
        
        # Create features
        ts_analyzer = TimeSeriesAnalyzer()
        features = ts_analyzer.make_features(data, lags=lags, roll_windows=roll_windows)
        
        if len(features) < 10:
            raise ValueError("Insufficient data for training. Need at least 10 records.")
        
        # Split data
        split_idx = int(len(features) * (1 - test_size))
        train = features.iloc[:split_idx]
        test = features.iloc[split_idx:]
        
        if len(train) < 5 or len(test) < 2:
            raise ValueError("Insufficient data after train/test split.")
        
        # Prepare features and target
        X_train = train.drop(columns=["y"])
        y_train = train["y"]
        X_test = test.drop(columns=["y"])
        y_test = test["y"]
        
        # Store feature columns for later use
        self.feature_columns = X_train.columns.tolist()
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = sqrt(mean_squared_error(y_train, train_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = sqrt(mean_squared_error(y_test, test_pred))
        
        metrics = {
            "train_mae": float(train_mae),
            "train_rmse": float(train_rmse),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "train_size": len(train),
            "test_size": len(test),
            "feature_importance": dict(zip(self.feature_columns, model.feature_importances_))
        }
        
        return model, metrics
    
    def get_predictions(self, model, data, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Get predictions for historical data
        
        Args:
            model: Trained model
            data (pd.DataFrame): Time series data
            lags (list): List of lag periods
            roll_windows (list): List of rolling window sizes
            
        Returns:
            dict: Predictions data
        """
        from modules.time_series_analyzer import TimeSeriesAnalyzer
        
        ts_analyzer = TimeSeriesAnalyzer()
        features = ts_analyzer.make_features(data, lags=lags, roll_windows=roll_windows)
        
        if features.empty:
            return {"error": "No features could be created from the data"}
        
        # Make predictions
        X = features.drop(columns=["y"])
        predictions = model.predict(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            "Timestamp": features.index,
            "Actual": features["y"],
            "Predicted": predictions
        })
        
        # Convert to JSON serializable format
        results["Timestamp"] = results["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "predictions": results.to_dict('records'),
            "metrics": {
                "mae": float(mean_absolute_error(features["y"], predictions)),
                "rmse": float(sqrt(mean_squared_error(features["y"], predictions)))
            }
        }
    
    def get_forecast(self, model, data, n_hours=24, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Get forecast for next n_hours
        
        Args:
            model: Trained model
            data (pd.DataFrame): Historical time series data
            n_hours (int): Number of hours to forecast
            lags (list): List of lag periods
            roll_windows (list): List of rolling window sizes
            
        Returns:
            dict: Forecast data
        """
        from modules.time_series_analyzer import TimeSeriesAnalyzer
        
        ts_analyzer = TimeSeriesAnalyzer()
        
        # Create forecast features
        forecast_features = ts_analyzer.create_forecast_features(
            data, forecast_hours=n_hours, lags=lags, roll_windows=roll_windows
        )
        
        # Make predictions
        predictions = model.predict(forecast_features)
        
        # Create forecast dataframe
        last_timestamp = data["Timestamp"].max()
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=n_hours,
            freq='H'
        )
        
        forecast_df = pd.DataFrame({
            "Timestamp": forecast_timestamps,
            "Forecast": predictions
        })
        
        # Convert to JSON serializable format
        forecast_df["Timestamp"] = forecast_df["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "forecast": forecast_df.to_dict('records'),
            "summary": {
                "forecast_hours": n_hours,
                "avg_forecast": float(np.mean(predictions)),
                "max_forecast": float(np.max(predictions)),
                "min_forecast": float(np.min(predictions))
            }
        }
    
    def get_forecast_with_status(self, model, data, n_hours=24, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Get forecast with status classification
        
        Args:
            model: Trained model
            data (pd.DataFrame): Historical time series data
            n_hours (int): Number of hours to forecast
            lags (list): List of lag periods
            roll_windows (list): List of rolling window sizes
            
        Returns:
            dict: Forecast data with status classification
        """
        from modules.time_series_analyzer import TimeSeriesAnalyzer
        
        ts_analyzer = TimeSeriesAnalyzer()
        
        # Get basic forecast
        forecast_result = self.get_forecast(model, data, n_hours, lags, roll_windows)
        
        # Add status classification
        forecast_data = forecast_result["forecast"]
        forecast_values = [item["Forecast"] for item in forecast_data]
        statuses = ts_analyzer.classify_waste_status(forecast_values)
        
        # Add status to forecast data
        for i, item in enumerate(forecast_data):
            item["Status"] = statuses[i]
        
        # Calculate status summary
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "forecast_with_status": forecast_data,
            "summary": {
                "forecast_hours": n_hours,
                "avg_forecast": float(np.mean(forecast_values)),
                "max_forecast": float(np.max(forecast_values)),
                "min_forecast": float(np.min(forecast_values)),
                "status_distribution": status_counts
            }
        }
    
    def get_model_info(self, model):
        """
        Get information about the trained model
        
        Args:
            model: Trained model
            
        Returns:
            dict: Model information
        """
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
        else:
            feature_importance = {}
        
        return {
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', 'N/A'),
            "max_depth": getattr(model, 'max_depth', 'N/A'),
            "feature_importance": feature_importance,
            "feature_columns": self.feature_columns
        }
    
    def save_model(self, model, place, filepath):
        """
        Save trained model to file
        
        Args:
            model: Trained model
            place (str): Name of the place
            filepath (str): Path to save the model
        """
        import joblib
        
        model_data = {
            'model': model,
            'place': place,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            tuple: (model, place, feature_columns)
        """
        import joblib
        
        model_data = joblib.load(filepath)
        return model_data['model'], model_data['place'], model_data['feature_columns']
