# -*- coding: utf-8 -*-
"""
Time Series Analysis Module for Waste Management Data
Handles feature engineering, data aggregation, and time series operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesAnalyzer:
    """Class for time series analysis operations"""
    
    def __init__(self):
        self.hourly_data = None
    
    def create_hourly_data(self, df):
        """
        Create hourly aggregated data from raw data
        
        Args:
            df (pd.DataFrame): Raw data with Timestamp, Tempat, Jumlah, Kategori columns
            
        Returns:
            pd.DataFrame: Hourly aggregated data
        """
        # Set timestamp as index for resampling
        df_indexed = df.set_index("Timestamp")
        
        # Create hourly aggregation
        df_hourly = (
            df_indexed
            .groupby("Tempat")
            .resample("H")
            .agg({
                "Jumlah": "sum",
                "Kategori": lambda x: x.mode().iat[0] if len(x.dropna()) > 0 else "Unknown"
            })
            .reset_index()
        )
        
        # Create full time index for all places
        all_places = df_hourly["Tempat"].unique()
        tmin = df_hourly["Timestamp"].min()
        tmax = df_hourly["Timestamp"].max()
        full_index = pd.date_range(start=tmin.floor("H"), end=tmax.ceil("H"), freq="H")
        
        # Create complete time series for all places
        full = []
        for p in all_places:
            tmp = pd.DataFrame({"Timestamp": full_index})
            tmp["Tempat"] = p
            full.append(tmp)
        full_df = pd.concat(full, ignore_index=True)
        
        # Merge with actual data and fill missing values
        df_hourly = full_df.merge(df_hourly, on=["Tempat", "Timestamp"], how="left")
        df_hourly["Jumlah"] = df_hourly["Jumlah"].fillna(0)
        df_hourly["Kategori"] = df_hourly["Kategori"].fillna("Unknown")
        
        self.hourly_data = df_hourly
        return df_hourly
    
    def make_features(self, data, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Create features for machine learning model
        
        Args:
            data (pd.DataFrame): Data with Timestamp and Jumlah columns
            lags (list): List of lag periods to create
            roll_windows (list): List of rolling window sizes
            
        Returns:
            pd.DataFrame: Features dataframe
        """
        df = data.sort_values("Timestamp").set_index("Timestamp").copy()
        X = pd.DataFrame(index=df.index)
        X["y"] = df["Jumlah"]
        
        # Time-based features
        X["hour"] = X.index.hour
        X["dayofweek"] = X.index.dayofweek
        X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
        X["dayofmonth"] = X.index.day
        X["month"] = X.index.month
        
        # Lag features
        for l in lags:
            X[f"lag_{l}"] = X["y"].shift(l)
        
        # Rolling mean features
        for w in roll_windows:
            X[f"roll_mean_{w}"] = X["y"].shift(1).rolling(window=w, min_periods=1).mean()
        
        # Rolling standard deviation features
        for w in roll_windows:
            X[f"roll_std_{w}"] = X["y"].shift(1).rolling(window=w, min_periods=1).std()
        
        # Remove rows with NaN values
        X = X.dropna()
        
        return X
    
    def get_place_time_series(self, place, hourly_data=None):
        """
        Get time series data for a specific place
        
        Args:
            place (str): Name of the place
            hourly_data (pd.DataFrame): Hourly data (optional)
            
        Returns:
            pd.DataFrame: Time series data for the place
        """
        if hourly_data is None:
            hourly_data = self.hourly_data
        
        if hourly_data is None:
            return pd.DataFrame()
        
        return hourly_data[hourly_data["Tempat"] == place][["Timestamp", "Jumlah"]].copy()
    
    def calculate_time_series_stats(self, data, place=None):
        """
        Calculate time series statistics
        
        Args:
            data (pd.DataFrame): Time series data
            place (str): Name of the place (optional)
            
        Returns:
            dict: Time series statistics
        """
        if data.empty:
            return {"error": "No data available"}
        
        stats = {
            "place": place,
            "total_records": len(data),
            "date_range": {
                "start": data["Timestamp"].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": data["Timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            "waste_stats": {
                "mean": float(data["Jumlah"].mean()),
                "median": float(data["Jumlah"].median()),
                "std": float(data["Jumlah"].std()),
                "min": float(data["Jumlah"].min()),
                "max": float(data["Jumlah"].max()),
                "total": float(data["Jumlah"].sum())
            },
            "time_patterns": {
                "hourly_avg": data.groupby(data["Timestamp"].dt.hour)["Jumlah"].mean().to_dict(),
                "daily_avg": data.groupby(data["Timestamp"].dt.dayofweek)["Jumlah"].mean().to_dict(),
                "monthly_avg": data.groupby(data["Timestamp"].dt.month)["Jumlah"].mean().to_dict()
            }
        }
        
        return stats
    
    def detect_anomalies(self, data, method='iqr', threshold=1.5):
        """
        Detect anomalies in time series data
        
        Args:
            data (pd.DataFrame): Time series data
            method (str): Method for anomaly detection ('iqr', 'zscore')
            threshold (float): Threshold for anomaly detection
            
        Returns:
            pd.DataFrame: Data with anomaly flags
        """
        df = data.copy()
        
        if method == 'iqr':
            Q1 = df["Jumlah"].quantile(0.25)
            Q3 = df["Jumlah"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df["is_anomaly"] = (df["Jumlah"] < lower_bound) | (df["Jumlah"] > upper_bound)
            
        elif method == 'zscore':
            mean_val = df["Jumlah"].mean()
            std_val = df["Jumlah"].std()
            df["zscore"] = abs((df["Jumlah"] - mean_val) / std_val)
            df["is_anomaly"] = df["zscore"] > threshold
        
        return df
    
    def create_forecast_features(self, data, forecast_hours=24, lags=[1, 24, 48], roll_windows=[3, 24]):
        """
        Create features for forecasting
        
        Args:
            data (pd.DataFrame): Historical data
            forecast_hours (int): Number of hours to forecast
            lags (list): List of lag periods
            roll_windows (list): List of rolling window sizes
            
        Returns:
            pd.DataFrame: Features for forecasting
        """
        df = data.sort_values("Timestamp").set_index("Timestamp").copy()
        results = []
        last_index = df.index.max()
        current_series = df["Jumlah"].copy()
        
        for i in range(1, forecast_hours + 1):
            cur_time = last_index + pd.Timedelta(hours=i)
            feat = {}
            
            # Time features
            feat["hour"] = cur_time.hour
            feat["dayofweek"] = cur_time.dayofweek
            feat["is_weekend"] = int(cur_time.dayofweek >= 5)
            feat["dayofmonth"] = cur_time.day
            feat["month"] = cur_time.month
            
            # Lag features
            for l in lags:
                lag_time = cur_time - pd.Timedelta(hours=l)
                feat[f"lag_{l}"] = current_series.reindex([lag_time]).iat[0] if lag_time in current_series.index else 0.0
            
            # Rolling features
            for w in roll_windows:
                window_start = cur_time - pd.Timedelta(hours=w)
                window_end = cur_time - pd.Timedelta(hours=1)
                roll_vals = current_series.reindex(pd.date_range(window_start, window_end, freq="H")).fillna(0)
                feat[f"roll_mean_{w}"] = roll_vals.mean()
                feat[f"roll_std_{w}"] = roll_vals.std()
            
            results.append(feat)
        
        return pd.DataFrame(results)
    
    def classify_waste_status(self, forecast_values, thresholds=[2, 5]):
        """
        Classify waste status based on forecast values
        
        Args:
            forecast_values (list): List of forecast values
            thresholds (list): Thresholds for classification
            
        Returns:
            list: List of status classifications
        """
        statuses = []
        for value in forecast_values:
            if value <= thresholds[0]:
                statuses.append("Sepi")
            elif value <= thresholds[1]:
                statuses.append("Sedang")
            else:
                statuses.append("Rame")
        
        return statuses
