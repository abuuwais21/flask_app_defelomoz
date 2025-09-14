# -*- coding: utf-8 -*-
"""
Test script to verify the feature consistency fix
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_processor import DataProcessor
from modules.time_series_analyzer import TimeSeriesAnalyzer
from modules.ml_model import MLModel

def test_feature_consistency():
    """Test that features are consistent between training and prediction"""
    
    print("Testing Feature Consistency Fix")
    print("=" * 40)
    
    # Create sample data
    print("1. Creating sample data...")
    start_date = datetime.now() - timedelta(days=7)
    dates = pd.date_range(start=start_date, periods=168, freq='H')  # 7 days, hourly
    
    data = []
    for date in dates:
        amount = max(0, 2 + np.sin(date.hour * np.pi / 12) + np.random.uniform(-0.5, 0.5))
        data.append({
            "Timestamp": date,
            "Jumlah": round(amount, 1)
        })
    
    df = pd.DataFrame(data)
    print(f"   Created {len(df)} records")
    
    # Initialize modules
    print("\n2. Initializing modules...")
    ts_analyzer = TimeSeriesAnalyzer()
    ml_model = MLModel()
    
    # Create features for training
    print("\n3. Creating features for training...")
    features = ts_analyzer.make_features(df, lags=[1, 24], roll_windows=[3, 24])
    print(f"   Training features shape: {features.shape}")
    print(f"   Training feature columns: {list(features.columns)}")
    
    # Train model
    print("\n4. Training model...")
    try:
        model, metrics = ml_model.train_model(df, "TestPlace", test_size=0.2, lags=[1, 24], roll_windows=[3, 24])
        print("   ✓ Model trained successfully")
        print(f"   Test MAE: {metrics['test_mae']:.3f}")
    except Exception as e:
        print(f"   ✗ Training failed: {str(e)}")
        return False
    
    # Test predictions
    print("\n5. Testing predictions...")
    try:
        predictions = ml_model.get_predictions(model, df, "TestPlace", lags=[1, 24], roll_windows=[3, 24])
        print("   ✓ Predictions successful")
        print(f"   Prediction MAE: {predictions['metrics']['mae']:.3f}")
    except Exception as e:
        print(f"   ✗ Predictions failed: {str(e)}")
        return False
    
    # Test forecast
    print("\n6. Testing forecast...")
    try:
        forecast = ml_model.get_forecast(model, df, "TestPlace", n_hours=12, lags=[1, 24], roll_windows=[3, 24])
        print("   ✓ Forecast successful")
        print(f"   Forecast hours: {forecast['summary']['forecast_hours']}")
        print(f"   Average forecast: {forecast['summary']['avg_forecast']:.2f}")
    except Exception as e:
        print(f"   ✗ Forecast failed: {str(e)}")
        return False
    
    # Test forecast with status
    print("\n7. Testing forecast with status...")
    try:
        forecast_status = ml_model.get_forecast_with_status(model, df, "TestPlace", n_hours=12, lags=[1, 24], roll_windows=[3, 24])
        print("   ✓ Forecast with status successful")
        print(f"   Status distribution: {forecast_status['summary']['status_distribution']}")
    except Exception as e:
        print(f"   ✗ Forecast with status failed: {str(e)}")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! Feature consistency fix is working.")
    return True

if __name__ == "__main__":
    success = test_feature_consistency()
    if not success:
        sys.exit(1)
