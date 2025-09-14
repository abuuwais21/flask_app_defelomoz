# -*- coding: utf-8 -*-
"""
Test script for Waste Management Time Series Analysis Flask API
This script demonstrates how to use the API endpoints
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:5000"

def test_api():
    """Test the API endpoints"""
    
    print("Testing Waste Management Time Series Analysis API")
    print("=" * 50)
    
    # Test 1: Check if API is running
    print("\n1. Testing API availability...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ API is running")
            print(f"  Response: {response.json()['message']}")
        else:
            print("✗ API is not responding")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure Flask app is running.")
        return
    
    # Test 2: Create sample data
    print("\n2. Creating sample data...")
    sample_data = create_sample_data()
    sample_data.to_csv("sample_waste_data.csv", index=False)
    print("✓ Sample data created: sample_waste_data.csv")
    
    # Test 3: Upload data
    print("\n3. Uploading sample data...")
    try:
        with open("sample_waste_data.csv", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            print("✓ Data uploaded successfully")
            print(f"  Shape: {response.json()['data_shape']}")
        else:
            print(f"✗ Upload failed: {response.json()['error']}")
            return
    except Exception as e:
        print(f"✗ Upload error: {str(e)}")
        return
    
    # Test 4: Get places
    print("\n4. Getting available places...")
    try:
        response = requests.get(f"{BASE_URL}/data/places")
        if response.status_code == 200:
            places = response.json()['places']
            print(f"✓ Found {len(places)} places: {places}")
            test_place = places[0] if places else "Place1"
        else:
            print(f"✗ Failed to get places: {response.json()['error']}")
            return
    except Exception as e:
        print(f"✗ Error getting places: {str(e)}")
        return
    
    # Test 5: Get data info
    print("\n5. Getting data information...")
    try:
        response = requests.get(f"{BASE_URL}/data/info")
        if response.status_code == 200:
            info = response.json()
            print("✓ Data info retrieved")
            print(f"  Columns: {info['columns']}")
            print(f"  Date range: {info['date_range']}")
        else:
            print(f"✗ Failed to get data info: {response.json()['error']}")
    except Exception as e:
        print(f"✗ Error getting data info: {str(e)}")
    
    # Test 6: Train model
    print(f"\n6. Training model for {test_place}...")
    try:
        data = {"place": test_place}
        response = requests.post(f"{BASE_URL}/model/train", json=data)
        if response.status_code == 200:
            print("✓ Model trained successfully")
            metrics = response.json()['metrics']
            print(f"  Test MAE: {metrics['test_mae']:.3f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.3f}")
        else:
            print(f"✗ Model training failed: {response.json()['error']}")
            return
    except Exception as e:
        print(f"✗ Error training model: {str(e)}")
        return
    
    # Test 7: Get forecast with status
    print(f"\n7. Getting forecast with status for {test_place}...")
    try:
        response = requests.get(f"{BASE_URL}/model/forecast/{test_place}?hours=12")
        if response.status_code == 200:
            forecast = response.json()
            print("✓ Forecast with status retrieved")
            print(f"  Forecast hours: {forecast['summary']['forecast_hours']}")
            print(f"  Average forecast: {forecast['summary']['avg_forecast']:.2f}")
            print(f"  Status distribution: {forecast['summary']['status_distribution']}")
        else:
            print(f"✗ Forecast failed: {response.json()['error']}")
    except Exception as e:
        print(f"✗ Error getting forecast: {str(e)}")
    
    # Test 8: Get status classification
    print(f"\n8. Getting status classification for {test_place}...")
    try:
        response = requests.get(f"{BASE_URL}/model/status/{test_place}?hours=24")
        if response.status_code == 200:
            status = response.json()
            print("✓ Status classification retrieved")
            print(f"  Status distribution: {status['summary']['status_distribution']}")
        else:
            print(f"✗ Status classification failed: {response.json()['error']}")
    except Exception as e:
        print(f"✗ Error getting status classification: {str(e)}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")

def create_sample_data():
    """Create sample waste management data"""
    
    # Create date range
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start=start_date, periods=720, freq='H')  # 30 days, hourly
    
    # Create sample places
    places = ["Place1", "Place2", "Place3"]
    categories = ["Organic", "Plastic", "Paper", "Metal"]
    
    data = []
    for date in dates:
        for place in places:
            # Simulate waste patterns
            base_amount = 2
            hour_factor = 1 + 0.5 * np.sin(date.hour * np.pi / 12)  # Daily pattern
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
            random_factor = np.random.uniform(0.5, 2.0)
            
            amount = max(0, base_amount * hour_factor * weekend_factor * random_factor)
            category = np.random.choice(categories)
            
            data.append({
                "Timestamp": date,
                "Tempat": place,
                "Jumlah": round(amount, 1),
                "Kategori": category
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    import numpy as np
    test_api()
