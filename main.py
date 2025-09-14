# -*- coding: utf-8 -*-
"""
Main Flask Application for Waste Management Time Series Analysis
Run this file to start the Flask server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from werkzeug.utils import secure_filename

from modules.data_processor import DataProcessor
from modules.time_series_analyzer import TimeSeriesAnalyzer
from modules.ml_model import MLModel

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize modules
data_processor = DataProcessor()
time_series_analyzer = TimeSeriesAnalyzer()
ml_model = MLModel()

# Global variables to store data and models
current_data = None
trained_models = {}

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Waste Management Time Series Analysis API",
        "version": "1.0.0",
        "description": "REST API for waste management time series analysis and prediction",
        "endpoints": {
            "/upload": "POST - Upload CSV data file",
            "/data/info": "GET - Get data information",
            "/data/places": "GET - Get list of waste collection places",
            "/data/summary": "GET - Get data summary statistics",
            "/analysis/hourly": "GET - Get hourly aggregated data",
            "/analysis/plot/<place>": "GET - Get plot data for specific place",
            "/analysis/stats/<place>": "GET - Get time series statistics for specific place",
            "/model/train": "POST - Train ML model for predictions",
            "/model/predict/<place>": "GET - Get predictions for specific place",
            "/model/forecast/<place>": "GET - Get forecast for next N hours",
            "/model/status/<place>": "GET - Get status classification for place",
            "/model/info/<place>": "GET - Get model information for specific place"
        },
        "example_usage": {
            "upload_data": "POST /upload with CSV file",
            "get_places": "GET /data/places",
            "train_model": "POST /model/train with {\"place\": \"PlaceName\"}",
            "get_forecast": "GET /model/forecast/PlaceName?hours=24"
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload CSV data file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the uploaded data
            global current_data
            current_data = data_processor.load_and_process_data(filepath)
            
            # Validate data
            validation = data_processor.validate_data()
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'data_shape': current_data.shape,
                'columns': current_data.columns.tolist(),
                'validation': validation
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/data/info', methods=['GET'])
def get_data_info():
    """Get information about the current dataset"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    info = data_processor.get_data_info()
    return jsonify(info)

@app.route('/data/places', methods=['GET'])
def get_places():
    """Get list of waste collection places"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    places = data_processor.get_places()
    return jsonify({'places': places})

@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    """Get data summary statistics"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    summary = data_processor.get_summary_stats()
    return jsonify(summary)

@app.route('/analysis/hourly', methods=['GET'])
def get_hourly_analysis():
    """Get hourly aggregated data"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    
    # Convert to JSON serializable format
    hourly_data['Timestamp'] = hourly_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify({
        'hourly_data': hourly_data.to_dict('records'),
        'summary': {
            'total_records': len(hourly_data),
            'date_range': {
                'start': hourly_data['Timestamp'].min(),
                'end': hourly_data['Timestamp'].max()
            },
            'places_count': hourly_data['Tempat'].nunique()
        }
    })

@app.route('/analysis/plot/<place>', methods=['GET'])
def get_plot_data(place):
    """Get plot data for specific place"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    # Convert to JSON serializable format
    place_data['Timestamp'] = place_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify({
        'place': place,
        'data': place_data.to_dict('records'),
        'summary': {
            'total_records': len(place_data),
            'avg_waste': float(place_data['Jumlah'].mean()),
            'max_waste': float(place_data['Jumlah'].max()),
            'min_waste': float(place_data['Jumlah'].min())
        }
    })

@app.route('/analysis/stats/<place>', methods=['GET'])
def get_place_stats(place):
    """Get time series statistics for specific place"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    stats = time_series_analyzer.calculate_time_series_stats(place_data, place)
    return jsonify(stats)

@app.route('/model/train', methods=['POST'])
def train_model():
    """Train ML model for predictions"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    place = data.get('place')
    test_size = data.get('test_size', 0.2)
    lags = data.get('lags', [1, 24, 48])
    roll_windows = data.get('roll_windows', [3, 24])
    
    if not place:
        return jsonify({'error': 'Place parameter is required'}), 400
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    try:
        # Train the model
        model, metrics = ml_model.train_model(
            place_data, 
            place,
            test_size=test_size, 
            lags=lags, 
            roll_windows=roll_windows
        )
        trained_models[place] = model
        
        return jsonify({
            'message': f'Model trained successfully for {place}',
            'metrics': metrics,
            'training_data_size': len(place_data)
        })
    except Exception as e:
        return jsonify({'error': f'Error training model: {str(e)}'}), 400

@app.route('/model/predict/<place>', methods=['GET'])
def get_predictions(place):
    """Get predictions for specific place"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    try:
        # Get predictions
        predictions = ml_model.get_predictions(trained_models[place], place_data, place)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': f'Error getting predictions: {str(e)}'}), 400

@app.route('/model/forecast/<place>', methods=['GET'])
def get_forecast(place):
    """Get forecast for next N hours"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    # Get parameters from query string
    n_hours = request.args.get('hours', 24, type=int)
    lags = request.args.get('lags', [1, 24, 48])
    roll_windows = request.args.get('roll_windows', [3, 24])
    
    # Convert string parameters to lists if needed
    if isinstance(lags, str):
        lags = eval(lags)
    if isinstance(roll_windows, str):
        roll_windows = eval(roll_windows)
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    try:
        # Get forecast
        forecast = ml_model.get_forecast(
            trained_models[place], 
            place_data, 
            place,
            n_hours=n_hours,
            lags=lags,
            roll_windows=roll_windows
        )
        return jsonify(forecast)
    except Exception as e:
        return jsonify({'error': f'Error getting forecast: {str(e)}'}), 400

@app.route('/model/status/<place>', methods=['GET'])
def get_status_classification(place):
    """Get status classification for place"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    # Get parameters from query string
    n_hours = request.args.get('hours', 24, type=int)
    lags = request.args.get('lags', [1, 24, 48])
    roll_windows = request.args.get('roll_windows', [3, 24])
    
    # Convert string parameters to lists if needed
    if isinstance(lags, str):
        lags = eval(lags)
    if isinstance(roll_windows, str):
        roll_windows = eval(roll_windows)
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    try:
        # Get forecast with status classification
        forecast_with_status = ml_model.get_forecast_with_status(
            trained_models[place], 
            place_data, 
            place,
            n_hours=n_hours,
            lags=lags,
            roll_windows=roll_windows
        )
        return jsonify(forecast_with_status)
    except Exception as e:
        return jsonify({'error': f'Error getting status classification: {str(e)}'}), 400

@app.route('/model/info/<place>', methods=['GET'])
def get_model_info(place):
    """Get model information for specific place"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    model_info = ml_model.get_model_info(trained_models[place], place)
    return jsonify(model_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Waste Management Time Series Analysis API...")
    print("API Documentation available at: http://localhost:5000/")
    print("Upload your CSV file to: http://localhost:5000/upload")
    app.run(debug=True, host='0.0.0.0', port=5000)
