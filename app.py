# -*- coding: utf-8 -*-
"""
Flask REST API for Waste Management Time Series Analysis
Based on the original main.py analysis
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
        "endpoints": {
            "/upload": "POST - Upload CSV data file",
            "/data/info": "GET - Get data information",
            "/data/places": "GET - Get list of waste collection places",
            "/analysis/hourly": "GET - Get hourly aggregated data",
            "/analysis/plot/<place>": "GET - Get plot data for specific place",
            "/model/train": "POST - Train ML model for predictions",
            "/model/predict/<place>": "GET - Get predictions for specific place",
            "/model/forecast/<place>": "GET - Get forecast for next N hours",
            "/model/status/<place>": "GET - Get status classification for place"
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
        
        # Process the uploaded data
        global current_data
        current_data = data_processor.load_and_process_data(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'data_shape': current_data.shape,
            'columns': current_data.columns.tolist()
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/data/info', methods=['GET'])
def get_data_info():
    """Get information about the current dataset"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    info = {
        'shape': current_data.shape,
        'columns': current_data.columns.tolist(),
        'data_types': current_data.dtypes.to_dict(),
        'missing_values': current_data.isnull().sum().to_dict(),
        'sample_data': current_data.head().to_dict('records')
    }
    
    return jsonify(info)

@app.route('/data/places', methods=['GET'])
def get_places():
    """Get list of waste collection places"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    places = current_data['Tempat'].unique().tolist()
    return jsonify({'places': places})

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
            'avg_waste': place_data['Jumlah'].mean(),
            'max_waste': place_data['Jumlah'].max(),
            'min_waste': place_data['Jumlah'].min()
        }
    })

@app.route('/model/train', methods=['POST'])
def train_model():
    """Train ML model for predictions"""
    if current_data is None:
        return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    place = data.get('place')
    test_size = data.get('test_size', 0.2)
    
    if not place:
        return jsonify({'error': 'Place parameter is required'}), 400
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    # Train the model
    model, metrics = ml_model.train_model(place_data, test_size=test_size)
    trained_models[place] = model
    
    return jsonify({
        'message': f'Model trained successfully for {place}',
        'metrics': metrics,
        'training_data_size': len(place_data)
    })

@app.route('/model/predict/<place>', methods=['GET'])
def get_predictions(place):
    """Get predictions for specific place"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    # Get predictions
    predictions = ml_model.get_predictions(trained_models[place], place_data)
    
    return jsonify({
        'place': place,
        'predictions': predictions
    })

@app.route('/model/forecast/<place>', methods=['GET'])
def get_forecast(place):
    """Get forecast for next N hours"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    # Get parameters from query string
    n_hours = request.args.get('hours', 24, type=int)
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    # Get forecast
    forecast = ml_model.get_forecast(trained_models[place], place_data, n_hours=n_hours)
    
    return jsonify({
        'place': place,
        'forecast_hours': n_hours,
        'forecast': forecast
    })

@app.route('/model/status/<place>', methods=['GET'])
def get_status_classification(place):
    """Get status classification for place"""
    if place not in trained_models:
        return jsonify({'error': f'No trained model found for place: {place}. Please train the model first.'}), 404
    
    # Get parameters from query string
    n_hours = request.args.get('hours', 24, type=int)
    
    hourly_data = time_series_analyzer.create_hourly_data(current_data)
    place_data = hourly_data[hourly_data['Tempat'] == place][['Timestamp', 'Jumlah']].copy()
    
    if place_data.empty:
        return jsonify({'error': f'No data found for place: {place}'}), 404
    
    # Get forecast with status classification
    forecast_with_status = ml_model.get_forecast_with_status(trained_models[place], place_data, n_hours=n_hours)
    
    return jsonify({
        'place': place,
        'forecast_hours': n_hours,
        'forecast_with_status': forecast_with_status
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
