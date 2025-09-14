# Waste Management Time Series Analysis Flask API

A REST API for waste management time series analysis and prediction, based on the original `main.py` analysis. This Flask application provides endpoints for data upload, analysis, and machine learning predictions.

## Features

- **Data Upload**: Upload CSV files with waste management data
- **Time Series Analysis**: Hourly data aggregation and feature engineering
- **Machine Learning**: Random Forest model training and prediction
- **Forecasting**: Predict waste levels for future hours
- **Status Classification**: Classify waste levels as "Sepi", "Sedang", or "Rame"
- **REST API**: Complete RESTful API with JSON responses

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python main.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Data Management
- `POST /upload` - Upload CSV data file
- `GET /data/info` - Get data information
- `GET /data/places` - Get list of waste collection places
- `GET /data/summary` - Get data summary statistics

### Analysis
- `GET /analysis/hourly` - Get hourly aggregated data
- `GET /analysis/plot/<place>` - Get plot data for specific place
- `GET /analysis/stats/<place>` - Get time series statistics for specific place

### Machine Learning
- `POST /model/train` - Train ML model for predictions
- `GET /model/predict/<place>` - Get predictions for specific place
- `GET /model/forecast/<place>` - Get forecast for next N hours
- `GET /model/status/<place>` - Get status classification for place
- `GET /model/info/<place>` - Get model information for specific place

## Usage Examples

### 1. Upload Data
```bash
curl -X POST -F "file=@tempat_sampah.csv" http://localhost:5000/upload
```

### 2. Get Available Places
```bash
curl http://localhost:5000/data/places
```

### 3. Train Model
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"place": "PlaceName"}' \
  http://localhost:5000/model/train
```

### 4. Get Forecast
```bash
curl "http://localhost:5000/model/forecast/PlaceName?hours=24"
```

### 5. Get Status Classification
```bash
curl "http://localhost:5000/model/status/PlaceName?hours=48"
```

## Data Format

The CSV file should contain the following columns:
- `Timestamp`: Date and time of the record
- `Tempat`: Location/place name
- `Jumlah`: Amount of waste
- `Kategori`: Category of waste

## Model Parameters

The Random Forest model uses the following features:
- **Time features**: hour, day of week, weekend flag
- **Lag features**: 1, 24, 48 hours ago
- **Rolling features**: 3 and 24 hour rolling means and standard deviations

## Status Classification

Waste levels are classified as:
- **Sepi**: ≤ 2 units
- **Sedang**: 2-5 units  
- **Rame**: > 5 units

## Project Structure

```
flask_app/
├── main.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── data_processor.py   # Data loading and preprocessing
│   ├── time_series_analyzer.py  # Time series analysis
│   └── ml_model.py        # Machine learning model
└── uploads/               # Directory for uploaded files
```

## Error Handling

The API includes comprehensive error handling:
- File validation
- Data validation
- Model training validation
- Proper HTTP status codes
- Detailed error messages

## Development

To run in development mode:
```bash
python main.py
```

The application will run with debug mode enabled and auto-reload on changes.

## Dependencies

- Flask 2.3.3
- Flask-CORS 4.0.0
- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
- matplotlib 3.7.2
- seaborn 0.12.2
- Werkzeug 2.3.7
- joblib 1.3.2
