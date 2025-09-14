# Flask API Analisis Time Series Manajemen Sampah

REST API untuk analisis time series dan prediksi manajemen sampah, berdasarkan analisis `main.py` asli. Aplikasi Flask ini menyediakan endpoint untuk upload data, analisis, dan prediksi machine learning.

## Fitur

- **Upload Data**: Upload file CSV dengan data manajemen sampah
- **Analisis Time Series**: Agregasi data per jam dan feature engineering
- **Machine Learning**: Pelatihan model Random Forest dan prediksi
- **Forecasting**: Prediksi tingkat sampah untuk jam-jam mendatang
- **Klasifikasi Status**: Klasifikasi tingkat sampah sebagai "Sepi", "Sedang", atau "Rame"
- **REST API**: API RESTful lengkap dengan respons JSON

## Instalasi

1. Install dependensi yang diperlukan:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi Flask:
```bash
python main.py
```

API akan tersedia di `http://localhost:5000`

## Endpoint API

### Manajemen Data
- `POST /upload` - Upload file data CSV
- `GET /data/info` - Mendapatkan informasi data
- `GET /data/places` - Mendapatkan daftar tempat pengumpulan sampah
- `GET /data/summary` - Mendapatkan statistik ringkasan data

### Analisis
- `GET /analysis/hourly` - Mendapatkan data agregasi per jam
- `GET /analysis/plot/<place>` - Mendapatkan data plot untuk tempat tertentu
- `GET /analysis/stats/<place>` - Mendapatkan statistik time series untuk tempat tertentu

### Machine Learning
- `POST /model/train` - Melatih model ML untuk prediksi
- `GET /model/predict/<place>` - Mendapatkan prediksi untuk tempat tertentu
- `GET /model/forecast/<place>` - Mendapatkan forecast dengan klasifikasi status untuk N jam ke depan
- `GET /model/status/<place>` - Mendapatkan klasifikasi status untuk tempat
- `GET /model/info/<place>` - Mendapatkan informasi model untuk tempat tertentu

## Contoh Penggunaan

### 1. Upload Data
```bash
curl -X POST -F "file=@tempat_sampah.csv" http://localhost:5000/upload
```

### 2. Mendapatkan Tempat yang Tersedia
```bash
curl http://localhost:5000/data/places
```

### 3. Melatih Model
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"place": "NamaTempat"}' \
  http://localhost:5000/model/train
```

### 4. Mendapatkan Forecast dengan Status
```bash
curl "http://localhost:5000/model/forecast/NamaTempat?hours=24"
```

**Response format:**
```json
{
  "forecast_with_status": [
    {
      "Timestamp": "2024-01-01 13:00:00",
      "Forecast": 3.2,
      "Status": "Sedang"
    }
  ],
  "summary": {
    "forecast_hours": 24,
    "avg_forecast": 2.8,
    "max_forecast": 5.1,
    "min_forecast": 1.2,
    "status_distribution": {
      "Sepi": 8,
      "Sedang": 12,
      "Rame": 4
    }
  }
}
```

### 5. Mendapatkan Klasifikasi Status
```bash
curl "http://localhost:5000/model/status/NamaTempat?hours=48"
```

## Format Data

File CSV harus berisi kolom-kolom berikut:
- `Timestamp`: Tanggal dan waktu record
- `Tempat`: Nama lokasi/tempat
- `Jumlah`: Jumlah sampah
- `Kategori`: Kategori sampah

## Parameter Model

Model Random Forest menggunakan fitur-fitur berikut:
- **Fitur waktu**: jam, hari dalam seminggu, flag weekend
- **Fitur lag**: 1, 24, 48 jam yang lalu
- **Fitur rolling**: rata-rata dan standar deviasi rolling 3 dan 24 jam

## Klasifikasi Status

Tingkat sampah diklasifikasikan sebagai:
- **Sepi**: ≤ 2 unit
- **Sedang**: 2-5 unit  
- **Rame**: > 5 unit

## Struktur Proyek

```
flask_app/
├── main.py                 # Aplikasi Flask utama
├── requirements.txt        # Dependensi Python
├── modules/
│   ├── __init__.py
│   ├── data_processor.py   # Loading dan preprocessing data
│   ├── time_series_analyzer.py  # Analisis time series
│   └── ml_model.py        # Model machine learning
└── uploads/               # Direktori untuk file yang diupload
```

## Penanganan Error

API mencakup penanganan error yang komprehensif:
- Validasi file
- Validasi data
- Validasi pelatihan model
- Kode status HTTP yang tepat
- Pesan error yang detail

## Pengembangan

Untuk menjalankan dalam mode development:
```bash
python main.py
```

Aplikasi akan berjalan dengan mode debug aktif dan auto-reload saat ada perubahan.

## Dependensi

- Flask 2.3.3
- Flask-CORS 4.0.0
- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
- matplotlib 3.7.2
- seaborn 0.12.2
- Werkzeug 2.3.7
- joblib 1.3.2
- gunicorn 23.0.0
