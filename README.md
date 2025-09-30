# Macroeconomic Data Science Project

This project analyzes macroeconomic indicators (GDP Growth, Inflation, Unemployment) and provides predictive modeling capabilities.

## Project Structure

```
macroeconomic_data_science_project/
├── data/
│   ├── raw/              # Raw CSV files (GDP.csv, Inflation.csv, Unemployment.csv)
│   ├── interim/          # Cleaned data
│   └── processed/        # Merged and processed data
├── notebooks/
│   └── eda_analysis.ipynb  # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   └── data_pipeline.py  # Data processing pipeline
│   ├── models/
│   │   └── train_model.py    # Model training script
│   └── api/
│       └── app.py            # Flask API
├── models/               # Trained model files
├── logs/                 # Log files
├── config/               # Configuration files
├── docker/               # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd macroeconomic_data_science_project
pip install -r requirements.txt
```

### 2. Place Raw Data

Place your CSV files in the `data/raw/` folder:
- GDP.csv
- Inflation.csv
- Unemployment.csv

### 3. Run Data Pipeline

```bash
cd src/data
python data_pipeline.py
```

This will:
- Clean raw data and save to `data/interim/`
- Merge and transform data to `data/processed/macroeconomic.csv`

### 4. Exploratory Data Analysis

```bash
cd notebooks
jupyter notebook eda_analysis.ipynb
```

### 5. Train Model

```bash
cd src/models
python train_model.py
```

This will train a Random Forest model and save it to `models/`

### 6. Run API

```bash
cd src/api
python app.py
```

API will be available at `http://localhost:5000`

### 7. Docker Deployment

```bash
cd docker
docker-compose up -d
```

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "gdp_growth_lag1": 3.5,
    "inflation_lag1": 2.1,
    "unemployment_lag1": 5.2,
    "gdp_growth_ma3": 3.2,
    "inflation_ma3": 2.3,
    "year": 2024
  }'
```

## Features

- **Data Pipeline**: Automated data cleaning and transformation
- **EDA**: Comprehensive exploratory data analysis with visualizations
- **Predictive Modeling**: Random Forest model for forecasting
- **REST API**: Flask-based API for model predictions
- **Containerization**: Docker support for easy deployment
- **Logging**: Comprehensive logging for debugging and monitoring

## Model Features

The model uses the following features for prediction:
- Previous year's GDP Growth (lag1)
- Previous year's Inflation (lag1)
- Previous year's Unemployment (lag1)
- 3-year moving average of GDP Growth
- 3-year moving average of Inflation
- Year

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- flask
- matplotlib
- seaborn
- plotly
- jupyter

## License

MIT License