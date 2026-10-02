"""
Configuration settings for the Stock Market Prediction System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Stock symbols and companies
STOCK_SYMBOLS = {
    'AAPL': 'Apple Inc.',
    'GOOG': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'MSFT': 'Microsoft Corporation',
    'NFLX': 'Netflix Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TCS': 'Tata Consultancy Services'
}

# Data collection settings
DEFAULT_PERIOD = '1y'  # Default data collection period
DEFAULT_INTERVAL = '1d'  # Default data interval

# Technical indicators settings
RSI_PERIOD = 14
MA_PERIODS = [5, 10, 20, 50, 200]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature engineering
LAG_FEATURES = [1, 3, 7]  # Days to look back
ROLLING_WINDOWS = [5, 10, 20]  # Rolling window periods

# Sentiment analysis
SENTIMENT_KEYWORDS = {
    'bullish': ['bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'up', 'strong'],
    'bearish': ['bearish', 'negative', 'decline', 'loss', 'fall', 'down', 'weak', 'drop']
}

# Visualization settings
PLOTLY_TEMPLATE = 'plotly_white'
CHART_HEIGHT = 600
CHART_WIDTH = 800

# Database settings (InfluxDB)
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', '')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'stock_data')

# API settings
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Model save paths
NUMERICAL_MODELS_PATH = MODELS_DIR / "numerical"
TEXTUAL_MODELS_PATH = MODELS_DIR / "textual"
HYBRID_MODELS_PATH = MODELS_DIR / "hybrid"

# Create model directories
NUMERICAL_MODELS_PATH.mkdir(exist_ok=True)
TEXTUAL_MODELS_PATH.mkdir(exist_ok=True)
HYBRID_MODELS_PATH.mkdir(exist_ok=True)

# Performance metrics thresholds
GOOD_ACCURACY_THRESHOLD = 0.8
GOOD_RMSE_THRESHOLD = 5.0
GOOD_MAE_THRESHOLD = 3.0

# Cache settings
CACHE_EXPIRY = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000  # Maximum number of cached items

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Streamlit settings
STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730",
    "font": "sans serif"
} 