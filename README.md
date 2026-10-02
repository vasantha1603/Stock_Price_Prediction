# Advanced Stock Market Prediction System

A comprehensive hybrid stock market prediction system that combines numerical time-series analysis and textual sentiment analysis for improved accuracy in stock price prediction.

## 🚀 Features

- **Hybrid Model**: Combines numerical and textual analysis
- **Technical Indicators**: RSI, Moving Averages, MACD, Bollinger Bands
- **Machine Learning Models**: Naive Bayes, MLP, Logistic Regression, Random Forest
- **Real-time Visualization**: Streamlit dashboard with interactive charts
- **Data Storage**: InfluxDB for time-series data management
- **Sentiment Analysis**: NLP-based news sentiment analysis

## 📊 Supported Companies

- **AAPL** (Apple Inc.)
- **GOOG** (Alphabet/Google)
- **AMZN** (Amazon)
- **META** (Meta Platforms)
- **MSFT** (Microsoft)
- **NFLX** (Netflix)
- **NVDA** (Nvidia)
- **TCS** (Tata Consultancy Services)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock-market-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

1. **Run the Streamlit app**
```bash
streamlit run app/main.py
```

2. **Access the application**
- Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
stock-market-prediction/
├── app/
│   ├── main.py                 # Main Streamlit application
│   ├── components/             # UI components
│   └── pages/                  # Streamlit pages
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed datasets
│   └── news/                   # News sentiment data
├── models/
│   ├── numerical/              # Numerical ML models
│   ├── textual/                # Text sentiment models
│   └── hybrid/                 # Hybrid fusion models
├── src/
│   ├── data_collection.py      # Data collection utilities
│   ├── feature_engineering.py  # Technical indicators
│   ├── sentiment_analysis.py   # NLP processing
│   ├── model_training.py       # Model training pipeline
│   └── visualization.py        # Plotting utilities
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebooks
├── config/
│   └── settings.py             # Configuration settings
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 📈 Data Sources

### Historical Stock Data
- **Yahoo Finance**: Real-time and historical stock data
- **Macrotrends**: Long-term historical data for analysis

### News Sentiment Data
- **Kaggle Dataset**: News & Stock Price correlation data
- **Financial News APIs**: Real-time financial news

## 🧠 Model Architecture

### Numerical Analysis
- **Technical Indicators**: RSI, MA, MACD, Bollinger Bands
- **Feature Engineering**: Lag features, volatility, returns
- **Models**: Random Forest, MLP, Logistic Regression

### Textual Analysis
- **NLP Pipeline**: NLTK, spaCy, TextBlob
- **Sentiment Extraction**: Polarity and subjectivity scores
- **Model**: Naive Bayes for sentiment classification

### Hybrid Fusion
- **Late Fusion**: Decision-level combination
- **Ensemble Methods**: Weighted averaging of predictions
- **Performance**: 7-12% improvement over individual models

## 📊 Performance Metrics

- **RMSE**: Root Mean Square Error for price prediction
- **MAE**: Mean Absolute Error for accuracy assessment
- **Accuracy**: Binary trend direction prediction
- **F1-Score**: Balanced precision and recall

## 🎯 Key Results

- **Hybrid Model**: Outperforms individual models by 7-12%
- **Random Forest**: Best numerical model (RMSE: 2.13, MAE: 1.72)
- **MLP**: Strong learning capabilities (94.6% training, 91.3% validation)
- **Sentiment Integration**: Improves prediction during market volatility

## 🔧 Configuration

Create a `.env` file in the root directory:

```env
# Database Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token_here
INFLUXDB_ORG=your_org_here
INFLUXDB_BUCKET=stock_data

# API Keys (optional)
YAHOO_FINANCE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Model Settings
MODEL_SAVE_PATH=./models/
DATA_CACHE_PATH=./data/cache/
```

## 🐳 Docker Deployment

1. **Build the Docker image**
```bash
docker build -t stock-prediction .
```

2. **Run the container**
```bash
docker run -p 8501:8501 stock-prediction
```

## 📝 Usage Examples

### Basic Stock Analysis
```python
from src.data_collection import StockDataCollector
from src.feature_engineering import TechnicalIndicators

# Collect data
collector = StockDataCollector()
data = collector.get_stock_data('AAPL', period='1y')

# Calculate indicators
indicators = TechnicalIndicators()
features = indicators.calculate_all(data)
```

### Model Training
```python
from src.model_training import HybridModel

# Train hybrid model
model = HybridModel()
model.train(numerical_data, textual_data)
predictions = model.predict(new_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research paper: "Advanced Stock Price Forecasting Using a Hybrid Model of Numerical and Textual Analysis"
- Reference implementation: [Stock-Market-Prediction](https://github.com/madhurimarawat/Stock-Market-Prediction)
- Data sources: Yahoo Finance, Kaggle, Macrotrends

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a research project for educational purposes. Always do your own research before making investment decisions. 