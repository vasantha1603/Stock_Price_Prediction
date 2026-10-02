# 🚀 Quick Start Guide

## Stock Market Prediction System

This guide will help you get the Stock Market Prediction System up and running quickly.

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🛠️ Installation

### Option 1: Local Installation

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

### Option 2: Docker Installation

1. **Build Docker image**
   ```bash
   docker build -t stock-prediction .
   ```

2. **Run container**
   ```bash
   docker run -p 8501:8501 stock-prediction
   ```

## 🚀 Running the Application

### Local Development

1. **Start the Streamlit app**
   ```bash
   streamlit run app/main.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The application will open automatically

### Docker Deployment

1. **Run with Docker**
   ```bash
   docker run -p 8501:8501 stock-prediction
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## 🧪 Testing the System

Run the test script to verify everything is working:

```bash
python test_system.py
```

This will test:
- ✅ Module imports
- ✅ Data collection
- ✅ Feature engineering
- ✅ Sentiment analysis
- ✅ Model training
- ✅ Visualization

## 📊 Using the Application

### 1. Welcome Page
- Overview of system capabilities
- Performance metrics
- Supported companies

### 2. Configuration (Sidebar)
- **Stock Selection**: Choose from 8 supported companies
- **Time Period**: Select data range (1 month to 5 years)
- **Analysis Type**: Choose analysis mode

### 3. Analysis Types

#### Technical Analysis
- 📈 Price charts with volume
- 🔧 Technical indicators (RSI, MACD, Moving Averages)
- 📊 Correlation matrix
- 📈 Returns distribution

#### Sentiment Analysis
- 📰 News sentiment processing
- 📊 Sentiment score visualization
- 📰 Recent news headlines
- 📊 Sentiment summary statistics

#### Hybrid Analysis
- 🔗 Combined numerical and textual analysis
- 📊 Comprehensive dashboard
- 💡 Hybrid insights
- 🔗 Sentiment-price correlation

#### Model Training
- 🤖 Train multiple ML models
- 📊 Performance comparison
- 🏆 Best model identification
- 📈 Model metrics visualization

## 🎯 Key Features

### Supported Companies
- **AAPL** (Apple Inc.)
- **GOOG** (Alphabet/Google)
- **AMZN** (Amazon)
- **META** (Meta Platforms)
- **MSFT** (Microsoft)
- **NFLX** (Netflix)
- **NVDA** (Nvidia)
- **TCS** (Tata Consultancy Services)

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Moving Averages (SMA, EMA, WMA)
- Bollinger Bands
- Stochastic Oscillator
- Volume indicators

### Machine Learning Models
- Random Forest (Regression & Classification)
- Multi-Layer Perceptron (MLP)
- Logistic Regression
- Naive Bayes
- Hybrid Fusion Models

### Visualization Features
- Interactive Plotly charts
- Real-time data updates
- Technical indicator overlays
- Model performance comparison
- Sentiment analysis charts

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# Database Configuration (Optional)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token_here
INFLUXDB_ORG=your_org_here
INFLUXDB_BUCKET=stock_data

# API Keys (Optional)
YAHOO_FINANCE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

### Customization
- Modify `config/settings.py` for custom parameters
- Adjust technical indicator periods
- Change model hyperparameters
- Customize visualization themes

## 📈 Data Sources

### Historical Stock Data
- **Yahoo Finance**: Real-time and historical data
- **Macrotrends**: Long-term historical data

### News Sentiment Data
- **Kaggle Dataset**: News & Stock Price correlation
- **Financial News APIs**: Real-time financial news

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Data Collection Issues**
   - Check internet connection
   - Verify stock symbol is valid
   - Try different time periods

3. **Model Training Errors**
   - Ensure sufficient data (at least 30 days)
   - Check feature engineering output
   - Verify data quality

4. **Visualization Issues**
   - Update Plotly: `pip install --upgrade plotly`
   - Clear browser cache
   - Check JavaScript console

### Performance Tips

1. **Faster Loading**
   - Use shorter time periods for initial testing
   - Cache frequently used data
   - Optimize feature engineering

2. **Memory Management**
   - Limit feature columns for large datasets
   - Use data sampling for testing
   - Monitor system resources

## 📚 Next Steps

1. **Explore the Code**
   - Review `src/` modules for implementation details
   - Check `config/settings.py` for configuration options
   - Examine `app/main.py` for UI customization

2. **Extend Functionality**
   - Add new technical indicators
   - Implement additional ML models
   - Integrate real-time news APIs
   - Add portfolio optimization

3. **Deploy to Production**
   - Set up InfluxDB for time-series storage
   - Configure Grafana dashboards
   - Implement real-time data streaming
   - Add authentication and user management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- Check the [README.md](README.md) for detailed documentation
- Review the test results for system status
- Open an issue for bugs or feature requests

---

**Happy Trading! 📈** 