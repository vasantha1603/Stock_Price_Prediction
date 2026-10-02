"""
Test script for Stock Market Prediction System
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules at top-level for reuse in tests
from src.data_collection import StockDataCollector, NewsDataCollector
from src.feature_engineering import FeatureEngineering, TechnicalIndicators
from src.sentiment_analysis import SentimentAnalyzer, FinancialNewsProcessor
from src.model_training import ModelTrainer
from src.visualization import StockVisualizer, DashboardVisualizer
from config.settings import STOCK_SYMBOLS

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Already imported at module level
        _ = (StockDataCollector, NewsDataCollector)
        print("✅ Data collection module imported successfully")
    except Exception as e:
        print(f"❌ Error importing data collection: {e}")
        return False
    
    try:
        _ = (FeatureEngineering, TechnicalIndicators)
        print("✅ Feature engineering module imported successfully")
    except Exception as e:
        print(f"❌ Error importing feature engineering: {e}")
        return False
    
    try:
        _ = (SentimentAnalyzer, FinancialNewsProcessor)
        print("✅ Sentiment analysis module imported successfully")
    except Exception as e:
        print(f"❌ Error importing sentiment analysis: {e}")
        return False
    
    try:
        _ = ModelTrainer
        print("✅ Model training module imported successfully")
    except Exception as e:
        print(f"❌ Error importing model training: {e}")
        return False
    
    try:
        _ = (StockVisualizer, DashboardVisualizer)
        print("✅ Visualization module imported successfully")
    except Exception as e:
        print(f"❌ Error importing visualization: {e}")
        return False
    
    try:
        _ = STOCK_SYMBOLS
        print("✅ Configuration module imported successfully")
    except Exception as e:
        print(f"❌ Error importing configuration: {e}")
        return False
    
    return True


def test_data_collection():
    """Test data collection functionality"""
    print("\nTesting data collection...")
    
    try:
        collector = StockDataCollector()
        data = collector.get_stock_data('AAPL', period='1mo')
        
        if not data.empty:
            print(f"✅ Successfully collected {len(data)} records for AAPL")
            print(f"   Data shape: {data.shape}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            return True
        else:
            print("❌ No data collected")
            return False
            
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\nTesting feature engineering...")
    
    try:
        # Get sample data
        collector = StockDataCollector()
        data = collector.get_stock_data('AAPL', period='1mo')
        
        # Create features
        fe = FeatureEngineering()
        features = fe.prepare_features(data)
        
        if not features.empty:
            print(f"✅ Successfully created {len(features)} feature records")
            print(f"   Feature shape: {features.shape}")
            print(f"   Number of features: {len(fe.get_feature_columns(features))}")
            return True
        else:
            print("❌ No features created")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return False


def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\nTesting sentiment analysis...")
    
    try:
        processor = FinancialNewsProcessor()
        sample_news = processor.create_sample_news_data('AAPL', days=10)
        sentiment_df = processor.process_news_data(sample_news, 'AAPL')
        
        if not sentiment_df.empty:
            print(f"✅ Successfully processed {len(sentiment_df)} sentiment records")
            print(f"   Sentiment shape: {sentiment_df.shape}")
            return True
        else:
            print("❌ No sentiment data created")
            return False
            
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        return False


def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        # Get data
        collector = StockDataCollector()
        data = collector.get_stock_data('AAPL', period='1mo')
        
        # Create features
        fe = FeatureEngineering()
        features = fe.prepare_features(data)
        
        # Create sentiment data
        processor = FinancialNewsProcessor()
        sample_news = processor.create_sample_news_data('AAPL', days=len(features))
        sentiment_df = processor.process_news_data(sample_news, 'AAPL')
        
        # Train models
        trainer = ModelTrainer()
        numerical_results = trainer.train_numerical_models(features)
        hybrid_results = trainer.train_hybrid_model(features, sentiment_df)
        
        print(f"✅ Successfully trained {len(numerical_results)} numerical models")
        print(f"✅ Successfully trained {len(hybrid_results)} hybrid models")
        
        # Show some results
        for name, result in numerical_results.items():
            if 'rmse' in result:
                print(f"   {name}: RMSE = {result['rmse']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False


def test_visualization():
    """Test visualization functionality"""
    print("\nTesting visualization...")
    
    try:
        # Get data
        collector = StockDataCollector()
        data = collector.get_stock_data('AAPL', period='1mo')
        
        # Create features
        fe = FeatureEngineering()
        features = fe.prepare_features(data)
        
        # Create visualizations
        viz = StockVisualizer()
        
        # Test price chart
        _ = viz.create_price_chart(data, 'AAPL')
        print("✅ Price chart created successfully")
        
        # Test technical indicators chart
        _ = viz.create_technical_indicators_chart(features, 'AAPL')
        print("✅ Technical indicators chart created successfully")
        
        # Test correlation heatmap
        numerical_cols = features.select_dtypes(include=[np.number]).columns[:10]
        corr_data = features[numerical_cols]
        _ = viz.create_correlation_heatmap(corr_data, 'AAPL')
        print("✅ Correlation heatmap created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Stock Market Prediction System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_collection,
        test_feature_engineering,
        test_sentiment_analysis,
        test_model_training,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To run the application:")
        print("   streamlit run app/main.py")
        print("\n🐳 To run with Docker:")
        print("   docker build -t stock-prediction .")
        print("   docker run -p 8501:8501 stock-prediction")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main() 