"""
Model training module for stock market prediction system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from config.settings import (
    RANDOM_STATE, TEST_SIZE, NUMERICAL_MODELS_PATH, 
    TEXTUAL_MODELS_PATH, HYBRID_MODELS_PATH
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train machine learning models for stock prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Future_Return_1d') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Get feature columns (exclude targets and basic OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        target_patterns = ['Future_', 'Price_Up_']
        
        feature_cols = []
        for col in data.columns:
            if col not in exclude_cols and not any(pattern in col for pattern in target_patterns):
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        
        # Prepare features and target
        X = data[feature_cols].fillna(0)
        y = data[target_column].fillna(0)
        
        # Remove rows where target is NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def train_numerical_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train numerical models"""
        logger.info("Training numerical models...")
        
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        results = {}
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['random_forest'] = {
            'model': rf_model,
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred)
        }
        
        # MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['mlp'] = scaler
        
        mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=RANDOM_STATE)
        mlp_model.fit(X_train_scaled, y_train)
        mlp_pred = mlp_model.predict(X_test_scaled)
        results['mlp'] = {
            'model': mlp_model,
            'rmse': np.sqrt(mean_squared_error(y_test, mlp_pred)),
            'mae': mean_absolute_error(y_test, mlp_pred)
        }
        
        # Logistic Regression (classification)
        y_binary = (y > 0).astype(int)
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        lr_model = LogisticRegression(random_state=RANDOM_STATE)
        lr_model.fit(X_train_bin, y_train_bin)
        lr_pred = lr_model.predict(X_test_bin)
        results['logistic_regression'] = {
            'model': lr_model,
            'accuracy': accuracy_score(y_test_bin, lr_pred)
        }
        
        # Naive Bayes (classification)
        nb_model = GaussianNB()
        nb_model.fit(X_train_bin, y_train_bin)
        nb_pred = nb_model.predict(X_test_bin)
        results['naive_bayes'] = {
            'model': nb_model,
            'accuracy': accuracy_score(y_test_bin, nb_pred)
        }
        
        self.models = {name: result['model'] for name, result in results.items()}
        return results
    
    def train_hybrid_model(self, numerical_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """Train hybrid model combining numerical and textual features"""
        logger.info("Training hybrid model...")
        
        # Combine features
        combined_data = numerical_data.merge(sentiment_data, left_index=True, right_index=True, how='inner')
        
        X, y = self.prepare_data(combined_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['hybrid'] = scaler
        
        # Train hybrid Random Forest
        rf_hybrid = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE)
        rf_hybrid.fit(X_train_scaled, y_train)
        rf_pred = rf_hybrid.predict(X_test_scaled)
        
        results = {
            'hybrid_random_forest': {
                'model': rf_hybrid,
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'mae': mean_absolute_error(y_test, rf_pred)
            }
        }
        
        self.models.update({name: result['model'] for name, result in results.items()})
        return results
    
    def save_models(self, symbol: str):
        """Save trained models"""
        NUMERICAL_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        HYBRID_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if 'hybrid' in name:
                model_path = HYBRID_MODELS_PATH / f"{symbol}_{name}.joblib"
            else:
                model_path = NUMERICAL_MODELS_PATH / f"{symbol}_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            if 'hybrid' in name:
                scaler_path = HYBRID_MODELS_PATH / f"{symbol}_{name}_scaler.joblib"
            else:
                scaler_path = NUMERICAL_MODELS_PATH / f"{symbol}_{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} scaler to {scaler_path}")


def main():
    """Test model training"""
    from src.data_collection import StockDataCollector
    from src.feature_engineering import FeatureEngineering
    from src.sentiment_analysis import FinancialNewsProcessor
    
    # Get data
    collector = StockDataCollector()
    data = collector.get_stock_data('AAPL', period='6mo')
    
    # Create features
    fe = FeatureEngineering()
    features = fe.prepare_features(data)
    
    # Create sample sentiment data
    processor = FinancialNewsProcessor()
    sample_news = processor.create_sample_news_data('AAPL', days=len(features))
    sentiment_df = processor.process_news_data(sample_news, 'AAPL')
    
    # Train models
    trainer = ModelTrainer()
    numerical_results = trainer.train_numerical_models(features)
    hybrid_results = trainer.train_hybrid_model(features, sentiment_df)
    
    print("Model Results:")
    for name, result in numerical_results.items():
        if 'rmse' in result:
            print(f"{name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")
        else:
            print(f"{name}: Accuracy={result['accuracy']:.4f}")
    
    for name, result in hybrid_results.items():
        print(f"{name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")


if __name__ == "__main__":
    main() 