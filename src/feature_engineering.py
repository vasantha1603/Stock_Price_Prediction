"""
Feature engineering module for stock market prediction system
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from config.settings import (
    RSI_PERIOD, MA_PERIODS, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, LAG_FEATURES, ROLLING_WINDOWS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=period)
            return rsi.rsi()
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_moving_averages(self, data: pd.DataFrame, periods: List[int] = None) -> Dict[str, pd.Series]:
        if periods is None:
            periods = MA_PERIODS
        
        ma_dict = {}
        
        try:
            max_len = len(data)
            for period in periods:
                if period > max_len:
                    # Skip periods longer than dataset length to avoid all-NaN columns
                    continue
                ma_dict[f'SMA_{period}'] = ta.trend.SMAIndicator(data['Close'], window=period).sma_indicator()
                ma_dict[f'EMA_{period}'] = ta.trend.EMAIndicator(data['Close'], window=period).ema_indicator()
                # Weighted MA may not exist on very short series; guard with try
                try:
                    ma_dict[f'WMA_{period}'] = ta.trend.WMAIndicator(data['Close'], window=period).wma()
                except Exception:
                    pass
            return ma_dict
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {}
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = MACD_FAST, 
                      slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Dict[str, pd.Series]:
        try:
            macd = ta.trend.MACD(data['Close'], window_fast=fast, 
                                window_slow=slow, window_sign=signal)
            return {
                'MACD': macd.macd(),
                'MACD_Signal': macd.macd_signal(),
                'MACD_Histogram': macd.macd_diff()
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {}
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = BOLLINGER_PERIOD,
                                 std: int = BOLLINGER_STD) -> Dict[str, pd.Series]:
        try:
            if period > len(data):
                return {}
            bb = ta.volatility.BollingerBands(data['Close'], window=period, window_dev=std)
            return {
                'BB_Upper': bb.bollinger_hband(),
                'BB_Middle': bb.bollinger_mavg(),
                'BB_Lower': bb.bollinger_lband(),
                'BB_Width': bb.bollinger_wband(),
                'BB_Position': bb.bollinger_pband()
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {}
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        try:
            if k_period > len(data):
                return {}
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], 
                                                   data['Close'], window=k_period, smooth_window=d_period)
            return {
                'Stoch_K': stoch.stoch(),
                'Stoch_D': stoch.stoch_signal()
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return {}
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        try:
            indicators: Dict[str, pd.Series] = {}
            # Volume Rate of Change (manual)
            indicators['Volume_ROC'] = data['Volume'].pct_change(periods=25)
            # VWAP
            try:
                vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], 
                                                           close=data['Close'], volume=data['Volume'])
                indicators['VWAP'] = vwap.volume_weighted_average_price()
            except Exception:
                pass
            # OBV
            try:
                obv = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume'])
                indicators['OBV'] = obv.on_balance_volume()
            except Exception:
                pass
            # Money Flow Index as an alternative volume-price indicator
            try:
                mfi = ta.volume.MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14)
                indicators['MFI'] = mfi.money_flow_index()
            except Exception:
                pass
            return indicators
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        indicators = {}
        indicators['RSI'] = self.calculate_rsi(data)
        ma_dict = self.calculate_moving_averages(data)
        indicators.update(ma_dict)
        macd_dict = self.calculate_macd(data)
        indicators.update(macd_dict)
        bb_dict = self.calculate_bollinger_bands(data)
        indicators.update(bb_dict)
        stoch_dict = self.calculate_stochastic(data)
        indicators.update(stoch_dict)
        volume_dict = self.calculate_volume_indicators(data)
        indicators.update(volume_dict)
        for name, series in indicators.items():
            if series is not None and not series.empty:
                result[name] = series
        return result


class FeatureEngineering:
    """
    Feature engineering for machine learning models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result['Price_Change'] = data['Close'].pct_change()
        result['Price_Change_Abs'] = abs(data['Close'].pct_change())
        result['Daily_Return'] = data['Close'].pct_change()
        result['Cumulative_Return'] = (1 + result['Daily_Return']).cumprod() - 1
        result['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
        result['HL_Spread_Pct'] = ((data['High'] - data['Low']) / data['Close']) * 100
        result['OC_Spread'] = (data['Close'] - data['Open']) / data['Open']
        result['OC_Spread_Pct'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        result['Volume_Change'] = data['Volume'].pct_change()
        result['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        result['Volume_Ratio'] = data['Volume'] / result['Volume_MA']
        return result
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        if lags is None:
            lags = LAG_FEATURES
        result = data.copy()
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                for lag in lags:
                    result[f'{col}_Lag_{lag}'] = data[col].shift(lag)
        if 'Volume' in data.columns:
            for lag in lags:
                result[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        return result
    
    def create_rolling_features(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        if windows is None:
            windows = ROLLING_WINDOWS
        result = data.copy()
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                for window in windows:
                    if window > len(data):
                        continue
                    result[f'{col}_Rolling_Mean_{window}'] = data[col].rolling(window=window).mean()
                    result[f'{col}_Rolling_Std_{window}'] = data[col].rolling(window=window).std()
                    result[f'{col}_Rolling_Min_{window}'] = data[col].rolling(window=window).min()
                    result[f'{col}_Rolling_Max_{window}'] = data[col].rolling(window=window).max()
        if 'Volume' in data.columns:
            for window in windows:
                if window > len(data):
                    continue
                result[f'Volume_Rolling_Mean_{window}'] = data['Volume'].rolling(window=window).mean()
                result[f'Volume_Rolling_Std_{window}'] = data['Volume'].rolling(window=window).std()
        return result
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        returns = data['Close'].pct_change()
        for window in [5, 10, 20, 30]:
            if window > len(data):
                continue
            result[f'Volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        if 'High' in data.columns and 'Low' in data.columns:
            for window in [5, 10, 20]:
                if window > len(data):
                    continue
                hl_ratio = np.log(data['High'] / data['Low'])
                result[f'Parkinson_Vol_{window}d'] = hl_ratio.rolling(window=window).std() * np.sqrt(252)
        return result
    
    def create_target_variables(self, data: pd.DataFrame, forward_periods: List[int] = [1, 3, 7]) -> pd.DataFrame:
        result = data.copy()
        for period in forward_periods:
            # Compute forward returns explicitly to avoid pct_change fill behavior
            future_price = data['Close'].shift(-period)
            result[f'Future_Return_{period}d'] = (future_price / data['Close']) - 1.0
            result[f'Future_Price_{period}d'] = future_price
        for period in forward_periods:
            future_price = data['Close'].shift(-period)
            future_return = (future_price / data['Close']) - 1.0
            result[f'Price_Up_{period}d'] = (future_return > 0).astype(int)
        return result
    
    def prepare_features(self, data: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
        logger.info("Starting feature engineering...")
        data_with_indicators = TechnicalIndicators().calculate_all_indicators(data)
        data_with_price_features = self.create_price_features(data_with_indicators)
        data_with_lags = self.create_lag_features(data_with_price_features)
        data_with_rolling = self.create_rolling_features(data_with_lags)
        data_with_volatility = self.create_volatility_features(data_with_rolling)
        if include_targets:
            final_data = self.create_target_variables(data_with_volatility)
        else:
            final_data = data_with_volatility
        # Drop rows that are all NaN across engineered columns but keep sufficient rows
        final_data = final_data.dropna()
        logger.info(f"Feature engineering completed. Final shape: {final_data.shape}")
        return final_data
    
    def get_feature_columns(self, data: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
        feature_columns = []
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if exclude_targets:
            target_patterns = ['Future_', 'Price_Up_']
            for col in data.columns:
                if any(pattern in col for pattern in target_patterns):
                    exclude_cols.append(col)
        for col in data.columns:
            if col not in exclude_cols:
                feature_columns.append(col)
        return feature_columns
    
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str], 
                      fit_scaler: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(data[feature_columns])
        else:
            scaled_features = self.scaler.transform(data[feature_columns])
        scaled_data = data.copy()
        scaled_data[feature_columns] = scaled_features
        return scaled_data, self.scaler


def main():
    from src.data_collection import StockDataCollector
    collector = StockDataCollector()
    data = collector.get_stock_data('AAPL', period='6mo')
    fe = FeatureEngineering()
    features = fe.prepare_features(data)
    print(f"Original data shape: {data.shape}")
    print(f"Features data shape: {features.shape}")
    print(f"Number of features: {len(fe.get_feature_columns(features))}")
    feature_cols = fe.get_feature_columns(features)
    print(f"Sample features: {feature_cols[:10]}")


if __name__ == "__main__":
    main() 