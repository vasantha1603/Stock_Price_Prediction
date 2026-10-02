"""
Data collection module for stock market prediction system
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import requests
import json
from pathlib import Path
import pickle
import time

from config.settings import (
    STOCK_SYMBOLS, DEFAULT_PERIOD, DEFAULT_INTERVAL,
    CACHE_DIR, CACHE_EXPIRY, MAX_CACHE_SIZE
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Collects stock market data from various sources
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        
    def get_stock_data(self, symbol: str, period: str = DEFAULT_PERIOD, 
                      interval: str = DEFAULT_INTERVAL, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock data, preferring local MacroTrends CSV if available, else Yahoo Finance
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if use_cache and self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        local_df = self._try_load_local_macrotrends(symbol)
        if local_df is not None and not local_df.empty:
            data = self._clean_data(local_df)
            if use_cache:
                self._cache_data(cache_key, data)
            logger.info(f"Loaded {symbol} from local MacroTrends CSV: {len(data)} rows")
            return data
        try:
            logger.info(f"Fetching data for {symbol} from Yahoo Finance for period {period}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            data = self._clean_data(data)
            if use_cache:
                self._cache_data(cache_key, data)
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Yahoo Finance")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def _detect_header_start(self, file_path: Path) -> int:
        """
        Detect line number where the CSV header begins (0-based).
        Returns 0 if not found.
        """
        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    low = line.strip().lower()
                    if low.startswith('date,') or low == 'date':
                        return idx
        except Exception:
            pass
        return 0

    def _try_load_local_macrotrends(self, symbol: str) -> Optional[pd.DataFrame]:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        candidates: List[Path] = []
        if symbol.upper() == 'TCS':
            candidates.append(RAW_DATA_DIR / 'TCS_Dataset.csv')
        for path in RAW_DATA_DIR.glob('MacroTrends_Data_Download_*.csv'):
            if symbol.upper() in path.stem.upper():
                candidates.append(path)
        candidates.append(RAW_DATA_DIR / f"{symbol.upper()}.csv")
        for file_path in candidates:
            if file_path.exists():
                try:
                    header_start = self._detect_header_start(file_path)
                    df = pd.read_csv(
                        file_path,
                        skiprows=header_start,
                        engine='python'
                    )
                    df = self._normalize_macrotrends_dataframe(df)
                    return df
                except Exception as e:
                    logger.warning(f"Failed to read local CSV {file_path}: {e}")
                    continue
        return None

    def _normalize_macrotrends_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = 'date'
        # Identify close column preference
        close_col = 'close'
        if 'close' not in df.columns:
            if 'adj close' in df.columns:
                close_col = 'adj close'
            elif 'adjclose' in df.columns:
                close_col = 'adjclose'
        required = [date_col, 'open', 'high', 'low', close_col, 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in MacroTrends CSV: {missing}")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        for c in ['open', 'high', 'low', close_col, 'volume']:
            df[c] = df[c].astype(str).str.replace(',', '', regex=False).str.replace('$', '', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            close_col: 'Close',
            'volume': 'Volume',
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df

    def get_multiple_stocks(self, symbols: List[str], period: str = DEFAULT_PERIOD,
                          interval: str = DEFAULT_INTERVAL) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period, interval)
                results[symbol] = data
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
        return results
    
    def get_all_supported_stocks(self, period: str = DEFAULT_PERIOD,
                               interval: str = DEFAULT_INTERVAL) -> Dict[str, pd.DataFrame]:
        return self.get_multiple_stocks(list(STOCK_SYMBOLS.keys()), period, interval)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.dropna()
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        if data['Volume'].dtype == 'object':
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        data = data.sort_index()
        return data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self.cache_timestamps:
            return False
        timestamp = self.cache_timestamps[cache_key]
        age = time.time() - timestamp
        return age < CACHE_EXPIRY
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        if len(self.cache) >= MAX_CACHE_SIZE:
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = time.time()
    
    def save_data_to_csv(self, data: pd.DataFrame, symbol: str, 
                        output_dir: Path = None) -> str:
        if output_dir is None:
            output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = output_dir / filename
        data.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")
        return str(filepath)
    
    def get_stock_info(self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'name': 'Unknown'}


class NewsDataCollector:
    """
    Collects financial news data for sentiment analysis
    """
    
    def __init__(self):
        self.news_cache = {}
        
    def get_kaggle_news(self) -> Optional[pd.DataFrame]:
        news_path = RAW_DATA_DIR / 'news_stock_price.csv'
        if not news_path.exists():
            return None
        try:
            df = pd.read_csv(news_path)
            cols = [c.strip().lower() for c in df.columns]
            df.columns = cols
            date_col = 'date' if 'date' in cols else 'published' if 'published' in cols else None
            title_col = 'title' if 'title' in cols else None
            content_col = 'content' if 'content' in cols else 'text' if 'text' in cols else None
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                df = df.sort_values(date_col)
            out = pd.DataFrame({
                'date': df[date_col] if date_col in df.columns else pd.NaT,
                'title': df[title_col] if title_col in df.columns else '',
                'content': df[content_col] if content_col in df.columns else ''
            })
            return out
        except Exception as e:
            logger.warning(f"Failed to read Kaggle news CSV: {e}")
            return None
    
    def get_financial_news(self, symbol: str, days_back: int = 30) -> List[Dict]:
        kaggle_df = self.get_kaggle_news()
        if kaggle_df is not None and not kaggle_df.empty:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
            df = kaggle_df[kaggle_df['date'] >= cutoff]
            records = []
            for _, row in df.iterrows():
                records.append({
                    'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else datetime.now().strftime('%Y-%m-%d'),
                    'title': str(row.get('title', '')),
                    'content': str(row.get('content', '')),
                    'sentiment': ''
                })
            if records:
                return records
        return [
            {
                'title': f'Positive outlook for {symbol}',
                'content': f'Analysts are bullish on {symbol} stock performance',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'sentiment': 'positive'
            },
            {
                'title': f'{symbol} quarterly results',
                'content': f'{symbol} reported strong quarterly earnings',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'sentiment': 'positive'
            }
        ]
    
    def get_news_sentiment_data(self, symbol: str, start_date: str, 
                               end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_data = []
        for date in dates:
            sentiment_data.append({
                'date': date,
                'symbol': symbol,
                'sentiment_score': np.random.uniform(-1, 1),
                'news_count': np.random.randint(1, 10),
                'positive_count': np.random.randint(0, 5),
                'negative_count': np.random.randint(0, 3),
                'neutral_count': np.random.randint(0, 2)
            })
        return pd.DataFrame(sentiment_data)


def main():
    collector = StockDataCollector()
    try:
        data = collector.get_stock_data('AAPL', period='1mo')
        print(f"Successfully collected {len(data)} records for AAPL")
        print(data.head())
    except Exception as e:
        print(f"Error in data collection: {str(e)}")


if __name__ == "__main__":
    main() 