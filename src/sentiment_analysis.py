"""
Sentiment analysis module for stock market prediction system
"""

import pandas as pd
import numpy as np
import nltk
import spacy
from textblob import TextBlob
from typing import Dict, List, Optional, Tuple
import logging
import re
from datetime import datetime, timedelta
import json

from config.settings import SENTIMENT_KEYWORDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None


class SentimentAnalyzer:
    """
    Analyze sentiment from financial news and text data
    """
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.bullish_keywords = SENTIMENT_KEYWORDS['bullish']
        self.bearish_keywords = SENTIMENT_KEYWORDS['bearish']
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        if not nlp:
            return []
        
        try:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE']]
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def calculate_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"Error calculating TextBlob sentiment: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def calculate_keyword_sentiment(self, text: str) -> Dict[str, int]:
        """
        Calculate sentiment based on keyword frequency
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with bullish and bearish keyword counts
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        return {
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count,
            'sentiment_score': bullish_count - bearish_count
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Comprehensive sentiment analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all sentiment metrics
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'bullish_keywords': 0,
                'bearish_keywords': 0,
                'sentiment_score': 0,
                'entities': [],
                'sentiment_label': 'neutral'
            }
        
        # Calculate different sentiment metrics
        textblob_sentiment = self.calculate_textblob_sentiment(processed_text)
        keyword_sentiment = self.calculate_keyword_sentiment(processed_text)
        entities = self.extract_entities(text)
        
        # Combine sentiment scores
        combined_score = (textblob_sentiment['polarity'] + keyword_sentiment['sentiment_score'] * 0.1) / 2
        
        # Determine sentiment label
        if combined_score > 0.1:
            sentiment_label = 'positive'
        elif combined_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'polarity': textblob_sentiment['polarity'],
            'subjectivity': textblob_sentiment['subjectivity'],
            'bullish_keywords': keyword_sentiment['bullish_keywords'],
            'bearish_keywords': keyword_sentiment['bearish_keywords'],
            'sentiment_score': keyword_sentiment['sentiment_score'],
            'combined_score': combined_score,
            'entities': entities,
            'sentiment_label': sentiment_label
        }
    
    def analyze_news_batch(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of news articles
        
        Args:
            news_data: List of news articles with 'title' and 'content' fields
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for article in news_data:
            # Combine title and content
            full_text = f"{article.get('title', '')} {article.get('content', '')}"
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(full_text)
            
            # Add article metadata
            result = {
                'date': article.get('date', datetime.now().strftime('%Y-%m-%d')),
                'title': article.get('title', ''),
                'content': article.get('content', ''),
                **sentiment
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def create_sentiment_features(self, news_df: pd.DataFrame, 
                                date_column: str = 'date') -> pd.DataFrame:
        """
        Create sentiment features for machine learning
        
        Args:
            news_df: DataFrame with sentiment analysis results
            date_column: Name of the date column
            
        Returns:
            DataFrame with aggregated sentiment features
        """
        # Convert date column to datetime
        news_df[date_column] = pd.to_datetime(news_df[date_column])
        
        # Set date as index
        news_df = news_df.set_index(date_column)
        
        # Aggregate by date
        daily_sentiment = news_df.groupby(news_df.index).agg({
            'polarity': ['mean', 'std', 'count'],
            'subjectivity': ['mean', 'std'],
            'combined_score': ['mean', 'std'],
            'bullish_keywords': 'sum',
            'bearish_keywords': 'sum',
            'sentiment_score': 'sum'
        }).fillna(0)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
        
        # Add sentiment ratio features
        daily_sentiment['sentiment_ratio'] = (
            daily_sentiment['bullish_keywords_sum'] / 
            (daily_sentiment['bullish_keywords_sum'] + daily_sentiment['bearish_keywords_sum'] + 1)
        )
        
        # Add rolling averages
        for window in [3, 7, 14]:
            daily_sentiment[f'polarity_mean_{window}d_ma'] = daily_sentiment['polarity_mean'].rolling(window=window).mean()
            daily_sentiment[f'combined_score_mean_{window}d_ma'] = daily_sentiment['combined_score_mean'].rolling(window=window).mean()
        
        return daily_sentiment


class FinancialNewsProcessor:
    """
    Process financial news data for sentiment analysis
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def process_news_data(self, news_data: List[Dict], symbol: str) -> pd.DataFrame:
        """
        Process news data for a specific stock symbol
        
        Args:
            news_data: List of news articles
            symbol: Stock symbol
            
        Returns:
            DataFrame with processed sentiment data
        """
        # Add symbol to each article
        for article in news_data:
            article['symbol'] = symbol
        
        # Analyze sentiment
        sentiment_df = self.sentiment_analyzer.analyze_news_batch(news_data)
        
        # Create features
        features_df = self.sentiment_analyzer.create_sentiment_features(sentiment_df)
        
        # Add symbol column
        features_df['symbol'] = symbol
        
        return features_df
    
    def create_sample_news_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Create sample news data for testing
        
        Args:
            symbol: Stock symbol
            days: Number of days to generate
            
        Returns:
            List of sample news articles
        """
        sample_news = []
        
        # Sample headlines and content
        headlines = [
            f"{symbol} reports strong quarterly earnings",
            f"Analysts upgrade {symbol} stock rating",
            f"{symbol} announces new product launch",
            f"Market volatility affects {symbol} performance",
            f"{symbol} faces regulatory challenges",
            f"Positive outlook for {symbol} growth",
            f"{symbol} expands into new markets",
            f"Competition heats up for {symbol}",
            f"{symbol} beats earnings expectations",
            f"Economic factors impact {symbol} stock"
        ]
        
        content_templates = [
            "The company reported impressive quarterly results with revenue growth of {growth}%.",
            "Analysts are optimistic about the company's future prospects.",
            "The stock has shown strong performance in recent trading sessions.",
            "Market conditions have created challenges for the company.",
            "Investors are closely watching the company's strategic moves.",
            "The company's innovative approach is driving growth.",
            "Regulatory changes may impact the company's operations.",
            "The competitive landscape is becoming more challenging.",
            "The company's financial health remains strong.",
            "Economic factors are influencing the stock's performance."
        ]
        
        # Generate sample data
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # Randomly select headline and content
            headline = np.random.choice(headlines)
            content = np.random.choice(content_templates).format(growth=np.random.randint(5, 25))
            
            # Add some randomness to sentiment
            if np.random.random() > 0.5:
                headline = headline.replace("strong", "weak").replace("positive", "negative")
                content = content.replace("impressive", "disappointing").replace("optimistic", "concerned")
            
            sample_news.append({
                'date': date.strftime('%Y-%m-%d'),
                'title': headline,
                'content': content,
                'symbol': symbol
            })
        
        return sample_news
    
    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment data
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if sentiment_df.empty:
            return {}
        
        summary = {
            'total_articles': len(sentiment_df),
            'avg_polarity': sentiment_df['polarity'].mean(),
            'avg_subjectivity': sentiment_df['subjectivity'].mean(),
            'avg_combined_score': sentiment_df['combined_score'].mean(),
            'positive_articles': len(sentiment_df[sentiment_df['sentiment_label'] == 'positive']),
            'negative_articles': len(sentiment_df[sentiment_df['sentiment_label'] == 'negative']),
            'neutral_articles': len(sentiment_df[sentiment_df['sentiment_label'] == 'neutral']),
            'total_bullish_keywords': sentiment_df['bullish_keywords'].sum(),
            'total_bearish_keywords': sentiment_df['bearish_keywords'].sum()
        }
        
        return summary


def main():
    """
    Test the sentiment analysis functionality
    """
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test text
    test_text = "Apple reports strong quarterly earnings with impressive revenue growth. Analysts are bullish on the stock."
    
    sentiment = analyzer.analyze_sentiment(test_text)
    print("Sentiment Analysis Results:")
    print(json.dumps(sentiment, indent=2))
    
    # Test news processor
    processor = FinancialNewsProcessor()
    
    # Create sample news data
    sample_news = processor.create_sample_news_data('AAPL', days=10)
    
    # Process news data
    sentiment_df = processor.process_news_data(sample_news, 'AAPL')
    
    print(f"\nProcessed {len(sentiment_df)} sentiment records")
    print(sentiment_df.head())
    
    # Get summary
    summary = processor.get_sentiment_summary(sentiment_df)
    print(f"\nSentiment Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main() 