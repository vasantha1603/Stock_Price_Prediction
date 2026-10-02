"""
Visualization module for stock market prediction system
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import PLOTLY_TEMPLATE, CHART_HEIGHT, CHART_WIDTH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockVisualizer:
    """
    Create visualizations for stock market data
    """
    
    def __init__(self):
        self.template = PLOTLY_TEMPLATE
        self.height = CHART_HEIGHT
        self.width = CHART_WIDTH
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create candlestick chart with volume
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Stock Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Stock Price and Volume',
            yaxis_title='Price',
            yaxis2_title='Volume',
            template=self.template,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def create_technical_indicators_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create chart with technical indicators
        
        Args:
            data: DataFrame with technical indicators
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price with Moving Averages', 'RSI', 'MACD'),
            row_width=[0.5, 0.25, 0.25]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add moving averages if available
        ma_cols = [col for col in data.columns if 'SMA_' in col or 'EMA_' in col]
        for col in ma_cols[:3]:  # Show first 3 MAs
            fig.add_trace(
                go.Scatter(x=data.index, y=data[col], name=col, line=dict(dash='dash')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
                row=3, col=1
            )
            if 'MACD_Histogram' in data.columns:
                fig.add_trace(
                    go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'),
                    row=3, col=1
                )
        
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            template=self.template,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def create_prediction_chart(self, actual: pd.Series, predicted: pd.Series, 
                              symbol: str, model_name: str) -> go.Figure:
        """
        Create chart comparing actual vs predicted values
        
        Args:
            actual: Actual values
            predicted: Predicted values
            symbol: Stock symbol
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=predicted.index,
                y=predicted.values,
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            title=f'{symbol} - {model_name} Predictions vs Actual',
            xaxis_title='Date',
            yaxis_title='Price',
            template=self.template,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def create_sentiment_chart(self, sentiment_data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create sentiment analysis chart
        
        Args:
            sentiment_data: DataFrame with sentiment data
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Sentiment Score Over Time', 'Sentiment Distribution'),
            row_width=[0.6, 0.4]
        )
        
        # Sentiment score over time
        if 'combined_score_mean' in sentiment_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data.index,
                    y=sentiment_data['combined_score_mean'],
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
        
        # Sentiment distribution
        if 'sentiment_label' in sentiment_data.columns:
            sentiment_counts = sentiment_data['sentiment_label'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    name='Sentiment Count',
                    marker_color=['green', 'red', 'gray']
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} Sentiment Analysis',
            template=self.template,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def create_model_comparison_chart(self, results: Dict[str, Dict], metric: str = 'rmse') -> go.Figure:
        """
        Create chart comparing model performance
        
        Args:
            results: Dictionary with model results
            metric: Metric to compare ('rmse', 'mae', 'accuracy')
            
        Returns:
            Plotly figure
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models if metric in results[model]]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=f'Model Comparison - {metric.upper()}',
            xaxis_title='Model',
            yaxis_title=metric.upper(),
            template=self.template,
            height=400,
            width=600
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            data: DataFrame with features
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=f'{symbol} Feature Correlation Matrix',
            template=self.template,
            height=600,
            width=800
        )
        
        return fig
    
    def create_returns_distribution(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create returns distribution chart
        
        Args:
            data: DataFrame with price data
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns Distribution',
                marker_color='lightblue'
            )
        )
        
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm * len(returns) * (returns.max() - returns.min()) / 50,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            )
        )
        
        fig.update_layout(
            title=f'{symbol} Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            template=self.template,
            height=400,
            width=600
        )
        
        return fig


class DashboardVisualizer:
    """
    Create dashboard-style visualizations
    """
    
    def __init__(self):
        self.stock_viz = StockVisualizer()
    
    def create_overview_dashboard(self, data: pd.DataFrame, sentiment_data: pd.DataFrame, 
                                symbol: str, model_results: Dict = None) -> go.Figure:
        """
        Create comprehensive dashboard
        
        Args:
            data: Stock data
            sentiment_data: Sentiment data
            symbol: Stock symbol
            model_results: Model performance results
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Stock Price', 'Volume',
                'Technical Indicators', 'Sentiment Score',
                'Model Performance', 'Returns Distribution'
            ),
            specs=[
                [{"type": "candlestick"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ]
        )
        
        # Stock price
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
            row=1, col=2
        )
        
        # Technical indicators (RSI)
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
        
        # Sentiment score
        if 'combined_score_mean' in sentiment_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data.index,
                    y=sentiment_data['combined_score_mean'],
                    name='Sentiment',
                    line=dict(color='green')
                ),
                row=2, col=2
            )
        
        # Model performance
        if model_results:
            models = list(model_results.keys())
            rmse_values = [model_results[model]['rmse'] for model in models if 'rmse' in model_results[model]]
            fig.add_trace(
                go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightblue'),
                row=3, col=1
            )
        
        # Returns distribution
        returns = data['Close'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, name='Returns', marker_color='lightgreen'),
            row=3, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Dashboard',
            template=self.stock_viz.template, # Use template from StockVisualizer
            height=800,
            width=1200
        )
        
        return fig


def main():
    """
    Test visualization functionality
    """
    from src.data_collection import StockDataCollector
    from src.feature_engineering import FeatureEngineering
    from src.sentiment_analysis import FinancialNewsProcessor
    
    # Get sample data
    collector = StockDataCollector()
    data = collector.get_stock_data('AAPL', period='3mo')
    
    # Create features
    fe = FeatureEngineering()
    features = fe.prepare_features(data)
    
    # Create sample sentiment data
    processor = FinancialNewsProcessor()
    sample_news = processor.create_sample_news_data('AAPL', days=len(features))
    sentiment_df = processor.process_news_data(sample_news, 'AAPL')
    
    # Create visualizations
    viz = StockVisualizer()
    
    # Price chart
    price_fig = viz.create_price_chart(data, 'AAPL')
    price_fig.show()
    
    # Technical indicators
    tech_fig = viz.create_technical_indicators_chart(features, 'AAPL')
    tech_fig.show()
    
    # Sentiment chart
    sentiment_fig = viz.create_sentiment_chart(sentiment_df, 'AAPL')
    sentiment_fig.show()
    
    # Dashboard
    dashboard_viz = DashboardVisualizer()
    dashboard_fig = dashboard_viz.create_overview_dashboard(data, sentiment_df, 'AAPL')
    dashboard_fig.show()


if __name__ == "__main__":
    main() 