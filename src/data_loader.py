import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class for financial news and stock data
    """
    
    def __init__(self):
        self.data = None
        self.stock_data = None
        
    def load_news_data(self, file_path: str) -> pd.DataFrame:
        """
        Load news data from CSV file
        """
        try:
            self.news_data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded news data with {len(self.news_data)} rows")
            return self.news_data
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            raise
            
    def load_stock_data(self, file_path: str) -> pd.DataFrame:
        """
        Load stock data from CSV file
        """
        try:
            self.stock_data = pd.read_csv(file_path, parse_dates=['Date'])
            self.stock_data.set_index('Date', inplace=True)
            logger.info(f"Successfully loaded stock data with {len(self.stock_data)} rows")
            return self.stock_data
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about loaded datasets
        """
        info = {}
        if hasattr(self, 'news_data') and self.news_data is not None:
            info['news_data_shape'] = self.news_data.shape
            info['news_data_columns'] = self.news_data.columns.tolist()
            info['news_data_info'] = self.news_data.info()
            
        if self.stock_data is not None:
            info['stock_data_shape'] = self.stock_data.shape
            info['stock_data_columns'] = self.stock_data.columns.tolist()
            
        return info