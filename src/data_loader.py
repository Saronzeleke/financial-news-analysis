import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

class DataLoader:
    """Data loader class for handling financial news data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Data loaded successfully with {len(self.data)} rows")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Handle missing values
        self.data = self.data.dropna(subset=['headline', 'publisher', 'publication_date'])
        
        # Convert publication date to datetime
        self.data['publication_date'] = pd.to_datetime(self.data['publication_date'])
        
        # Extract date components for time series analysis
        self.data['publication_day'] = self.data['publication_date'].dt.day_name()
        self.data['publication_hour'] = self.data['publication_date'].dt.hour
        self.data['publication_week'] = self.data['publication_date'].dt.isocalendar().week
        
        return self.data
    
    def get_basic_info(self) -> dict:
        """Get basic information about the dataset"""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'date_range': {
                'start': self.data['publication_date'].min(),
                'end': self.data['publication_date'].max()
            },
            'publishers_count': self.data['publisher'].nunique()
        }