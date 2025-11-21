import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """
    Exploratory Data Analysis for financial news data
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
    def descriptive_statistics(self) -> Dict:
        """
        Perform descriptive statistics analysis
        """
        stats = {}
        
        # Text length statistics
        if 'headline' in self.data.columns:
            self.data['headline_length'] = self.data['headline'].str.len()
            stats['headline_length'] = {
                'mean': self.data['headline_length'].mean(),
                'std': self.data['headline_length'].std(),
                'min': self.data['headline_length'].min(),
                'max': self.data['headline_length'].max()
            }
            
        # Articles per publisher
        if 'publisher' in self.data.columns:
            publisher_counts = self.data['publisher'].value_counts()
            stats['publisher_counts'] = publisher_counts.to_dict()
            stats['top_publishers'] = publisher_counts.head(10).to_dict()
            
        return stats
    
    def analyze_publication_dates(self, date_column: str = 'date') -> Dict:
        """
        Analyze publication date trends
        """
        if date_column not in self.data.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return {}
            
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Extract time components
        self.data['publication_year'] = self.data[date_column].dt.year
        self.data['publication_month'] = self.data[date_column].dt.month
        self.data['publication_day'] = self.data[date_column].dt.day
        self.data['publication_dayofweek'] = self.data[date_column].dt.dayofweek
        self.data['publication_hour'] = self.data[date_column].dt.hour
        
        trends = {
            'articles_per_year': self.data['publication_year'].value_counts().sort_index(),
            'articles_per_month': self.data['publication_month'].value_counts().sort_index(),
            'articles_per_dayofweek': self.data['publication_dayofweek'].value_counts().sort_index(),
            'articles_per_hour': self.data['publication_hour'].value_counts().sort_index()
        }
        
        return trends
    
    def create_time_series_plot(self, date_column: str = 'date'):
        """
        Create time series visualization of publication frequency
        """
        if date_column not in self.data.columns:
            return
            
        plt.figure(figsize=(15, 8))
        
        # Resample by day
        daily_counts = self.data.set_index(date_column).resample('D').size()
        
        plt.subplot(2, 2, 1)
        daily_counts.plot(title='Daily Publication Frequency', color=self.colors[0])
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        weekly_counts = self.data.set_index(date_column).resample('W').size()
        weekly_counts.plot(title='Weekly Publication Frequency', color=self.colors[1])
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        month_counts = self.data.set_index(date_column).resample('M').size()
        month_counts.plot(title='Monthly Publication Frequency', color=self.colors[2])
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        hour_counts = self.data['publication_hour'].value_counts().sort_index()
        hour_counts.plot(kind='bar', title='Publication Frequency by Hour', color=self.colors[3])
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        
        plt.tight_layout()
        plt.savefig('publication_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def publisher_analysis(self) -> Dict:
        """
        Analyze publisher patterns and domains
        """
        analysis = {}
        
        if 'publisher' not in self.data.columns:
            return analysis
            
        # Top publishers
        top_publishers = self.data['publisher'].value_counts().head(15)
        analysis['top_publishers'] = top_publishers
        
        # Domain analysis (if publisher contains email-like patterns)
        email_mask = self.data['publisher'].str.contains('@', na=False)
        if email_mask.any():
            domains = self.data.loc[email_mask, 'publisher'].str.split('@').str[1]
            analysis['top_domains'] = domains.value_counts().head(10)
        
        return analysis
    
    def generate_eda_report(self) -> Dict:
        """
        Generate comprehensive EDA report
        """
        report = {
            'descriptive_stats': self.descriptive_statistics(),
            'time_trends': self.analyze_publication_dates(),
            'publisher_analysis': self.publisher_analysis(),
            'data_shape': self.data.shape,
            'data_columns': self.data.columns.tolist(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return report