import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class EDAAnalyzer:
    """Exploratory Data Analysis for financial news data"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        self.figsize = (12, 8)
    
    def calculate_descriptive_stats(self) -> Dict:
        """Calculate descriptive statistics for textual lengths"""
        stats = {}
        
        # Headline length statistics
        self.data['headline_length'] = self.data['headline'].str.len()
        stats['headline_length'] = {
            'mean': self.data['headline_length'].mean(),
            'median': self.data['headline_length'].median(),
            'std': self.data['headline_length'].std(),
            'min': self.data['headline_length'].min(),
            'max': self.data['headline_length'].max()
        }
        
        # Articles per publisher
        publisher_counts = self.data['publisher'].value_counts()
        stats['publisher_distribution'] = {
            'top_10_publishers': publisher_counts.head(10).to_dict(),
            'total_publishers': len(publisher_counts),
            'articles_per_publisher_avg': publisher_counts.mean()
        }
        
        return stats
    
    def analyze_publication_trends(self) -> Dict:
        """Analyze publication trends over time"""
        trends = {}
        
        # Daily trends
        daily_counts = self.data.groupby(self.data['publication_date'].dt.date).size()
        trends['daily_stats'] = {
            'mean_articles_per_day': daily_counts.mean(),
            'max_articles_day': daily_counts.idxmax(),
            'max_articles_count': daily_counts.max()
        }
        
        # Day of week analysis
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.data['publication_day'].value_counts()
        day_counts = day_counts.reindex(day_order, fill_value=0)
        trends['day_of_week'] = day_counts.to_dict()
        
        # Hourly analysis
        hour_counts = self.data['publication_hour'].value_counts().sort_index()
        trends['hourly_distribution'] = hour_counts.to_dict()
        
        return trends
    
    def create_publication_trend_plots(self, save_path: str = None):
        """Create visualization for publication trends"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Articles per day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.data['publication_day'].value_counts()
        day_counts = day_counts.reindex(day_order, fill_value=0)
        axes[0, 0].bar(day_counts.index, day_counts.values)
        axes[0, 0].set_title('Articles Published by Day of Week')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Articles per hour
        hour_counts = self.data['publication_hour'].value_counts().sort_index()
        axes[0, 1].plot(hour_counts.index, hour_counts.values, marker='o')
        axes[0, 1].set_title('Articles Published by Hour of Day')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Articles')
        
        # Plot 3: Top publishers
        top_publishers = self.data['publisher'].value_counts().head(10)
        axes[1, 0].barh(top_publishers.index, top_publishers.values)
        axes[1, 0].set_title('Top 10 Publishers by Article Count')
        
        # Plot 4: Headline length distribution
        axes[1, 1].hist(self.data['headline_length'], bins=30, alpha=0.7)
        axes[1, 1].set_title('Headline Length Distribution')
        axes[1, 1].set_xlabel('Headline Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig