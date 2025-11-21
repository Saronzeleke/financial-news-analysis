
### Step 5: Testing Framework
import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader

class TestDataLoader:
    
    def test_initialization(self):
        loader = DataLoader()
        assert loader.data is None
        assert loader.stock_data is None
    
    def test_load_news_data(self, sample_news_data):
        loader = DataLoader()
        data = loader.load_news_data(sample_news_data)
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
    
    def test_get_basic_info(self, sample_news_data):
        loader = DataLoader()
        loader.load_news_data(sample_news_data)
        info = loader.get_basic_info()
        assert 'news_data_shape' in info

@pytest.fixture
def sample_news_data(tmp_path):
    """Create sample news data for testing"""
    data = {
        'headline': ['Test headline 1', 'Test headline 2'],
        'publisher': ['Publisher A', 'Publisher B'],
        'date': ['2023-01-01', '2023-01-02']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_news.csv"
    df.to_csv(file_path, index=False)
    return file_path