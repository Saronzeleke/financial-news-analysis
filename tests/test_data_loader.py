import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader

class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader('dummy_path.csv')
        assert loader.file_path == 'dummy_path.csv'
        assert loader.data is None
    
    def test_data_loading(self, sample_data):
        """Test data loading functionality"""
        loader = DataLoader(sample_data)
        data = loader.load_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
    
    def test_data_cleaning(self, sample_data):
        """Test data cleaning functionality"""
        loader = DataLoader(sample_data)
        loader.load_data()
        cleaned_data = loader.clean_data()
        
        assert 'publication_day' in cleaned_data.columns
        assert 'publication_hour' in cleaned_data.columns
        assert cleaned_data['publication_date'].dtype == 'datetime64[ns]'

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing"""
    data = {
        'headline': ['Test Headline 1', 'Test Headline 2'],
        'publisher': ['Publisher A', 'Publisher B'],
        'publication_date': ['2023-01-01', '2023-01-02']
    }
    df = pd.DataFrame(data)
    
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    
    return str(file_path)