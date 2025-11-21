import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

class TextAnalyzer:
    """Text analysis and topic modeling for financial news"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.financial_terms = set([
            'stock', 'market', 'price', 'earnings', 'revenue', 'profit', 
            'loss', 'growth', 'investment', 'trading', 'shares', 'dividend',
            'fda', 'approval', 'target', 'financial', 'quarter', 'annual'
        ])
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens, keep financial terms
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words or token in self.financial_terms
            if len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, top_n: int = 20) -> Dict:
        """Extract most common keywords"""
        all_text = ' '.join(self.data['headline'].astype(str))
        processed_text = self.preprocess_text(all_text)
        
        # Count word frequencies
        words = processed_text.split()
        word_freq = Counter(words)
        
        return dict(word_freq.most_common(top_n))
    
    def perform_topic_modeling(self, num_topics: int = 5) -> Tuple:
        """Perform LDA topic modeling"""
        # Preprocess all headlines
        processed_headlines = self.data['headline'].apply(self.preprocess_text)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_headlines)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics, 
            random_state=42,
            max_iter=10
        )
        lda.fit(tfidf_matrix)
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics[f"Topic_{topic_idx + 1}"] = top_features
        
        return topics, lda, vectorizer
    
    def analyze_publisher_domains(self) -> Dict:
        """Analyze publisher domains if emails are used"""
        domain_analysis = {}
        
        # Check if publisher names might be emails
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_publishers = self.data[
            self.data['publisher'].str.match(email_pattern, na=False)
        ]
        
        if not email_publishers.empty:
            # Extract domains
            email_publishers = email_publishers.copy()
            email_publishers.loc[:, 'domain'] = email_publishers['publisher'].str.split('@').str[1]
            
            domain_counts = email_publishers['domain'].value_counts()
            domain_analysis['unique_domains'] = len(domain_counts)
            domain_analysis['top_domains'] = domain_counts.head(10).to_dict()
        else:
            domain_analysis['unique_domains'] = 0
            domain_analysis['top_domains'] = {}
            self.logger.info("No email addresses found in publisher names")
        
        return domain_analysis