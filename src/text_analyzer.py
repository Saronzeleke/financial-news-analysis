import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
from collections import Counter
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """
    Text analysis and topic modeling for financial news
    """
    
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.setup_nltk()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            logger.warning("NLTK downloads failed - using basic preprocessing")
            
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        # Add financial stopwords
        financial_stopwords = {'said', 'inc', 'corp', 'ltd', 'co', 'year', 'quarter', 'month'}
        stop_words.update(financial_stopwords)
        
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, texts: List[str], max_features: int = 100) -> Dict:
        """
        Extract common keywords using TF-IDF
        """
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top keywords
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_keywords = dict(zip(feature_names, tfidf_scores))
            top_keywords = dict(sorted(top_keywords.items(), 
                                     key=lambda x: x[1], reverse=True)[:20])
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return {}
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 5) -> Dict:
        """
        Perform topic modeling using LDA and NMF
        """
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Create document-term matrix
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            dtm = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
            lda.fit(dtm)
            
            # NMF
            nmf = NMF(
                n_components=n_topics,
                random_state=42
            )
            nmf.fit(dtm)
            
            # Extract topics
            lda_topics = self._extract_topics(lda, feature_names, n_words=10)
            nmf_topics = self._extract_topics(nmf, feature_names, n_words=10)
            
            return {
                'lda_topics': lda_topics,
                'nmf_topics': nmf_topics,
                'feature_names': feature_names.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {}
    
    def _extract_topics(self, model, feature_names: np.ndarray, n_words: int = 10) -> List[Dict]:
        """
        Extract top words for each topic
        """
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words = ', '.join(top_features)
            topics.append({
                'topic_id': topic_idx,
                'words': top_features,
                'topic_description': topic_words
            })
        return topics
    
    def analyze_financial_phrases(self, texts: List[str]) -> Dict:
        """
        Analyze specific financial phrases and events
        """
        financial_phrases = {
            'fda_approval': ['fda approval', 'food and drug administration', 'regulatory approval'],
            'price_target': ['price target', 'target price', 'raised to', 'lowered to'],
            'earnings': ['earnings report', 'quarterly earnings', 'profit', 'revenue'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'buyout'],
            'clinical_trial': ['clinical trial', 'phase trial', 'trial results']
        }
        
        phrase_counts = {}
        for phrase_category, phrases in financial_phrases.items():
            count = 0
            for text in texts:
                text_lower = text.lower()
                for phrase in phrases:
                    if phrase in text_lower:
                        count += 1
                        break
            phrase_counts[phrase_category] = count
            
        return phrase_counts