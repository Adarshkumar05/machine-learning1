"""
Data Preprocessing Module
Currently handles only the train/test split of the raw data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Minimal preprocessing for the debutanizer data"""
    
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing (default: 0.3)
        random_state : int
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def preprocess(self, X, y):
        """
        Train/test split without additional preprocessing steps.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target variable
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy.ndarray
            Train/test split
        """
        print("=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        print(f"\nSplitting data (train: {1-self.test_size:.0%}, test: {self.test_size:.0%})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"\nFinal dataset sizes:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test

