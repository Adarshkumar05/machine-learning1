# Regression Models Module

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class RegressionModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
    def linear_regression_1st(self, X_train, y_train, X_test=None, y_test=None):
        """
        First-order linear regression.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_test : numpy.ndarray, optional
            Test features
        y_test : numpy.ndarray, optional
            Test target
            
        Returns:
        --------
        model : LinearRegression
            Fitted model
        y_pred_train : numpy.ndarray
            Training predictions
        y_pred_test : numpy.ndarray or None
            Test predictions
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test) if X_test is not None else None
        
        self.models['linear_1st'] = model
        
        return model, y_pred_train, y_pred_test
    
    def linear_regression_2nd(self, X_train, y_train, X_test=None, y_test=None):
        """
        Second-order polynomial regression.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_test : numpy.ndarray, optional
            Test features
        y_test : numpy.ndarray, optional
            Test target
            
        Returns:
        --------
        model : Pipeline
            Fitted model (PolynomialFeatures + LinearRegression)
        y_pred_train : numpy.ndarray
            Training predictions
        y_pred_test : numpy.ndarray or None
            Test predictions
        """
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test) if X_test is not None else None
        
        self.models['linear_2nd'] = model
        
        return model, y_pred_train, y_pred_test
    
    def lasso_regression(self, X_train, y_train, X_test=None, y_test=None, 
                        alpha=None, cv=5):
        """
        LASSO regression with cross-validation.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_test : numpy.ndarray, optional
            Test features
        y_test : numpy.ndarray, optional
            Test target
        alpha : float, optional
            Regularization parameter (if None, use CV to find optimal)
        cv : int
            Number of folds for cross-validation
            
        Returns:
        --------
        model : Lasso or LassoCV
            Fitted model
        y_pred_train : numpy.ndarray
            Training predictions
        y_pred_test : numpy.ndarray or None
            Test predictions
        """
        if alpha is None:
            alphas = np.logspace(-4, 1, 50)
            model = LassoCV(alphas=alphas, cv=cv, random_state=self.random_state, 
                          max_iter=1000, n_jobs=-1)
        else:
            model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test) if X_test is not None else None
        
        self.models['lasso'] = model
        
        return model, y_pred_train, y_pred_test
    
    def ridge_regression(self, X_train, y_train, X_test=None, y_test=None, 
                        alpha=None, cv=5):
        """
        Ridge regression (Bayesian Linear Regression).
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_test : numpy.ndarray, optional
            Test features
        y_test : numpy.ndarray, optional
            Test target
        alpha : float, optional
            Regularization parameter (if None, use CV to find optimal)
        cv : int
            Number of folds for cross-validation
            
        Returns:
        --------
        model : Ridge or RidgeCV
            Fitted model
        y_pred_train : numpy.ndarray
            Training predictions
        y_pred_test : numpy.ndarray or None
            Test predictions
        """
        if alpha is None:
            alphas = np.logspace(-4, 1, 50)
            model = RidgeCV(alphas=alphas, cv=cv)
        else:
            model = Ridge(alpha=alpha, random_state=self.random_state)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test) if X_test is not None else None
        
        self.models['ridge'] = model
        
        return model, y_pred_train, y_pred_test
    
    def pcr(self, X_train, y_train, X_test=None, y_test=None, 
            n_components=None, explained_variance=0.95):
        """
        Principal Component Regression (PCR).
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_test : numpy.ndarray, optional
            Test features
        y_test : numpy.ndarray, optional
            Test target
        n_components : int, optional
            Number of components (if None, use explained_variance)
        explained_variance : float
            Minimum explained variance ratio
            
        Returns:
        --------
        model : dict
            Dictionary with 'pca' and 'regression' models
        y_pred_train : numpy.ndarray
            Training predictions
        y_pred_test : numpy.ndarray or None
            Test predictions
        """
        # Determine number of components
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_train)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.where(cumsum_var >= explained_variance)[0][0] + 1
        
        # PCA transformation
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train)
        
        # Linear regression on PCA components
        reg = LinearRegression()
        reg.fit(X_train_pca, y_train)
        
        # Predictions
        y_pred_train = reg.predict(X_train_pca)
        
        y_pred_test = None
        if X_test is not None:
            X_test_pca = pca.transform(X_test)
            y_pred_test = reg.predict(X_test_pca)
        
        model = {'pca': pca, 'regression': reg}
        self.models['pcr'] = model
        
        return model, y_pred_train, y_pred_test
    
