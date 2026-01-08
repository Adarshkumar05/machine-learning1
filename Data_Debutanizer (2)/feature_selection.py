# Feature Selection Module

import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = {}
        self.pca_model = None
    
    def forward_selection(self, X, y, k_folds=5, max_features=None, scoring='r2'):
        n_features = X.shape[1]
        if max_features is None:
            max_features = n_features
        
        selected = []
        remaining = list(range(n_features))
        cv_scores = []
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        
        print(f"Forward selection with {k_folds}-fold CV:")
        
        for i in range(min(max_features, n_features)):
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining:
                candidate_features = selected + [feature]
                X_candidate = X[:, candidate_features]
                
                model = LinearRegression()
                scores = cross_val_score(model, X_candidate, y, cv=kf, 
                                        scoring=scoring, n_jobs=-1)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = feature
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                cv_scores.append(best_score)
                print(f"  Step {i+1}: Added feature {best_feature}, CV score: {best_score:.4f}")
        
        selected_indices = np.array(selected)
        self.selected_features['forward'] = selected_indices
        
        return selected_indices, cv_scores
    
    def lasso_selection(self, X, y, cv=5, alphas=None):
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)
        
        lasso = LassoCV(alphas=alphas, cv=cv, random_state=self.random_state, 
                       max_iter=1000, n_jobs=-1)
        lasso.fit(X, y)
        selected_indices = np.where(np.abs(lasso.coef_) > 1e-5)[0]
        
        self.selected_features['lasso'] = selected_indices
        
        print(f"LASSO-based selection:")
        print(f"  Optimal alpha: {lasso.alpha_:.6f}")
        print(f"  Selected {len(selected_indices)} features: {selected_indices}")
        print(f"  Coefficients: {lasso.coef_[selected_indices]}")
        
        return selected_indices, lasso
    
    def pca_transform(self, X, n_components=None, explained_variance=0.95):
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.where(cumsum_var >= explained_variance)[0][0] + 1
        
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca_model.fit_transform(X)
        
        print(f"PCA transformation:")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Principal components: {n_components}")
        print(f"  Explained variance: {self.pca_model.explained_variance_ratio_.sum():.4f}")
        print(f"  Individual variances: {self.pca_model.explained_variance_ratio_}")
        
        return X_pca, self.pca_model
    
    def apply_selection(self, X, method='forward', **kwargs):
        if method not in self.selected_features:
            raise ValueError(f"Method '{method}' not computed yet.")
        
        if method == 'pca':
            if self.pca_model is None:
                raise ValueError("PCA model not fitted yet.")
            return self.pca_model.transform(X)
        else:
            indices = self.selected_features[method]
            return X[:, indices]

