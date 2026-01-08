# Model Validation Module

import numpy as np
from sklearn.metrics import r2_score


class ModelValidator:
    @staticmethod
    def sse(y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2)
    
    @staticmethod
    def r2(y_true, y_pred):
        r2_val = r2_score(y_true, y_pred)
        return max(0.0, r2_val)  # don't allow negative R²
    
    @staticmethod
    def aic(y_true, y_pred, n_params):
        n = len(y_true)
        sse = np.sum((y_true - y_pred) ** 2)
        mse = sse / n
        aic = n * np.log(mse) + 2 * n_params
        return aic
    
    @staticmethod
    def bic(y_true, y_pred, n_params):
        n = len(y_true)
        sse = np.sum((y_true - y_pred) ** 2)
        mse = sse / n
        bic = n * np.log(mse) + n_params * np.log(n)
        return bic
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, n_params):
        # Clean predictions
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_true), posinf=np.nanmax(y_true), neginf=np.nanmin(y_true))
        
        metrics = {
            'SSE': ModelValidator.sse(y_true, y_pred),
            'R²': ModelValidator.r2(y_true, y_pred),
            'AIC': ModelValidator.aic(y_true, y_pred, n_params),
            'BIC': ModelValidator.bic(y_true, y_pred, n_params)
        }
        return metrics
    
    @staticmethod
    def count_parameters(model, model_name):
        if model_name == 'linear_1st':
            n_params = len(model.coef_) + 1
        elif model_name == 'linear_2nd':
            # handle polynomial pipeline
            try:
                lin = model.named_steps.get('linear')
                if lin is not None and hasattr(lin, 'coef_'):
                    n_params = len(lin.coef_) + 1
                else:
                    poly = model.named_steps.get('poly')
                    n_poly_features = getattr(poly, 'n_output_features_', None)
                    if n_poly_features is not None:
                        n_params = n_poly_features + 1
                    else:
                        try:
                            final = model.named_steps[list(model.named_steps)[-1]]
                            n_params = len(final.coef_) + 1
                        except:
                            n_params = 1
            except:
                n_params = 1
        elif model_name == 'lasso':
            n_params = np.sum(np.abs(model.coef_) > 1e-5) + 1
        elif model_name == 'ridge':
            n_params = len(model.coef_) + 1
        elif model_name == 'pcr':
            n_components = model['pca'].n_components_
            n_params = n_components + 1
        else:
            try:
                n_params = len(model.coef_) + 1
            except:
                n_params = 1
        
        return n_params

