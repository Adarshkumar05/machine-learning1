# Visualization Module

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


class Visualizer:
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
    
    def correlation_plot(self, X, y, feature_names=None, save_path=None):
        if feature_names is None:
            feature_names = [f'u{i+1}' for i in range(X.shape[1])]
        
        correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(range(len(feature_names)), correlations, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(feature_names))))
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Correlation with Target (y)', fontsize=12)
        ax.set_title('Feature-Target Correlation Plot', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def residual_plot(self, y_true, y_pred, sample_indices=None, save_path=None):
        residuals = y_true - y_pred
        
        if sample_indices is None:
            sample_indices = np.arange(len(residuals))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(sample_indices, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Zero residual')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Residuals (y_true - y_pred)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax.text(0.02, 0.98, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def scatter_plot(self, y_true, y_pred, model_name='Model', save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect prediction')
        
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def autocorrelation_plot(self, residuals, max_lag=None, save_path=None):
        if max_lag is None:
            max_lag = len(residuals) // 2
        
        autocorr = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] 
                   if lag > 0 else 1.0
                   for lag in range(max_lag + 1)]
        
        lags = np.arange(max_lag + 1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(lags, autocorr, width=0.8, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.95/np.sqrt(len(residuals)), color='r', 
                  linestyle='--', linewidth=1, label='95% confidence')
        ax.axhline(y=-0.95/np.sqrt(len(residuals)), color='r', 
                  linestyle='--', linewidth=1)
        
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title('Residual Autocorrelation Plot', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, max_lag + 0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def metrics_comparison_table(self, metrics_dict, save_path=None):
        df = pd.DataFrame(metrics_dict).T
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values.round(4),
                        rowLabels=df.index,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(df) + 1):
            if i % 2 == 0:
                for j in range(len(df.columns)):
                    table[(i, j)].set_facecolor('#f1f1f2')
        
        plt.title('Model Performance Comparison', fontsize=16, 
                 fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def time_series_plot(self, y_true, y_pred, model_name='Model', 
                        time_indices=None, save_path=None):
        if time_indices is None:
            time_indices = np.arange(len(y_true))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(time_indices, y_true, 'b-', linewidth=1.5, 
               label='True Values', alpha=0.7)
        ax.plot(time_indices, y_pred, 'r--', linewidth=1.5, 
               label='Predicted Values', alpha=0.7)
        
        ax.set_xlabel('Time / Sample Index', fontsize=12)
        ax.set_ylabel('Target Value (y)', fontsize=12)
        ax.set_title(f'{model_name}: True vs Predicted (Time Series)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        sse = np.sum((y_true - y_pred) ** 2)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        ax.text(0.02, 0.98, 
               f'R² = {r2:.4f}\nSSE = {sse:.4f}\nRMSE = {rmse:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def enhanced_residual_plot(self, y_true, y_pred, sample_indices=None, 
                             save_path=None, model_name='Model'):
        residuals = y_true - y_pred
        
        if sample_indices is None:
            sample_indices = np.arange(len(residuals))
        
        if len(residuals) > 1:
            autocorr_coef = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            autocorr_coef = 0.0
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.scatter(sample_indices, residuals, alpha=0.6, s=20, color='blue')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Zero residual')
        ax1.set_xlabel('Sample Index / Time', fontsize=12)
        ax1.set_ylabel('Residuals (y_true - y_pred)', fontsize=12)
        ax1.set_title(f'{model_name}: Residual Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax1.text(0.02, 0.98, 
                f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}\nAutocorr (lag-1): {autocorr_coef:.4f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        max_lag = min(50, len(residuals) // 2)
        if max_lag > 0:
            autocorr = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    autocorr.append(1.0)
                elif lag < len(residuals):
                    corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    autocorr.append(corr)
                else:
                    break
            
            lags = np.arange(len(autocorr))
            ax2.bar(lags, autocorr, width=0.8, alpha=0.7, color='green')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            conf_bound = 1.96 / np.sqrt(len(residuals))
            ax2.axhline(y=conf_bound, color='r', linestyle='--', 
                       linewidth=1, label='95% confidence')
            ax2.axhline(y=-conf_bound, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Lag', fontsize=12)
            ax2.set_ylabel('Autocorrelation Coefficient', fontsize=12)
            ax2.set_title(f'{model_name}: Residual Autocorrelation', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlim(-0.5, len(autocorr) - 0.5)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return autocorr_coef

