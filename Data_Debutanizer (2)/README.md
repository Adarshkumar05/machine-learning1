# Debutanizer Soft Sensor Project
## CH512: Machine Learning Applications to Process Engineering

This project implements a comprehensive machine learning pipeline for developing an optimal predictive model (soft sensor/virtual sensor) for butane concentration in a debutanizer column using multivariate regression techniques.

## Project Structure

```
.
├── data_loader.py          # Data loading module
├── preprocessing.py        # Data preprocessing module
├── feature_selection.py     # Feature selection methods
├── models.py                # Regression models
├── validation.py            # Validation metrics
├── visualization.py        # Visualization functions
├── nn_models.py            # Neural network models (Feed-forward, LSTM, Autoregressive)
├── jitl_models.py          # Just-in-Time Learning (JITL) model
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── output/                 # Generated plots and results
└── Data_Debutanizer/       # Data directory
    └── debutanizer_data.txt
```

## Features

### 1. Data Preprocessing
- Train/test split (70/30)

### 2. Feature Selection Methods
- **Forward selection with k-fold validation**: Greedy forward selection with cross-validation
- **LASSO-based selection**: Uses LASSO regression for feature selection
- **Principal Component Analysis (PCA)**: Dimensionality reduction

### 3. Regression & Neural Network Models
- **Linear Regression (1st order)**: Standard linear regression
- **Linear Regression (2nd order)**: Polynomial regression with degree 2
- **LASSO Regression**: L1 regularization with cross-validation
- **Ridge Regression**: L2 regularization (Bayesian Linear Regression)
- **Principal Component Regression (PCR)**: Regression on PCA components
- **Shallow Feed-forward Neural Network**: Single hidden-layer ANN with learning-rate/regularization sweep
- **Deep Feed-forward Neural Network**: Multi-layer ANN spanning horizontal (width) and vertical (depth) variations
- **Autoregressive Deep Neural Network**: Deep ANN trained on input variables augmented with one- and two-step delayed targets
- **LSTM Sequence Model**: Long-short term memory network trained on sliding windows of the selected inputs
- **Just-in-Time Learning (JITL) Model**: Non-parametric approach using k-NN search and local neural network models

### 4. Validation Metrics
- **Sum of Squared Errors (SSE)**
- **Coefficient of Determination (R²)**
- **Akaike Information Criterion (AIC)**
- **Bayesian Information Criterion (BIC)**

### 5. Visualizations
- **Correlation plots**: Feature-target correlations
- **Residual plots**: Enhanced residual analysis with autocorrelation coefficients
- **Time-series plots**: True vs predicted values w.r.t. time for all models
- **Scatter plots**: Predicted vs actual values with R² scores
- **Autocorrelation plots**: Residual autocorrelation analysis with confidence bounds
- **Metrics comparison tables**: Comprehensive performance metrics for all models

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution

Run the main script to execute the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Perform feature selection
3. Build and validate all regression models
4. Generate all visualizations and reports
5. Save results to the `output/` directory

### Output Files

All results are saved in the `output/` directory:

- **Plots**:
  - `correlation_plot.png`: Feature-target correlations
  - `residual_*.png`: Enhanced residual plots with autocorrelation for each model
  - `timeseries_*.png`: Time-series comparison plots (true vs predicted) for each model
  - `scatter_*.png`: Predicted vs actual scatter plots for each model
  - `autocorr_*.png`: Detailed autocorrelation plots for each model
  - `metrics_comparison.png`: Performance metrics comparison table
  - `final_metrics_comparison.png`: Final comprehensive metrics table

- **Data**:
  - `model_performance_metrics.csv`: Complete performance metrics for all models

## Dataset

The dataset contains:
- **7 input variables (u1-u7)**: Process variables (top temperature, top pressure, reflux flow, etc.)
- **1 output variable (y)**: Butane concentration (shifted by 8 samples to account for time delay)
- **~2,390 samples**: Process measurements

## Model Performance

The script evaluates all models and provides comprehensive performance metrics. The best model is identified based on R² score.

## Customization

- **Preprocessing**: Train/test split ratio
- **Feature Selection**: Number of features, cross-validation folds
- **Models**: Regularization parameters, number of components

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- PyTorch (for neural network models)

## Author

CH512 Course Project - Machine Learning Applications to Process Engineering

## License

This project is for educational purposes.

