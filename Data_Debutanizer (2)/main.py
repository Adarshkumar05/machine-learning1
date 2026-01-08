# Debutanizer Soft Sensor Project - Main Script
# CH512 Course Project

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # needed for running without display
from pathlib import Path

from data_loader import load_debutanizer_data
from preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from models import RegressionModels
from validation import ModelValidator
from visualization import Visualizer
from nn_models import NeuralModels, create_autoregressive_matrix, prepare_lstm_sequences
from jitl_models import JITLModel

np.random.seed(42)  # for reproducibility

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)


def main():
    print("=" * 80)
    print("DEBUTANIZER SOFT SENSOR PROJECT")
    print("CH512: Machine Learning Applications to Process Engineering")
    print("=" * 80)
    
    # Load data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)
    
    X, y, feature_names = load_debutanizer_data()
    
    # Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    
    preprocessor = DataPreprocessor(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y)
    
    # Correlation plots
    print("\n" + "=" * 80)
    print("STEP 3: DATA VISUALIZATION")
    print("=" * 80)
    
    visualizer = Visualizer(figsize=(12, 6))
    print("\nCreating correlation plot...")
    visualizer.correlation_plot(X_train, y_train, feature_names=feature_names,
                                save_path=output_dir / 'correlation_plot.png')
    
    # Feature selection
    print("\n" + "=" * 80)
    print("STEP 4: FEATURE SELECTION")
    print("=" * 80)
    
    feature_selector = FeatureSelector(random_state=42)
    print("\n4.1 Forward selection...")
    forward_indices, forward_scores = feature_selector.forward_selection(
        X_train, y_train, k_folds=5, max_features=5
    )
    
    print("\n4.2 LASSO selection...")
    lasso_indices, lasso_model = feature_selector.lasso_selection(
        X_train, y_train, cv=5
    )
    
    print("\n4.3 PCA...")
    X_train_pca, pca_model = feature_selector.pca_transform(
        X_train, explained_variance=0.95
    )
    X_test_pca = feature_selector.pca_model.transform(X_test)
    
    # Build models
    print("\n" + "=" * 80)
    print("STEP 5: MODEL BUILDING")
    print("=" * 80)
    
    models = RegressionModels(random_state=42)
    validator = ModelValidator()
    nn_models = NeuralModels(random_state=42)
    results = {}
    
    print("\n5.1 Linear Regression (1st order)...")
    model_1st, y_pred_train_1st, y_pred_test_1st = models.linear_regression_1st(
        X_train, y_train, X_test, y_test
    )
    n_params_1st = validator.count_parameters(model_1st, 'linear_1st')
    metrics_1st = validator.calculate_all_metrics(y_test, y_pred_test_1st, n_params_1st)
    results['Linear Regression (1st)'] = metrics_1st
    print(f"  Test R²: {metrics_1st['R²']:.4f}, SSE: {metrics_1st['SSE']:.4f}")
    
    # 5.2 Linear Regression (2nd order)
    print("\n5.2 Linear Regression (2nd order)...")
    model_2nd, y_pred_train_2nd, y_pred_test_2nd = models.linear_regression_2nd(
        X_train, y_train, X_test, y_test
    )
    n_params_2nd = validator.count_parameters(model_2nd, 'linear_2nd')
    metrics_2nd = validator.calculate_all_metrics(y_test, y_pred_test_2nd, n_params_2nd)
    results['Linear Regression (2nd)'] = metrics_2nd
    print(f"  Test R²: {metrics_2nd['R²']:.4f}, SSE: {metrics_2nd['SSE']:.4f}")
    
    # 5.3 LASSO Regression
    print("\n5.3 LASSO Regression...")
    model_lasso, y_pred_train_lasso, y_pred_test_lasso = models.lasso_regression(
        X_train, y_train, X_test, y_test, cv=5
    )
    n_params_lasso = validator.count_parameters(model_lasso, 'lasso')
    metrics_lasso = validator.calculate_all_metrics(y_test, y_pred_test_lasso, n_params_lasso)
    results['LASSO Regression'] = metrics_lasso
    print(f"  Test R²: {metrics_lasso['R²']:.4f}, SSE: {metrics_lasso['SSE']:.4f}")
    
    # 5.4 Ridge Regression
    print("\n5.4 Ridge Regression...")
    model_ridge, y_pred_train_ridge, y_pred_test_ridge = models.ridge_regression(
        X_train, y_train, X_test, y_test, cv=5
    )
    n_params_ridge = validator.count_parameters(model_ridge, 'ridge')
    metrics_ridge = validator.calculate_all_metrics(y_test, y_pred_test_ridge, n_params_ridge)
    results['Ridge Regression'] = metrics_ridge
    print(f"  Test R²: {metrics_ridge['R²']:.4f}, SSE: {metrics_ridge['SSE']:.4f}")
    
    # 5.5 Principal Component Regression (PCR)
    print("\n5.5 Principal Component Regression (PCR)...")
    model_pcr, y_pred_train_pcr, y_pred_test_pcr = models.pcr(
        X_train, y_train, X_test, y_test, explained_variance=0.95
    )
    n_params_pcr = validator.count_parameters(model_pcr, 'pcr')
    metrics_pcr = validator.calculate_all_metrics(y_test, y_pred_test_pcr, n_params_pcr)
    results['PCR'] = metrics_pcr
    print(f"  Test R²: {metrics_pcr['R²']:.4f}, SSE: {metrics_pcr['SSE']:.4f}")

    # ============================================================================
    # 5B. NEURAL NETWORK MODELS
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5B: NEURAL NETWORK MODELS")
    print("=" * 80)

    # Dictionary to store predictions for visualization
    model_predictions = {}

    # Use all features for NN
    X_train_nn = X_train
    X_test_nn = X_test

    print("\n5B.1 Shallow NN...")
    shallow_result = nn_models.train_feedforward(
        X_train_nn, y_train,
        X_test_nn, y_test,
        hidden_layer_configs=[(12,), (15,), (18,), (20,)],
        learning_rates=[1e-2, 2e-2, 3e-2],
        weight_decays=[0.0, 1e-7],
        batch_size=64,
        dropout=0.0,
        max_epochs=400,
        patience=40,
        use_batch_norm=True
    )
    metrics_shallow = validator.calculate_all_metrics(
        y_test, shallow_result['test_pred'], shallow_result['n_params']
    )
    results['Shallow NN'] = metrics_shallow
    model_predictions['Shallow NN'] = {'y_true': y_test, 'y_pred': shallow_result['test_pred']}
    print(f"  Best config: {shallow_result['params']}, Test R²: {metrics_shallow['R²']:.4f}")

    print("\n5B.2 Deep NN...")
    deep_result = nn_models.train_feedforward(
        X_train_nn, y_train,
        X_test_nn, y_test,
        hidden_layer_configs=[
            (18, 10),
            (20, 12),
            (18, 12, 8),
            (20, 15, 10)
        ],
        learning_rates=[1e-2, 2e-2, 3e-2],
        weight_decays=[0.0, 1e-7],
        batch_size=64,
        dropout=0.0,
        max_epochs=500,
        patience=50,
        use_batch_norm=True
    )
    metrics_deep = validator.calculate_all_metrics(
        y_test, deep_result['test_pred'], deep_result['n_params']
    )
    results['Deep NN'] = metrics_deep
    model_predictions['Deep NN'] = {'y_true': y_test, 'y_pred': deep_result['test_pred']}
    print(f"  Best config: {deep_result['params']}, Test R²: {metrics_deep['R²']:.4f}")

    print("\n5B.3 Autoregressive NN...")
    X_autoreg, y_autoreg = create_autoregressive_matrix(X, y, lags=(1, 2))
    split_idx = max(int(0.7 * len(X_autoreg)), 1)
    if split_idx >= len(X_autoreg):
        split_idx = len(X_autoreg) - 1
    X_autoreg_train, X_autoreg_test = X_autoreg[:split_idx], X_autoreg[split_idx:]
    y_autoreg_train, y_autoreg_test = y_autoreg[:split_idx], y_autoreg[split_idx:]

    autoreg_result = nn_models.train_autoregressive_nn(
        X_autoreg_train, y_autoreg_train,
        X_autoreg_test, y_autoreg_test,
        hidden_layer_configs=[
            (18, 10),
            (20, 12),
            (18, 12, 8),
            (20, 15, 10)
        ],
        learning_rates=[1e-2, 2e-2, 3e-2],
        weight_decays=[0.0, 1e-7],
        batch_size=64,
        dropout=0.0,
        max_epochs=500,
        patience=50,
        use_batch_norm=True
    )
    metrics_autoreg = validator.calculate_all_metrics(
        y_autoreg_test, autoreg_result['test_pred'], autoreg_result['n_params']
    )
    results['Autoregressive Deep NN'] = metrics_autoreg
    model_predictions['Autoregressive Deep NN'] = {
        'y_true': y_autoreg_test,
        'y_pred': autoreg_result['test_pred']
    }
    print(f"  Best config: {autoreg_result['params']}, Test R²: {metrics_autoreg['R²']:.4f}")

    print("\n5B.4 LSTM...")
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = prepare_lstm_sequences(
        X, y, seq_len=10, train_ratio=0.7
    )
    lstm_result = nn_models.train_lstm(
        X_lstm_train, y_lstm_train,
        X_lstm_test, y_lstm_test,
        hidden_sizes=[15, 18, 20],
        num_layers_options=[1, 2],
        learning_rates=[1e-2, 2e-2, 3e-2],
        weight_decays=[0.0, 1e-7],
        batch_size=64,
        dropout=0.0,
        max_epochs=400,
        patience=40
    )
    metrics_lstm = validator.calculate_all_metrics(
        y_lstm_test, lstm_result['test_pred'], lstm_result['n_params']
    )
    results['LSTM'] = metrics_lstm
    model_predictions['LSTM'] = {'y_true': y_lstm_test, 'y_pred': lstm_result['test_pred']}
    print(f"  Best config: {lstm_result['params']}, Test R²: {metrics_lstm['R²']:.4f}")
    
    # ============================================================================
    # 5C. JUST-IN-TIME LEARNING (JITL) MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5C: JUST-IN-TIME LEARNING (JITL) MODEL")
    print("=" * 80)
    
    print("\n5C.1 JITL model...")
    jitl_model = JITLModel(
        k_neighbors=15,
        hidden_units=10,
        max_epochs=100,
        learning_rate=0.01,
        batch_size=None,  # Use all k neighbors
        random_state=42
    )
    
    # Fit and predict using JITL
    y_pred_jitl = jitl_model.fit_predict(
        X_train_nn, y_train,
        X_test_nn, y_test
    )
    
    # Estimate params for JITL (local NN has input*hidden + hidden + hidden + 1)
    n_params_jitl = X_train_nn.shape[1] * 10 + 10 + 10 + 1
    metrics_jitl = validator.calculate_all_metrics(y_test, y_pred_jitl, n_params_jitl)
    results['JITL (k-NN + Local NN)'] = metrics_jitl
    model_predictions['JITL (k-NN + Local NN)'] = {'y_true': y_test, 'y_pred': y_pred_jitl}
    print(f"  Test R²: {metrics_jitl['R²']:.4f}, SSE: {metrics_jitl['SSE']:.4f}")
    
    # Analysis and plots
    print("\n" + "=" * 80)
    print("STEP 6: ANALYSIS")
    print("=" * 80)
    
    autocorr_coefficients = {}
    for model_name, prediction_data in model_predictions.items():
        y_true_model = prediction_data['y_true']
        y_pred = prediction_data['y_pred']
        print(f"\nCreating plots for {model_name}...")
        
        autocorr_coef = visualizer.enhanced_residual_plot(
            y_true_model, y_pred, model_name=model_name,
            save_path=output_dir / f'residual_{model_name.replace(" ", "_")}.png'
        )
        autocorr_coefficients[model_name] = autocorr_coef
        
        visualizer.time_series_plot(
            y_true_model, y_pred, model_name=model_name,
            save_path=output_dir / f'timeseries_{model_name.replace(" ", "_")}.png'
        )
        
        visualizer.scatter_plot(
            y_true_model, y_pred, model_name=model_name,
            save_path=output_dir / f'scatter_{model_name.replace(" ", "_")}.png'
        )
        
        residuals = y_true_model - y_pred
        visualizer.autocorrelation_plot(
            residuals, max_lag=50,
            save_path=output_dir / f'autocorr_{model_name.replace(" ", "_")}.png'
        )
    
    print("\nCreating metrics table...")
    visualizer.metrics_comparison_table(
        results, save_path=output_dir / 'metrics_comparison.png'
    )
    
    print("\n" + "=" * 80)
    print("AUTOCORRELATION ANALYSIS")
    print("=" * 80)
    print(f"{'Model':<30} {'Autocorr. Coef. (lag-1)':<25}")
    print("-" * 80)
    for model_name, autocorr_coef in autocorr_coefficients.items():
        print(f"{model_name:<30} {autocorr_coef:<25.4f}")
    print("-" * 80)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    visualizer.metrics_comparison_table(
        results, save_path=output_dir / 'final_metrics_comparison.png'
    )
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'model_performance_metrics.csv')
    print(f"\nResults saved to: {output_dir / 'model_performance_metrics.csv'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nModel Performance (Test Set):")
    print("-" * 80)
    print(f"{'Model':<30} {'R²':<10} {'SSE':<15} {'AIC':<15} {'BIC':<15}")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['R²']:<10.4f} {metrics['SSE']:<15.4f} "
              f"{metrics['AIC']:<15.4f} {metrics['BIC']:<15.4f}")
    print("-" * 80)
    
    best_model = max(results.items(), key=lambda x: x[1]['R²'])
    print(f"\nBest Model (by R²): {best_model[0]}")
    print(f"  R²: {best_model[1]['R²']:.4f}")
    print(f"  SSE: {best_model[1]['SSE']:.4f}")
    
    # Check R² values
    print("\n" + "=" * 80)
    print("R² VALIDATION")
    print("=" * 80)
    negative_r2_models = [name for name, metrics in results.items() if metrics['R²'] < 0]
    if negative_r2_models:
        print(f"WARNING: {len(negative_r2_models)} model(s) with negative R²:")
        for name in negative_r2_models:
            print(f"  - {name}: R² = {results[name]['R²']:.4f}")
    else:
        print("All models have non-negative R²")
    
    # Check if NN beats linear models
    nn_models_list = ['Shallow NN', 'Deep NN', 'Autoregressive Deep NN', 'LSTM', 'JITL (k-NN + Local NN)']
    linear_models_list = ['Linear Regression (1st)', 'Linear Regression (2nd)', 'LASSO Regression', 
                          'Ridge Regression', 'PCR']
    
    nn_best_r2 = max([results.get(name, {}).get('R²', -1) for name in nn_models_list if name in results], default=-1)
    linear_best_r2 = max([results.get(name, {}).get('R²', -1) for name in linear_models_list if name in results], default=-1)
    
    if nn_best_r2 > linear_best_r2:
        print(f"Neural networks outperform linear methods (NN: {nn_best_r2:.4f} > Linear: {linear_best_r2:.4f})")
    else:
        print(f"Warning: NN may need tuning (NN: {nn_best_r2:.4f} <= Linear: {linear_best_r2:.4f})")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots and results saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()

