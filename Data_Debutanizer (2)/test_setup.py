"""
Quick test script to verify the setup and data loading
"""

import sys
import numpy as np

print("Testing setup...")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

try:
    from data_loader import load_debutanizer_data
    print("\n✓ Data loader module imported successfully")
    
    # Test data loading
    print("\nTesting data loading...")
    X, y, feature_names = load_debutanizer_data('Data_Debutanizer/debutanizer_data.txt')
    print(f"✓ Data loaded successfully")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Features: {feature_names}")
    
    # Test other modules
    from preprocessing import DataPreprocessor
    print("\n✓ Preprocessing module imported successfully")
    
    from feature_selection import FeatureSelector
    print("✓ Feature selection module imported successfully")
    
    from models import RegressionModels
    print("✓ Models module imported successfully")
    
    from validation import ModelValidator
    print("✓ Validation module imported successfully")
    
    from visualization import Visualizer
    print("✓ Visualization module imported successfully")
    
    print("\n" + "="*60)
    print("✓ All modules imported successfully!")
    print("✓ Setup is ready!")
    print("="*60)
    print("\nYou can now run: python main.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

