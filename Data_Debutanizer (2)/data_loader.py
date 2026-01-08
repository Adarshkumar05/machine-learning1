"""
Data Loading Module for Debutanizer Dataset
Loads and parses the debutanizer_data.txt file
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_debutanizer_data(file_path=None):
    """Load the debutanizer data.

    Attempts these locations in order and uses the first one found:
      1. If file_path is provided and exists, use it.
      2. ./debutanizer_data.txt (current working directory)
      3. <module_dir>/debutanizer_data.txt (same directory as this module)

    Raises FileNotFoundError with a helpful message if none exist.
    """

    tried_paths = []

    candidates = []
    if file_path:
        candidates.append(Path(file_path))

    # Candidate: file in current working directory
    candidates.append(Path.cwd() / 'debutanizer_data.txt')

    # Candidate: file next to this module
    candidates.append(Path(__file__).resolve().parent / 'debutanizer_data.txt')

    # Try each candidate and pick the first that exists
    selected_path = None
    for p in candidates:
        tried_paths.append(p)
        if p.exists() and p.is_file():
            selected_path = p
            break

    if selected_path is None:
        raise FileNotFoundError(
            "Could not find 'debutanizer_data.txt'. Tried the following paths:\n" +
            "\n".join(str(p) for p in tried_paths)
        )

    file_path = selected_path

    # Read the data file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (first 5 lines contain metadata)
    data_lines = []
    for line in lines[5:]:  # Start from line 6 (index 5)
        line = line.strip()
        if line:  # Skip empty lines
            # Parse space-separated values
            values = line.split()
            if len(values) == 8:  # 7 inputs + 1 output
                data_lines.append([float(val) for val in values])
    
    # Convert to numpy array
    data = np.array(data_lines)
    
    # Split into features (u1-u7) and target (y)
    X = data[:, :7]  # First 7 columns
    y = data[:, 7]   # Last column
    
    # Feature names
    feature_names = [f'u{i+1}' for i in range(7)]
    
    print(f"Data loaded successfully:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Feature names: {feature_names}")
    print(f"  - Target variable: y (butane concentration)")
    
    return X, y, feature_names


def load_as_dataframe(file_path=None):
   
    X, y, feature_names = load_debutanizer_data(file_path)
    
    # Create DataFrame
    data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
    data_dict['y'] = y
    df = pd.DataFrame(data_dict)
    
    return df

