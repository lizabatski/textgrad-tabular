import numpy as np
import pandas as pd
import os

def generate_synthetic_dataset(n_samples=100, n_features=4, seed=42):
    """Generate synthetic 4-class dataset and save to CSV"""
    np.random.seed(seed)
    
    # feature names will be feature_0, feature_1, etc. 
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # random gaussian features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # initialize labels
    y = np.zeros(n_samples, dtype=int)
    
    # decision rules for 4 classes based on feature_0 and feature_2
    for i in range(n_samples):
        if X[i, 0] <= 0 and X[i, 1] <= 0: 
            y[i] = 0
        elif X[i, 0] > 0 and X[i, 2] > 0.5:  
            y[i] = 1
        else:  
            y[i] = 2
    
    # Create DataFrame and save
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = [f"class_{label}" for label in y]
    
    os.makedirs('datasets', exist_ok=True)
    csv_path = 'datasets/synthetic_dataset.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Generated and saved {n_samples} samples to {csv_path}")
    
    # Print class distribution
    class_counts = df['class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} samples")

if __name__ == "__main__":
    generate_synthetic_dataset()
    print("Dataset generation complete!")