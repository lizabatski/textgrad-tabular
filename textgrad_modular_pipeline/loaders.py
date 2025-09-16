import pandas as pd
import random
from typing import List, Tuple, Dict, Any
import os

def load_iris_dataset(path: str = "datasets/Iris.csv", seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    df = pd.read_csv(path)
    random.seed(seed)
    
    data = [
        (
            {
                "sepal_length": row.SepalLengthCm,
                "sepal_width": row.SepalWidthCm,
                "petal_length": row.PetalLengthCm,
                "petal_width": row.PetalWidthCm,
            },
            row.Species.split("-")[-1].lower()
        )
        for _, row in df.iterrows()
    ]
    random.shuffle(data)
    
    n = len(data)
    return data[:int(0.6*n)], data[int(0.6*n):int(0.8*n)], data[int(0.8*n):]

def load_bio_sample_dataset(path: str = "datasets/Iris.csv", seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    """Load biological sample dataset for classification"""
    df = pd.read_csv(path)
    random.seed(seed)
    
    # Define mappings for masking
    feature_mapping = {
        "sepal_length": "feature_0",
        "sepal_width": "feature_1", 
        "petal_length": "feature_2",
        "petal_width": "feature_3"
    }
    
    class_mapping = {
        "setosa": "class_0",
        "versicolor": "class_1",
        "virginica": "class_2"
    }
    
    data = []
    for _, row in df.iterrows():
        # Create masked features
        features = {
            "feature_0": row.SepalLengthCm,  # originally sepal_length
            "feature_1": row.SepalWidthCm,   # originally sepal_width
            "feature_2": row.PetalLengthCm,  # originally petal_length
            "feature_3": row.PetalWidthCm,   # originally petal_width
        }
        
        # Create masked label
        original_species = row.Species.split("-")[-1].lower()
        masked_label = class_mapping[original_species]
        
        data.append((features, masked_label))
    
    random.shuffle(data)
    
    n = len(data)
    return data[:int(0.6*n)], data[int(0.6*n):int(0.8*n)], data[int(0.8*n):]

def load_heart_dataset(path: str = "datasets/heart.csv", seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    df = pd.read_csv(path)
    random.seed(seed)

    data = []

    for _, row in df.iterrows():
        features = {
            "Age": row.Age,
            "Sex": row.Sex,
            "ChestPainType": row.ChestPainType,
            "RestingBP": row.RestingBP,
            "Cholesterol": row.Cholesterol,
            "FastingBS": row.FastingBS,
            "RestingECG": row.RestingECG,
            "MaxHR": row.MaxHR,
            "ExerciseAngina": row.ExerciseAngina,
            "Oldpeak": row.Oldpeak,
            "ST_Slope": row.ST_Slope
        }

        label = "heart_disease" if row.HeartDisease == 1 else "normal"
        data.append((features, label))

    random.shuffle(data)

    n = len(data)
    return data[:int(0.6 * n)], data[int(0.6 * n):int(0.8 * n)], data[int(0.8 * n):]

def load_synthetic_dataset(path: str = "datasets/synthetic_dataset.csv", seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    """Load synthetic dataset from CSV"""
    df = pd.read_csv(path)
    random.seed(seed)
    
    data = []
    for _, row in df.iterrows():
        features = {
            "feature_0": row.feature_0,
            "feature_1": row.feature_1,
            "feature_2": row.feature_2,
            "feature_3": row.feature_3,
        }
        label = row['class']  # Should be class_0, class_1, class_2
        data.append((features, label))
    
    random.shuffle(data)
    
    n = len(data)
    return data[:int(0.6*n)], data[int(0.6*n):int(0.8*n)], data[int(0.8*n):]

def get_dataset_loader(dataset_name: str):
    loaders = {
        "iris": load_iris_dataset,
        "bio_sample": load_bio_sample_dataset,  # Changed from iris_masked
        "heart": load_heart_dataset,
        "synthetic": load_synthetic_dataset
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()]

def load_dataset(dataset_name: str, dataset_path: str = None, seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    loader = get_dataset_loader(dataset_name)
    
    if dataset_path is None:
        default_paths = {
            "iris": "datasets/Iris.csv",
            "bio_sample": "datasets/Iris.csv",  
            "heart": "datasets/heart.csv",
            "synthetic": "datasets/synthetic_dataset.csv"
        }
        dataset_path = default_paths[dataset_name.lower()]
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading {dataset_name} dataset from: {dataset_path}")
    train_data, val_data, test_data = loader(dataset_path, seed)
    print(f"Dataset loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    
    return train_data, val_data, test_data