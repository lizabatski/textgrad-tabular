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

def get_dataset_loader(dataset_name: str):
    loaders = {
        "iris": load_iris_dataset,
        "heart": load_heart_dataset
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()]

def load_dataset(dataset_name: str, dataset_path: str = None, seed: int = 42) -> Tuple[List[Tuple[dict, str]], List[Tuple[dict, str]], List[Tuple[dict, str]]]:
    """Load any supported dataset"""
    loader = get_dataset_loader(dataset_name)
    
    if dataset_path is None:
        default_paths = {
            "iris": "datasets/Iris.csv",
            "heart": "datasets/heart.csv"
        }
        dataset_path = default_paths[dataset_name.lower()]
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading {dataset_name} dataset from: {dataset_path}")
    train_data, val_data, test_data = loader(dataset_path, seed)
    print(f"Dataset loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    
    return train_data, val_data, test_data