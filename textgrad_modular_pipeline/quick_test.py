import pandas as pd
import random
from simple_logger import SimpleLogger

def test_logger():
    """Quick test of the SimpleLogger functionality"""
    print("Testing SimpleLogger...")
    
    logger = SimpleLogger()
    
    # Test data
    test_data = [
        {
            "epoch": 1,
            "batch": 1,
            "sample": 1,
            "formatted_input": "Feature 0: 0.1, Feature 1: 0.7, Feature 2: 1.6, Feature 3: -1.2",
            "prediction": "class_2",
            "true_label": "class_1",
            "prompt_before": "Classify this data sample as: class_0, class_1, or class_2.",
            "raw_feedback": "<FEEDBACK>The prediction was incorrect. Consider feature thresholds.</FEEDBACK>",
            "prompt_after": "Classify this data sample carefully considering feature ranges: class_0, class_1, or class_2.",
            "correct": False
        },
        {
            "epoch": 1,
            "batch": 1,
            "sample": 2,
            "formatted_input": "Feature 0: 0.5, Feature 1: -0.2, Feature 2: 0.8, Feature 3: 1.1",
            "prediction": "class_0",
            "true_label": "class_0",
            "prompt_before": "Classify this data sample carefully considering feature ranges: class_0, class_1, or class_2.",
            "raw_feedback": "<FEEDBACK>Correct prediction! Good job.</FEEDBACK>",
            "prompt_after": "Classify this data sample carefully considering feature ranges: class_0, class_1, or class_2.",
            "correct": True
        }
    ]
    
    # Log the test data
    for data in test_data:
        logger.log_step(
            data["epoch"], data["batch"], data["sample"],
            data["formatted_input"], data["prediction"], data["true_label"],
            data["prompt_before"], data["raw_feedback"], 
            data["prompt_after"], data["correct"]
        )
    
    # Print stats
    # stats = logger.get_stats()
    # print(f"Logged {stats['total_samples']} samples")
    # print(f"Accuracy: {stats['accuracy']:.1%}")
    
    # Save the log
    logger.save("test_provider")
    print("Logger test completed successfully!")

def test_data_loading():
    """Test loading your synthetic dataset"""
    print("\nTesting data loading...")
    
    try:
        df = pd.read_csv('../datasets/synthetic_dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Classes: {df['class'].unique()}")
        print(f"Sample data:\n{df.head(3)}")
        
        # Convert to training format
        data = []
        for _, row in df.iterrows():
            sample = {
                'feature_0': row['feature_0'],
                'feature_1': row['feature_1'], 
                'feature_2': row['feature_2'],
                'feature_3': row['feature_3']
            }
            data.append((sample, row['class']))
        
        # Test train/val/test split
        random.shuffle(data)
        n = len(data)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        print("Data loading test completed successfully!")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def test_minimal_trainer():
    """Test a minimal version of your trainer components"""
    print("\nTesting minimal trainer components...")
    
    # Load data
    train_data, val_data, test_data = test_data_loading()
    if train_data is None:
        print("Skipping trainer test - data loading failed")
        return
    
    # Test formatting function (simplified)
    def simple_format(template, sample):
        return template.format(**sample)
    
    # Test with first few samples
    template = "Feature 0: {feature_0:.3f}, Feature 1: {feature_1:.3f}, Feature 2: {feature_2:.3f}, Feature 3: {feature_3:.3f}"
    
    for i, (sample, label) in enumerate(train_data[:3]):
        formatted = simple_format(template, sample)
        print(f"Sample {i+1}: {formatted} -> {label}")
    
    print("Minimal trainer test completed!")

if __name__ == "__main__":
    print("Running quick tests...")
    print("="*50)
    
    # Test 1: Logger functionality
    test_logger()
    
    # Test 2: Data loading
    test_data_loading()
    
    # Test 3: Minimal trainer components
    test_minimal_trainer()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("If no errors above, your setup should work.")