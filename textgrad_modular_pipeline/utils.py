import json
import re
from datetime import datetime
from .dataset_configs import get_dataset_config
import textgrad as tg

def evaluate(model, data, extract_prediction):
    correct = 0
    for features, label in data:
        inputs = {k: tg.Variable(v) for k, v in features.items()}
        output = model.forward(inputs)
        prediction = extract_prediction(output.value)
        if prediction == label:
            correct += 1
    return correct / len(data)

def extract_prediction(pred_text: str, dataset_name: str = "iris") -> str:
    config = get_dataset_config(dataset_name)
    pred_text = pred_text.lower().strip()
    
    for class_name in config["classes"]:
        if pred_text == class_name.lower() or class_name.lower() in pred_text:
            return class_name.lower()
    
    
    if dataset_name.lower() == "heart":
        if dataset_name.lower() == "heart":
            if any(word in pred_text for word in ["disease", "positive", "yes", "1", "cardiac", "abnormal"]):
                return "heart_disease"  
            elif any(word in pred_text for word in ["no_disease", "healthy", "negative", "no", "0", "normal"]):
                return "normal" 
    
    return "unknown"

def safe_format(template: str, sample: dict, dataset_name: str = "iris") -> str:
    try:
        return template.format(**sample)
    except Exception as e:
        print(f"Format error: {e}")
        config = get_dataset_config(dataset_name)
        values = [str(sample.get(feature, "N/A")) for feature in config["features"]]
        return ",".join(values)

def validate_format(format_string: str, sample: dict) -> tuple:
    try:
        result = format_string.format(**sample)
        return True, result
    except Exception as e:
        return False, str(e)

def track_format_changes(old_format: str, new_format: str, epoch: int, batch: int) -> bool:
    """Track and log format changes"""
    if old_format != new_format:
        print(f"   format changed (Epoch {epoch+1}, Batch {batch+1}):")
        print(f"   Old: {old_format}")
        print(f"   New: {new_format}")
        return True
    return False

def create_evaluator_context(serial_format: str, sample: dict, prediction: str, 
                           true_label: str, formatted_input: str, dataset_name: str = "iris") -> str:
    """Create evaluator context with dataset-specific information"""
    config = get_dataset_config(dataset_name)
    
    # Create sample data string
    sample_data = ", ".join([f"{feature}={sample.get(feature, 'N/A')}" for feature in config["features"]])
    
    # Create placeholder information
    placeholder_info = "{" + "}, {".join(config["features"]) + "}"
    
    return f"""Current serialization format: '{serial_format}'
Sample data: {sample_data}
Formatted as: '{formatted_input}'
Prediction: '{prediction}'
True label: '{true_label}'
Result: {'CORRECT' if prediction == true_label else 'INCORRECT'}

Improve the serialization format using placeholders: {placeholder_info}"""