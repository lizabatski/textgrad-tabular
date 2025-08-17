import textgrad as tg
import pandas as pd
import random
import json
import os
import re
from typing import List, Tuple, Dict
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient

PROVIDER = "openai" #"openai" 

if PROVIDER == "deepseek":
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    
    # create OpenAI client with DeepSeek API
    client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_key)
    
    # create engines using ChatExternalClient
    generator_engine = ChatExternalClient(client=client, model_string='deepseek-chat')
    evaluator_engine = ChatExternalClient(client=client, model_string='deepseek-reasoner')
    optimizer_engine = ChatExternalClient(client=client, model_string='deepseek-reasoner')
    
elif PROVIDER == "openai":
    generator_engine = tg.get_engine("gpt-3.5-turbo")
    evaluator_engine = tg.get_engine("gpt-4o")
    optimizer_engine = tg.get_engine("gpt-5")

# set backward engine with override=True for DeepSeek
tg.set_backward_engine(optimizer_engine, override=True)

# load and shuffle data but set seed
df = pd.read_csv("datasets/Iris.csv")
random.seed(42)

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

# split data 60:20:20
total_size = len(data)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print(f"Dataset splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

def test_api_connection():
    """Ttest which API is actually being called"""
    print(f"\n{'='*50}")
    print("TESTING API CONNECTION")
    print(f"{'='*50}")
    print(f"Provider setting: {PROVIDER}")
    
    if PROVIDER == "deepseek":
        print(f"DeepSeek API Key: {deepseek_key[:10]}..." if deepseek_key else "NOT SET")
        print(f"API Base: https://api.deepseek.com")
        test_engine = generator_engine  
    else:
        print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'NOT SET')[:10]}...")
        test_engine = generator_engine
    
    print(f"Testing with engine: {type(test_engine).__name__}")
    
    test_prompt = tg.Variable("Say 'Hello from [API_NAME]' and identify which AI service you are.", role_description="test prompt to verify API connection")
    
    try:
        test_model = tg.BlackboxLLM(test_engine)
        response = test_model(test_prompt)
        print(f"\nAPI Response: {response.value}")
        
        response_lower = response.value.lower()
        if "deepseek" in response_lower:
            print("Successfully connected to DEEPSEEK")
        elif "openai" in response_lower or "gpt" in response_lower:
            print("Connected to OpenAI")
        else:
            print("Connected to unknown service")
            
    except Exception as e:
        print(f"API test failed: {e}")
        return False
    
    return True

# processes which iris species was predicted
def extract_prediction(pred_text):
    pred_text = pred_text.lower().strip()
    for s in ["setosa", "versicolor", "virginica"]:
        if pred_text == s or s in pred_text:
            return s
    return "unknown"

# formats input data correctly falls back onto comma separated list
def safe_format(template, sample):
    try:
        return template.format(**sample)
    except Exception:
        return f"{sample['sepal_length']},{sample['sepal_width']},{sample['petal_length']},{sample['petal_width']}"

# Validate that a format string works
def validate_format(format_string, sample):
    try:
        result = format_string.format(**sample)
        return True, result
    except Exception as e:
        return False, str(e)

def track_format_changes(old_format, new_format, epoch, batch):
    if old_format != new_format:
        print(f"   format changed (Epoch {epoch+1}, Batch {batch+1}):")
        print(f"   Old: {old_format}")
        print(f"   New: {new_format}")
        return True
    return False

# initialize global variables
# this is the thing that needs to be changed with gradient updates
serialization_format = tg.Variable(
    "Petal Length:{petal_length:.1f} Petal Width:{petal_width:.1f} Sepal Length:{sepal_length:.1f} Sepal Width:{sepal_width:.1f}",
    requires_grad=True,
    role_description="Serialization format for Iris features"
)

# all of these measurements are from : https://peaceadegbite1.medium.com/iris-flower-classification-60790e9718a1
generator_system_prompt = tg.Variable(
    """You are an expert botanist, please classify Iris flowers into one of three species: setosa, versicolor, or virginica.

    Setosa: Petal length <= 2.0 cm, petal width <= 0.6 cm, sepal length < 6.0 cm, sepal width >= 2.3 cm
    Versicolor: Petal length >= 3.5 cm, petal width >= 0.9 cm, sepal width <= 3.4 cm
    Virginica: Petal length >= 4.0 cm, petal width >= 1.3 cm, sepal width <= 3.8 cm

    Petal measurements are more informative than sepal measurements.""",
    requires_grad=False,
    role_description="System prompt used by the generator model to classify Iris flowers"
)

generator_model = tg.BlackboxLLM(generator_engine, system_prompt=generator_system_prompt)

# Fixed function name and simplified
def create_evaluator_context(serial_format, sample, prediction, true_label, formatted_input):
    return f"""Current serialization format: '{serial_format}'
Sample data: sepal_length={sample['sepal_length']}, sepal_width={sample['sepal_width']}, petal_length={sample['petal_length']}, petal_width={sample['petal_width']}
Formatted as: '{formatted_input}'
Prediction: '{prediction}'
True label: '{true_label}'
Result: {'CORRECT' if prediction == true_label else 'INCORRECT'}

Improve the serialization format using placeholders: {{sepal_length}}, {{sepal_width}}, {{petal_length}}, {{petal_width}}"""

def evaluate_dataset(data_split: List, split_name: str, verbose: bool = True) -> float:
    correct = 0
    
    if verbose:
        print(f"\nEvaluating on {split_name} set ({len(data_split)} samples)...")
        print(f"Using format: {serialization_format.value}")
    
    for idx, (sample, true_label) in enumerate(data_split):
        # inject numbers from dataset into the new serialization format
        formatted_input = safe_format(serialization_format.value, sample)
        prompt = tg.Variable(
            f"Classify this Iris flower:\n{formatted_input}\n\nSpecies:",
            role_description=f"Classification prompt for {split_name} set"
        )

        if verbose and idx % 10 == 0:
            print(f"  Processing {split_name} sample {idx + 1}/{len(data_split)}...")
        
        output = generator_model(prompt)
        prediction = extract_prediction(output.value)
        if prediction == true_label:
            correct += 1
        
        if verbose and idx < 5:  # show first 5 predictions
            print(f"    {formatted_input} -> {prediction} (actual: {true_label}) {'CORRECT' if prediction == true_label else 'INCORRECT'}")
    
    accuracy = correct / len(data_split)
    if verbose:
        print(f"{split_name.capitalize()} accuracy: {accuracy:.1%} ({correct}/{len(data_split)} correct)")
    
    return accuracy

#evaluator_loss_fn is a reusable TextLoss function that will be used in the training loop
# optimizer is a reusable TextGrad optimizer that will be used to update the serialization format 
def train_epoch(epoch: int, batch_size: int = 3, evaluator_loss_fn=None, optimizer=None) -> Dict:
    epoch_feedbacks = []
    
    # shuffle training data at start of epoch
    shuffled_train = train_data.copy()
    random.shuffle(shuffled_train)
    
    num_batches = (len(shuffled_train) + batch_size - 1) // batch_size
    epoch_correct = 0
    epoch_total = 0
    
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch + 1}")
    print(f"{'='*50}")
    print(f"Training samples in this epoch: {len(shuffled_train)}")
    print(f"Number of batches: {num_batches}")
    
    # loop through all batches in the epoch
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(shuffled_train))
        batch_data = shuffled_train[batch_start:batch_end]
        batch_feedbacks = []
        
        print(f"\n=== Batch {batch_idx + 1}/{num_batches} ===")
        print(f"Current format: {serialization_format.value}")
        
        format_before_batch = serialization_format.value
        total_loss = None # will accumulate the loss over the batch
        batch_correct = 0 # tracks correct predictions within this batch
        
        for i, (sample, true_label) in enumerate(batch_data):
            formatted_input = safe_format(serialization_format.value, sample)
            print(f"\nSample {i + 1}/{len(batch_data)}:")
            print(f"  Input: {formatted_input}")
            
            classification_prompt = tg.Variable(
                f"Classify this Iris flower:\n{formatted_input}\n\nSpecies:",
                role_description="Classification prompt",
                predecessors=[serialization_format]
            )
            
            print(f"  Calling generator...")
            generator_output = generator_model(classification_prompt)
            prediction = extract_prediction(generator_output.value)
            
            if prediction == true_label:
                batch_correct += 1
                epoch_correct += 1
            epoch_total += 1
            
            print(f"  Prediction: {prediction} | Actual: {true_label} | {'CORRECT' if prediction == true_label else 'INCORRECT'}")
            
            # create evaluator context (just a string, not a Variable)
            print(f"  Creating evaluator context...")
            dynamic_context = create_evaluator_context(
                serialization_format.value,
                sample,
                prediction,
                true_label,
                formatted_input
            )

            # Combine the generator output and context into a single Variable
            evaluation_input = tg.Variable(
                f"{generator_output.value}\n\nContext:\n{dynamic_context}",
                role_description="Combined evaluation input",
                predecessors=[generator_output]  # maintain gradient flow
            )

            print(f"  Calling evaluator for feedback...")
            
            loss = evaluator_loss_fn(evaluation_input)
                        
            # extract feedback
            raw_feedback = loss.value
            match = re.search(r"<FEEDBACK>(.*?)</FEEDBACK>", raw_feedback, re.DOTALL)
            extracted_feedback = match.group(1).strip() if match else "NO FEEDBACK FOUND"
            
            #skip redunant feedback or instructions to not change
            normalized_feedback = extracted_feedback.lower().strip()
            skip_keywords = ["keep", "no change", "same format", "unchanged", "do not modify"]

            if any(k in normalized_feedback for k in skip_keywords):
                print("  Skipping feedback: Detected redundant/no-change feedback.")
                continue
            
            print(f"  Evaluator feedback: {extracted_feedback}")
            
            # save feedback
            feedback_log = {
                "epoch": epoch,
                "batch": batch_idx,
                "sample": i,
                "formatted_input": formatted_input,
                "prediction": prediction,
                "true_label": true_label,
                "serialization_format_before": serialization_format.value,
                "raw_feedback": raw_feedback,
                "extracted_feedback": extracted_feedback,
                "correct": prediction == true_label
            }
            
            epoch_feedbacks.append(feedback_log)
            batch_feedbacks.append(feedback_log)
            
            # accumulate loss
            if prediction != true_label:
                total_loss = loss if total_loss is None else tg.sum([total_loss, loss])
            print(f"  Loss accumulated.")
        
        batch_accuracy = batch_correct / len(batch_data)
        print(f"\nBatch {batch_idx + 1} Summary:")
        print(f"  Batch accuracy: {batch_accuracy:.1%} ({batch_correct}/{len(batch_data)} correct)")
        
        print("\nBatch feedback summary:")
        for j, fb in enumerate(batch_feedbacks):
            print(f"  Sample {j + 1}: {'CORRECT' if fb['correct'] else 'INCORRECT'}")
            print(f"    Input: {fb['formatted_input']}")
            print(f"    Prediction: {fb['prediction']} (true: {fb['true_label']})")
            print(f"    Feedback: {fb['extracted_feedback']}\n")
        
        # backpropagate and update using the optimizer passed as argument
        if total_loss and optimizer:
            print("Backpropagating combined loss...")
            total_loss.backward()
            
            try:
                print("Optimizer stepping...")
                optimizer.step()
                print(f"  Format update attempted!")
                
                # check if format actually changed
                format_changed = track_format_changes(
                    format_before_batch, 
                    serialization_format.value, 
                    epoch, 
                    batch_idx
                )
                
                # validate the new format
                if format_changed:
                    is_valid, validation_result = validate_format(serialization_format.value, sample)
                    if not is_valid:
                        print(f"   WARNING: New format failed validation: {validation_result}")
                    else:
                        print(f"   New format validated successfully: {validation_result}")
                
                # update feedback logs with final format
                for log in batch_feedbacks:
                    log["serialization_format_after"] = serialization_format.value
                    
            except Exception as e:
                print(f"  Optimizer failed: {e}")
                if hasattr(optimizer, "last_response"):
                    print(f"    Last response: {optimizer.last_response}")
    
    epoch_train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
    return {
        "epoch": epoch,
        "train_accuracy": epoch_train_acc,
        "feedbacks": epoch_feedbacks
    }

def train_with_validation(num_epochs: int = 5, batch_size: int = 3):
    all_results = {
        "epochs": [],
        "best_format": None,
        "best_val_accuracy": 0,
        "final_test_accuracy": 0
    }
    all_feedback_logs = []
    
    print("="*70)
    print("TRAINING STARTED")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Total epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Test samples: {len(test_data)}")
    print(f"\nInitial serialization format:")
    print(f"  {serialization_format.value}")
    print("="*70)
    
    best_format = serialization_format.value
    best_val_accuracy = 0
    
    # create optimizer once outside the training loop
    print("Creating optimizer...")
    optimizer = tg.TextualGradientDescent(
    parameters=[serialization_format],
    constraints=[
        "CRITICAL: You MUST use EXACTLY the format provided in the <FEEDBACK> tags",
        "DO NOT modify or interpret the feedback - copy it exactly as provided",
        "DO NOT create JSON, XML, or any structured format",
        "The format must be a single line of human-readable text",
        "The format MUST use these exact placeholders: {sepal_length}, {sepal_width}, {petal_length}, {petal_width}",
        "The format MUST work with Python's .format() method",
        "If you see the same feedback multiple times, that means it's correct - use it exactly",
        "Example valid format: 'Petal Length: {petal_length:.1f} cm, Petal Width: {petal_width:.1f} cm'",
        "NEVER use double braces {{ or }}, only single braces for placeholders"
    ],
    new_variable_tags=["<FEEDBACK>", "</FEEDBACK>"]
)
    print("Optimizer created successfully!")
    
    for epoch in range(num_epochs):
        epoch_start_time = pd.Timestamp.now() if 'pd' in globals() else None
        
        # run one epoch, passing the reusable optimizer
        epoch_results = train_epoch(epoch, batch_size, evaluator_loss_fn, optimizer)
        
        if "feedbacks" in epoch_results:
            all_feedback_logs.extend(epoch_results["feedbacks"])
        
        # validating
        print(f"\n{'='*50}")
        print(f"VALIDATION - Epoch {epoch + 1}")
        print(f"{'='*50}")
        val_accuracy = evaluate_dataset(val_data, "validation", verbose=True)
        
        # based on validation accuracy we update the best format
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_format = serialization_format.value
            print(f"\n NEW BEST VALIDATION ACCURACY!")
            print(f"   Previous best: {all_results['best_val_accuracy']:.1%}")
            print(f"   New best: {best_val_accuracy:.1%}")
            print(f"   Saving format: {best_format}")
        else:
            print(f"\n   Current val: {val_accuracy:.1%} | Best val: {best_val_accuracy:.1%}")
        
        # store epoch results
        epoch_results["val_accuracy"] = val_accuracy
        all_results["epochs"].append(epoch_results)
        
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*50}")
        print(f"  Train accuracy: {epoch_results['train_accuracy']:.1%}")
        print(f"  Val accuracy: {val_accuracy:.1%}")
        print(f"  Best val accuracy so far: {best_val_accuracy:.1%}")
        print(f"  Current format: {serialization_format.value}")
        
        if epoch_start_time:
            epoch_duration = (pd.Timestamp.now() - epoch_start_time).total_seconds()
            print(f"  Epoch duration: {epoch_duration:.1f} seconds")
            estimated_remaining = epoch_duration * (num_epochs - epoch - 1)
            print(f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
    
    # save all feedback logs
    print(f"\nSaving feedback logs to textgrad_feedback_logs_gpt5.json...")
    with open("textgrad_feedback_logs_gpt5.json", "w") as f:
        json.dump(all_feedback_logs, f, indent=2)
    print(f"Saved {len(all_feedback_logs)} feedback entries")
    
    # final test with best format
    print(f"\n{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}")
    print(f"Restoring best format from validation...")
    serialization_format.set_value(best_format)
    print(f"Best format: {best_format}")
    print(f"Best validation accuracy: {best_val_accuracy:.1%}")
    
    test_accuracy = evaluate_dataset(test_data, "test", verbose=True)
    
    all_results["best_format"] = best_format
    all_results["best_val_accuracy"] = best_val_accuracy
    all_results["final_test_accuracy"] = test_accuracy

    enhanced_results = {
        "experiment_info": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "provider": PROVIDER,
            "generator_model": "deepseek-chat" if PROVIDER == "deepseek" else "gpt-3.5-turbo",
            "evaluator_model": "deepseek-reasoner" if PROVIDER == "deepseek" else "gpt-4o",
            "optimizer_model": "gpt-5",
            "num_epochs": num_epochs,
            "batch_size": batch_size
        },
        "dataset": {
            "train_samples": len(train_data),
            "val_samples": len(val_data), 
            "test_samples": len(test_data)
        },
        "results": {
            "best_val_accuracy": best_val_accuracy,
            "final_test_accuracy": test_accuracy,
            "best_format": best_format,
            "initial_format": "Petal Length:{petal_length:.1f} Petal Width:{petal_width:.1f} Sepal Length:{sepal_length:.1f} Sepal Width:{sepal_width:.1f}"
        },
        "epochs": all_results["epochs"]
    }
    
    print(f"\nSaving enhanced results to training_results_gpt5.json...")
    with open("training_results_gpt5.json", "w") as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"Enhanced results saved")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  - Best validation accuracy: {best_val_accuracy:.1%}")
    print(f"  - Final test accuracy: {test_accuracy:.1%}")
    print(f"  - Best format found: {best_format}")
    print(f"\nFiles saved:")
    print(f"  - textgrad_feedback_logs_gpt5.json (detailed feedback)")
    print(f"  - training_results_gpt5.json (enhanced training metrics)")
    print(f"{'='*70}")
    
    return all_results  

if __name__ == "__main__":
    print("TextGrad Prompt Optimization for Iris Classification")
    print(f"Total dataset size: {len(data)}")

    if not test_api_connection():
        print("API test failed. Please check your configuration.")
        exit(1)

    evaluator_system_prompt = tg.Variable(
    """You are an expert in prompt optimization for Iris flower classification.

EVALUATION TASK:
Analyze the classification result and current format effectiveness.

IF PREDICTION IS CORRECT:
- The current format is working well
- Only suggest MINOR refinements if there's clear room for improvement
- Consider keeping the format as-is

IF PREDICTION IS INCORRECT:
- Identify which features might need more emphasis
- Suggest format changes that highlight discriminative features
- For misclassified samples, emphasize the features that distinguish the true class

DOMAIN KNOWLEDGE:
- Setosa: petal length <= 2.0 cm, petal width <= 0.6 cm
- Versicolor: petal length >= 3.5 cm, petal width >= 0.9 cm  
- Virginica: petal length >= 4.0 cm, petal width >= 1.3 cm
- Petal measurements are MORE informative than sepal measurements

OUTPUT FORMAT:
<FEEDBACK>format_here</FEEDBACK>
<CONFIDENCE>0.0-1.0 score of how much change is needed</CONFIDENCE>""",
    requires_grad=False,
    role_description="System prompt for evaluator"
)

    evaluator_loss_fn = tg.TextLoss(evaluator_system_prompt, engine=evaluator_engine,)

    print("Starting training with fixed architecture...")
    results = train_with_validation(num_epochs=2, batch_size=3)
