import json
import textgrad as tg
import re
import random
import pandas as pd
from .utils import extract_prediction, safe_format, validate_format, track_format_changes, create_evaluator_context
from .simple_logger import SimpleLogger
from .dataset_configs import get_dataset_config
from typing import List, Tuple, Dict


def evaluate_dataset(data, dataset_type="test", verbose=False, serialization_format=None, 
                    generator_model=None, dataset_name="iris"):
    correct = 0
    total = len(data)
    
    for sample, true_label in data:
        formatted_input = safe_format(serialization_format.value, sample, dataset_name)
        
        classification_prompt = tg.Variable(
            f"Classify this {dataset_name} sample:\n{formatted_input}\n\nClass:",
            role_description="Classification prompt"
        )
        
        generator_output = generator_model(classification_prompt)
        prediction = extract_prediction(generator_output.value, dataset_name)
        
        if prediction == true_label:
            correct += 1
    
    accuracy = correct / total
    if verbose:
        print(f"{dataset_type.title()} accuracy: {accuracy:.1%} ({correct}/{total})")
    
    return accuracy


def train_epoch(
    epoch: int,
    batch_size: int,
    train_data: List[Tuple[dict, str]],
    serialization_format: tg.Variable,
    generator_model,
    evaluator_loss_fn,
    optimizer=None,
    seen_formats=None,
    logger=None,
    provider="unknown",
    dataset_name: str = "iris"
) -> Dict:
    epoch_feedbacks = []

    if seen_formats is None:
        seen_formats = set()

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

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(shuffled_train))
        batch_data = shuffled_train[batch_start:batch_end]
        batch_feedbacks = []

        print(f"\n=== Batch {batch_idx + 1}/{num_batches} ===")
        print(f"Current format: {serialization_format.value}")

        format_before_batch = serialization_format.value
        total_loss = None
        batch_correct = 0

        for i, (sample, true_label) in enumerate(batch_data):
            formatted_input = safe_format(serialization_format.value, sample, dataset_name)
            print(f"\nSample {i + 1}/{len(batch_data)}:")
            print(f"  Input: {formatted_input}")

            format_before_sample = serialization_format.value

            classification_prompt = tg.Variable(
                f"Classify this {dataset_name} sample:\n{formatted_input}\n\nClass:",
                role_description="Classification prompt",
                predecessors=[serialization_format]
            )

            print(f"  Calling generator...")
            generator_output = generator_model(classification_prompt)
            prediction = extract_prediction(generator_output.value, dataset_name)

            if prediction == true_label:
                batch_correct += 1
                epoch_correct += 1
            epoch_total += 1

            print(f"  Prediction: {prediction} | Actual: {true_label} | {'CORRECT' if prediction == true_label else 'INCORRECT'}")

            dynamic_context = create_evaluator_context(
                serialization_format.value,
                sample,
                prediction,
                true_label,
                formatted_input, 
                dataset_name
            )

            evaluation_input = tg.Variable(
                f"{generator_output.value}\n\nContext:\n{dynamic_context}",
                role_description="Combined evaluation input",
                predecessors=[generator_output]
            )

            print(f"  Calling evaluator for feedback...")
            loss = evaluator_loss_fn(evaluation_input)

            raw_feedback = loss.value
            print(f"  Evaluator raw output:\n{loss.value}")

            match = re.search(r"<FEEDBACK>(.*?)</FEEDBACK>", raw_feedback, re.DOTALL)
            extracted_feedback = match.group(1).strip() if match else "NO FEEDBACK FOUND"

            # debugging and noticed that not everything I wanted was being logged - Don't skip logging, just mark for skipping optimizer
            skip_optimizer = False
            normalized_feedback = extracted_feedback.lower().strip()
            skip_keywords = ["keep", "no change", "same format", "unchanged", "do not modify"]
            if any(k in normalized_feedback for k in skip_keywords):
                print("  Detected redundant/no-change feedback - will skip optimizer step but still log.")
                skip_optimizer = True

            print(f"  Evaluator feedback: {extracted_feedback}")

            feedback_log = {
                "epoch": epoch,
                "batch": batch_idx,
                "sample": i,
                "formatted_input": formatted_input,
                "prediction": prediction,
                "true_label": true_label,
                "serialization_format_before": format_before_sample,
                "raw_feedback": raw_feedback,
                "extracted_feedback": extracted_feedback,
                "correct": prediction == true_label,
                "feedback_skipped": skip_optimizer
            }

            epoch_feedbacks.append(feedback_log)
            batch_feedbacks.append(feedback_log)

            # only accumulate loss if not skipping and prediction is wrong - this is a flaw I am getting rid of this
            
            total_loss = loss if total_loss is None else tg.sum([total_loss, loss])
            
            print(f"  Loss accumulated: {'No (skipped or correct)' if skip_optimizer or prediction == true_label else 'Yes'}")

        batch_accuracy = batch_correct / len(batch_data)
        print(f"\nBatch {batch_idx + 1} Summary:")
        print(f"  Batch accuracy: {batch_accuracy:.1%} ({batch_correct}/{len(batch_data)} correct)")

        # FIXED VERSION - Move logging outside optimizer block
        current_format_after_batch = serialization_format.value  # Capture current format

        # Handle optimizer step if needed
        if total_loss and optimizer:
            print("Backpropagating combined loss...")
            total_loss.backward()
            try:
                print("Optimizer stepping...")
                optimizer.step()
                print("  Format update attempted!")

                if serialization_format.value in seen_formats:
                    print("   WARNING: This format has been seen before, possible cycling detected.")
                else:
                    seen_formats.add(serialization_format.value)

                    format_changed = track_format_changes(
                        format_before_batch, serialization_format.value, epoch, batch_idx
                    )

                    if format_changed:
                        is_valid, validation_result = validate_format(serialization_format.value, sample)
                        if not is_valid:
                            print(f"   WARNING: New format failed validation: {validation_result}")
                        else:
                            print(f"   New format validated successfully: {validation_result}")

                # Update the format after optimizer step
                current_format_after_batch = serialization_format.value

            except Exception as e:
                print(f"  Optimizer failed: {e}")
                if hasattr(optimizer, "last_response"):
                    print(f"    Last response: {optimizer.last_response}")

        # ALWAYS LOG - regardless of whether optimizer stepped
        print(f"  Logging {len(batch_feedbacks)} samples from this batch...")
        for log in batch_feedbacks:
            log["serialization_format_after"] = current_format_after_batch
            
            if logger:
                logger.log_step(
                    log["epoch"], log["batch"], log["sample"], 
                    log["formatted_input"], log["prediction"], log["true_label"],
                    log["serialization_format_before"], log["raw_feedback"], 
                    current_format_after_batch, log["correct"]
                )
                print(f"    Logged sample {log['sample']}: {log['prediction']} ({'✓' if log['correct'] else '✗'})")

    # Return statement at proper function level
    epoch_train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
    return {
        "epoch": epoch,
        "train_accuracy": epoch_train_acc,
        "feedbacks": epoch_feedbacks
    }


def train_with_validation(
    train_data: List[Tuple[dict, str]],
    val_data: List[Tuple[dict, str]],
    test_data: List[Tuple[dict, str]],
    serialization_format: tg.Variable,
    generator_model,
    evaluator_loss_fn,
    num_epochs: int = 5,
    batch_size: int = 3,
    seen_formats=None,
    provider="unknown",
    dataset_name: str = "iris"
):

    logger = SimpleLogger()

    if seen_formats is None:
        seen_formats = set()

    best_format = serialization_format.value
    best_val_accuracy = 0
    no_improve_epochs = 0
    early_stop_patience = 3
    all_feedback_logs = []
    all_results = {
        "epochs": [],
        "best_format": None,
        "best_val_accuracy": 0,
        "final_test_accuracy": 0, 
        "accuracy_history": [] 
    }

    print("="*70)
    print("TRAINING STARTED")
    print("="*70)
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Provider: {provider}")
    print(f"  - Total epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Test samples: {len(test_data)}")
    print(f"\nInitial serialization format:\n  {serialization_format.value}")
    print("="*70)

    # get dataset-specific constraints for optimizer
    config = get_dataset_config(dataset_name)
    feature_placeholders = "{" + "}, {".join(config["features"]) + "}"
    
    optimizer = tg.TextualGradientDescent(
        parameters=[serialization_format],
        constraints=[
            f"Your job is to generate a new Python .format() string for serializing a {dataset_name} sample.",
            "Use the suggestion provided in the <FEEDBACK> tag to guide your rewrite.",
            f"Always return a single-line human-readable string that uses the placeholders: {feature_placeholders}.",
            "Focus on improving class discriminability and clarity.",
            "The output format MUST be compatible with Python's .format() method.",
            "DO NOT include class labels, predictions, or actuals in the format.",
            "NEVER use double braces {{ or }} — use only single braces.",
            "Avoid repeating the previous format unless it's explicitly requested.",
            f"Example valid format: '{config['default_format']}'"
        ],
        new_variable_tags=["<FEEDBACK>", "</FEEDBACK>"]
    )

    for epoch in range(num_epochs):
        epoch_start_time = pd.Timestamp.now()

        epoch_results = train_epoch(
            epoch=epoch,
            batch_size=batch_size,
            train_data=train_data,
            serialization_format=serialization_format,
            generator_model=generator_model,
            evaluator_loss_fn=evaluator_loss_fn,
            optimizer=optimizer,
            seen_formats=seen_formats,
            logger=logger,
            provider=provider,
            dataset_name=dataset_name
        )

        if "feedbacks" in epoch_results:
            all_feedback_logs.extend(epoch_results["feedbacks"])

        # Evaluate on validation set
        val_accuracy = evaluate_dataset(
            val_data, "validation", verbose=True, 
            serialization_format=serialization_format,
            generator_model=generator_model,
            dataset_name=dataset_name
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improve_epochs = 0
            best_format = serialization_format.value
            print("\n NEW BEST VALIDATION ACCURACY!")
            print(f"   Previous best: {all_results['best_val_accuracy']:.1%}")
            print(f"   New best: {best_val_accuracy:.1%}")
        else:
            no_improve_epochs += 1
            print(f"  No improvement in {no_improve_epochs} epochs")
            if no_improve_epochs >= early_stop_patience:
                print(f"\nEARLY STOPPING: No improvement for {early_stop_patience} consecutive epochs.")
                break

        epoch_results["val_accuracy"] = val_accuracy
        accuracy_record = {
            "epoch": epoch + 1,
            "train_accuracy": epoch_results["train_accuracy"],
            "val_accuracy": val_accuracy,
            "current_format": serialization_format.value
        }
        all_results["accuracy_history"].append(accuracy_record)
        all_results["epochs"].append(epoch_results)

        print(f"\nEPOCH {epoch + 1} SUMMARY")
        print(f"  Train accuracy: {epoch_results['train_accuracy']:.1%}")
        print(f"  Val accuracy: {val_accuracy:.1%}")
        print(f"  Best val accuracy so far: {best_val_accuracy:.1%}")
        print(f"  Current format: {serialization_format.value}")
        print(f"  Epoch duration: {(pd.Timestamp.now() - epoch_start_time).total_seconds():.1f} seconds")

    print(f"\n{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}")
    serialization_format.set_value(best_format)
    print(f"Restored best format: {best_format}")

    # evaluate on test set
    test_accuracy = evaluate_dataset(
        test_data, "test", verbose=True,
        serialization_format=serialization_format,
        generator_model=generator_model,
        dataset_name=dataset_name
    )

    all_results["best_format"] = best_format
    all_results["best_val_accuracy"] = best_val_accuracy
    all_results["final_test_accuracy"] = test_accuracy

    # save the existing logs
    with open("textgrad_feedback_logs.json", "w") as f:
        json.dump(all_feedback_logs, f, indent=2)

    # save the SimpleLogger logs
    logger.save(provider)

    print(f"\nTraining complete. Best format: {best_format}")
    print(f"Validation Accuracy: {best_val_accuracy:.1%}")
    print(f"Test Accuracy: {test_accuracy:.1%}")

    with open("textgrad_accuracy_history.json", "w") as f:
        json.dump(all_results["accuracy_history"], f, indent=2)


    with open("textgrad_training_summary.json", "w") as f:
        json.dump({
            "best_format": all_results["best_format"],
            "best_val_accuracy": all_results["best_val_accuracy"],
            "final_test_accuracy": all_results["final_test_accuracy"],
            "total_epochs": len(all_results["epochs"]),
            "dataset_name": dataset_name,
            "provider": provider
        }, f, indent=2)

    return all_results