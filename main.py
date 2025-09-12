import argparse
from textgrad_modular_pipeline.config import get_engines
from textgrad_modular_pipeline.loaders import load_dataset
from textgrad_modular_pipeline.models import init_models
from textgrad_modular_pipeline.trainer import train_with_validation
from textgrad_modular_pipeline.dataset_configs import get_available_datasets

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description="TextGrad Training with Multiple Datasets")
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=get_available_datasets(),
                       help="Dataset to use for training")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to dataset file (optional)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=9,
                       help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--optimize-mode", type=str, default="prompt",
                       choices=["format", "prompt", "both"],
                       help="What to optimize: 'format' (serialization only), 'prompt' (generator prompt only), or 'both'")
    parser.add_argument("--job-id", type=str, default=None,
                   help="Unique job identifier for output files")
    
    args = parser.parse_args()

    print("="*70)
    print(f"TextGrad Prompt Optimization for {args.dataset.title()} Classification")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Optimization mode: {args.optimize_mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    if args.dataset_path:
        print(f"Custom dataset path: {args.dataset_path}")
    print("="*70)

    try:
        # Get engines (unchanged)
        generator_engine, evaluator_engine, optimizer_engine, provider = get_engines()
        print(f"Using provider: {provider}")

        # Load the specified dataset
        train_data, val_data, test_data = load_dataset(
            dataset_name=args.dataset, 
            dataset_path=args.dataset_path,
            seed=args.seed
        )

        # Initialize models with optimization mode
        serialization_format, generator_model, evaluator_loss_fn, optimizer = init_models(
            generator_engine, evaluator_engine, optimizer_engine, 
            args.dataset, 
            optimize_mode=args.optimize_mode
        )

        print(f"Initial format: {serialization_format.value}")

        # Train with dataset name passed to trainer
        results = train_with_validation(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            serialization_format=serialization_format,
            generator_model=generator_model,
            evaluator_loss_fn=evaluator_loss_fn,
            optimizer=optimizer,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seen_formats=set(),
            provider=provider,
            dataset_name=args.dataset,
            optimize_mode=args.optimize_mode,
            job_id=args.job_id
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Optimization mode: {args.optimize_mode}")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.1%}")
        print(f"Final test accuracy: {results['final_test_accuracy']:.1%}")
        print(f"Best prompt: {results['best_system_prompt']}")
        
        print("\nFiles saved:")
        print("- textgrad_feedback_logs.json (detailed feedback)")
        print("- textgrad_accuracy_history.json (epoch-by-epoch accuracy)")
        print("- textgrad_training_summary.json (final results)")
        print(f"- simple_logger_{provider}.json (SimpleLogger output)")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure your dataset file exists and has the correct format.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())