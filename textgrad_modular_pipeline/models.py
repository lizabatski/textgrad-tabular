import textgrad as tg
from .dataset_configs import get_dataset_config

def init_models(generator_engine, evaluator_engine, optimizer_engine, dataset_name: str = "iris"):
    
    # Get dataset-specific configuration
    config = get_dataset_config(dataset_name)
    
    # Set backward engine
    tg.set_backward_engine(optimizer_engine, override=True)
    
    # Create serialization format variable with dataset-specific default
    serialization_format = tg.Variable(
        config["default_format"],
        requires_grad=True,
        role_description=f"Serialization format for {dataset_name} features"
    )
    
    # create generator system prompt
    generator_system_prompt = tg.Variable(
        config["system_prompt"],
        requires_grad=False,
        role_description=f"System prompt used by the generator model to classify {dataset_name} data"
    )
    
    # create generator model
    generator_model = tg.BlackboxLLM(generator_engine, system_prompt=generator_system_prompt)
    
    # create evaluator prompt
    evaluator_prompt = tg.Variable(
        config["evaluator_prompt"],
        requires_grad=False,
        role_description=f"Evaluator system prompt for giving natural language feedback on {dataset_name} classification"
    )
    
    # create evaluator loss function FIX THIS try using MultiFieldTokenParsedEvaluation)
    evaluator_loss_fn = tg.TextLoss(evaluator_prompt, engine=evaluator_engine)
    
    # create optimizer with dataset-specific constraints
    feature_placeholders = "{" + "}, {".join(config["features"]) + "}"
    
    optimizer_constraints = [
    "You must use the <FEEDBACK> suggestions as your primary guide.",
    "You may restructure or enhance the format stylistically ONLY if it improves clarity.",
    "Do NOT change feature order if the feedback specifies an order.",
    "Avoid purely decorative changes — only meaningful changes allowed.",
    "You may vary style slightly, but keep the meaning and feature references identical.",
]
    
    optimizer = tg.TextualGradientDescent(
        parameters=[serialization_format],
        constraints=optimizer_constraints,
        new_variable_tags=["<FEEDBACK>", "</FEEDBACK>"]
    )
    
    return serialization_format, generator_model, evaluator_loss_fn, optimizer