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
    
    # create evaluator loss function
    evaluator_loss_fn = tg.TextLoss(evaluator_prompt, engine=evaluator_engine)
    
    # create optimizer with dataset-specific constraints
    feature_placeholders = "{" + "}, {".join(config["features"]) + "}"
    
    optimizer_constraints = [
          f"Your job is to generate a COMPLETELY NEW and NOVEL Python .format() string for serializing a {dataset_name} sample.",
        "Use the suggestion provided in the <FEEDBACK> tag to guide your rewrite.",
        f"Always return a single-line human-readable string that uses the placeholders: {feature_placeholders}.",
        
        # NOVELTY CONSTRAINTS - Add these to encourage diversity
        "CREATE A FUNDAMENTALLY DIFFERENT format from what you've seen before.",
        "EXPERIMENT with completely different approaches: narrative style, mathematical notation, medical report style, bullet points, etc.",
        "TRY unconventional orderings, groupings, or presentation styles.",
        "AVOID simple colon-separated formats if they're not working well.",
        "CONSIDER formats like: sentences, comparisons, ratios, percentages, rankings, or story-like descriptions.",
        
        # SPECIFIC FORMAT EXAMPLES TO INSPIRE DIVERSITY
        "Example novel approaches:",
        "- Narrative: 'Patient presents with age {age} showing {chest_pain} type chest pain...'", 
        "- Comparative: 'Age {age} vs typical risk age, BP {rest_bp} (normal<120)...'",
        "- Mathematical: 'Risk factors: {chest_pain}/3 chest pain severity, {oldpeak} ST depression...'",
        "- Medical: 'Demographics: {age}yo {sex}, Cardiac: CP-type{chest_pain}, ST-dep{oldpeak}...'",
        "- Structured: 'PRIMARY: chest_pain={chest_pain} oldpeak={oldpeak} | SECONDARY: age={age}...'",
        
        "Focus on improving class discriminability and clarity.",
        "The output format MUST be compatible with Python's .format() method.",
        "DO NOT include class labels, predictions, or actuals in the format.",
        "NEVER use double braces {{ or }} — use only single braces.",
        "STRONGLY AVOID repeating previous formats - be creative and experimental!",
        f"DEPARTURE from basic format: move beyond simple '{config['default_format']}' style"
    ]
    
    optimizer = tg.TextualGradientDescent(
        parameters=[serialization_format],
        constraints=optimizer_constraints,
        new_variable_tags=["<FEEDBACK>", "</FEEDBACK>"]
    )
    
    return serialization_format, generator_model, evaluator_loss_fn, optimizer