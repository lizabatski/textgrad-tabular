import textgrad as tg
from .dataset_configs import get_dataset_config

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags. "
    "The text you send between the tags will directly replace the variable."
)

def init_models(generator_engine, evaluator_engine, optimizer_engine, dataset_name: str = "iris", optimize_mode: str = "format"):    
    
    config = get_dataset_config(dataset_name)
    
    tg.set_backward_engine(optimizer_engine, override=True)
    

    serialization_format = tg.Variable(
        config["default_format"],
        requires_grad=(optimize_mode in ["format", "both"]),
        role_description=f"Serialization format for {dataset_name} features"
    )
    
    generator_system_prompt = tg.Variable(
        config["system_prompt"],
        requires_grad=(optimize_mode in ["prompt", "both"]),
        role_description=f"System prompt used by the generator model to classify {dataset_name} data"
    )
    

    generator_model = tg.BlackboxLLM(generator_engine, system_prompt=generator_system_prompt)
    
    
    evaluator_prompt = tg.Variable(
        config["evaluator_prompt"],
        requires_grad=False,
        role_description=f"Evaluator system prompt for giving natural language feedback on {dataset_name} classification"
    )
    
    # create evaluator loss function using MultiFieldTokenParsedEvaluation
    # Use TextLoss instead
    evaluator_loss_fn = tg.TextLoss(evaluator_prompt, engine=evaluator_engine)
    
    # create optimizer with dataset-specific constraints
    optimizer_params = []
    optimizer_constraints = []
    
    # Create mode-specific system prompts and constraints
    if optimize_mode == "format":
        optimizer_params = [serialization_format]
        feature_placeholders = "{" + "}, {".join(config["features"]) + "}"
        optimizer_constraints = [
               f"You are updating a Python .format() string for {dataset_name} data serialization.",
                "You will receive feedback about the current format.",
                "Your task is to generate an IMPROVED format string wrapped in <IMPROVED_VARIABLE> tags.",
                f"Use these exact placeholders: {feature_placeholders}",
                "Example response format:",
                f"<IMPROVED_VARIABLE>{config['default_format']}</IMPROVED_VARIABLE>",
                "DO NOT return feedback or explanations - only the improved format string in the tags."
        ]
        
        custom_system_prompt = """You are a Python format string optimizer. 

        You receive:
        1. A current format string for data serialization
        2. Feedback about problems with that format

        Your task: Generate an improved format string that addresses the feedback.

        You must respond with ONLY the improved format string wrapped in <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags.

        Example response:
        <IMPROVED_VARIABLE>Petal Length: {{petal_length:.2f}}cm, Petal Width: {{petal_width:.2f}}cm</IMPROVED_VARIABLE>

        Do not explain your reasoning. Do not repeat the feedback. Just provide the improved format string."""
        
    elif optimize_mode == "prompt":
        optimizer_params = [generator_system_prompt]
        optimizer_constraints = [
            f"You are improving a system prompt for {dataset_name} classification.",
            "You must use the <FEEDBACK> suggestions as your primary guide.", 
            "Your task is to generate an IMPROVED system prompt wrapped in <IMPROVED_VARIABLE> tags.",
            "Focus on making the classification instructions clearer and more effective.",
            "Maintain the core task of classifying into the correct species.",
            "Include clear decision rules and feature importance guidance.",
            f"Example response format:",
            f"<IMPROVED_VARIABLE>You are an expert botanist...</IMPROVED_VARIABLE>",
            "DO NOT return feedback or explanations - only the improved prompt in the tags."
        ]
        
        custom_system_prompt = """You are a system prompt optimizer for classification models.

        You receive:
        1. A current system prompt for a classifier
        2. Feedback about classification performance

        Your task: Generate an improved system prompt that addresses the feedback.

        You must respond with ONLY the improved system prompt wrapped in {new_variable_start_tag} and {new_variable_end_tag} tags.

        Focus on:
        - Clearer classification instructions
        - Better feature importance guidance  
        - More precise decision rules
        - Improved clarity and specificity

        Do not explain your reasoning. Just provide the improved prompt."""
        
    elif optimize_mode == "both":
        optimizer_params = [serialization_format, generator_system_prompt]
        feature_placeholders = "{" + "}, {".join(config["features"]) + "}"
        optimizer_constraints = [
            "You are optimizing both the data format AND the system prompt.",
            "You must use the <FEEDBACK> suggestions as your primary guide.",
            "For the format: focus on clarity and discriminability.",
            "For the prompt: focus on better classification instructions.",
            f"Dataset: {dataset_name} with features: {', '.join(config['features'])}.",
            "Return both improved format and prompt wrapped in <IMPROVED_VARIABLE> tags."
        ]
        
        custom_system_prompt = """You are optimizing both format strings and system prompts for classification.

        You receive feedback about both the data format and classification prompt.

        Your task: Generate improved versions that address the feedback.

        You must respond with ONLY the improved content wrapped in {new_variable_start_tag} and {new_variable_end_tag} tags.

        Do not explain your reasoning. Just provide the improvements."""

    
    optimizer = tg.TextualGradientDescent(
        #parameters=optimizer_params,
        parameters=list(generator_model.parameters()),
        constraints=optimizer_constraints,
        new_variable_tags=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]
    )
    
    print(f"Initialized with optimization mode: {optimize_mode}")
    
    return serialization_format, generator_model, evaluator_loss_fn, optimizer