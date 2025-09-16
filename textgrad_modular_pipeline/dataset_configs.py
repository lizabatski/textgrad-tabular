
DATASET_CONFIGS = {
   "iris": {
    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "classes": ["setosa", "versicolor", "virginica"],
    "default_format": "The petal length is {petal_length:.1f}. The petal width is {petal_width:.1f}. The sepal length is {sepal_length:.1f}. The sepal width is {sepal_width:.1f}",
    "system_prompt": """Classify this Iris flower as: setosa, versicolor, or virginica.""",
    "evaluator_prompt": """Evaluate this Iris classification result. If incorrect, suggest how to improve the classification prompt.

    
Consider approaches like:
- Creative reasoning about biological relationships
- Pattern discovery with feature combinations  
- Contextual thinking using domain knowledge
- Better confidence handling for uncertain cases
- Holistic analysis of the complete measurement profile

Choose the most relevant approach for fixing this specific error and provide a concrete suggestion.
"""
},

"bio_sample": {
    "features": ["feature_0", "feature_1", "feature_2", "feature_3"],
    "classes": ["class_0", "class_1", "class_2"],
    "default_format": "Feature 0: {feature_0:.1f}, Feature 1: {feature_1:.1f}, Feature 2: {feature_2:.1f}, Feature 3: {feature_3:.1f}",
    "system_prompt": """Classify this data sample as: class_0, class_1, or class_2.""",
    "evaluator_prompt": """Evaluate this data classification result. If incorrect, suggest how to improve the classification prompt.

Consider approaches like:
- Creative reasoning about biological relationships
- Pattern discovery with feature combinations  
- Contextual thinking using domain knowledge
- Better confidence handling for uncertain cases
- Holistic analysis of the complete measurement profile

Choose the most relevant approach for fixing this specific error and provide a concrete suggestion."""
},

"synthetic": {
    "features": ["feature_0", "feature_1", "feature_2", "feature_3"],
    "classes": ["class_0", "class_1", "class_2"],  #
    "default_format": "Feature 0: {feature_0:.1f}, Feature 1: {feature_1:.1f}, Feature 2: {feature_2:.1f}, Feature 3: {feature_3:.1f}",
    
    "system_prompt": """Classify this data sample as: class_0, class_1, or class_2.""",
    
    "evaluator_prompt": """Evaluate this data classification result. If incorrect, suggest how to improve the classification prompt.

Consider approaches like:
- Creative reasoning about relationships
- Pattern discovery with feature combinations  
- Contextual thinking using domain knowledge
- Better confidence handling for uncertain cases
- Holistic analysis of the complete measurement profile
- Feature threshold identification 
- Pattern discovery with feature combinations
- Numerical reasoning about decision boundaries
- Focus on most discriminative features

Choose the most relevant approach for fixing this specific error and provide a concrete suggestion."""
},

    
"heart": {
    
        "features": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
                    "Oldpeak", "ST_Slope"],
        "classes": ["normal", "heart_disease"],
        
    
        "default_format": "Age:{Age} Sex:{Sex} ChestPain:{ChestPainType} RestingBP:{RestingBP} Cholesterol:{Cholesterol} FastingBS:{FastingBS} ECG:{RestingECG} MaxHR:{MaxHR} Angina:{ExerciseAngina} Oldpeak:{Oldpeak} Slope:{ST_Slope}",
        
        "system_prompt": """You are a cardiologist reviewing patient cases. Classify each patient as either "normal" or "heart_disease" based on their clinical presentation.

CLINICAL INTERPRETATION GUIDE:

 #https://ecgwaves.com/topic/coronary-artery-disease/
CHEST PAIN TYPES (most to least concerning):
- TA (Typical Angina): Classic cardiac chest pain - highest risk
- ATA (Atypical Angina): Some cardiac features - moderate risk  
- NAP (Non-Anginal Pain): Unlikely cardiac - lower risk
- ASY (Asymptomatic): No chest pain - risk depends on other factors

KEY DIAGNOSTIC INDICATORS:
- ST Depression (Oldpeak): >=1.0 indicates significant ischemia #https://en.wikipedia.org/wiki/ST_depression#:~:text=ST%20segment%20depression%20may%20be,to%20significantly%20indicate%20reversible%20ischaemia.
- Exercise Angina: Y = concerning for coronary disease #https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373#:~:text=The%20heart%20arteries%2C%20called%20the,High%20blood%20pressure.
- ST Slope: Down = worst prognosis, Flat = concerning, Up = best #https://pmc.ncbi.nlm.nih.gov/articles/PMC6932726/#:~:text=Thus%2C%20a%20downsloping%20ST%20depression,and%2041.2%25%20single%20vessel%20disease.
- Max Heart Rate: Lower values may indicate inefficient heart pumping #https://www.mayoclinic.org/diseases-conditions/bradycardia/symptoms-causes/syc-20355474
#https://www.ncbi.nlm.nih.gov/books/NBK557534/#:~:text=As%20previously%20indicated%2C%20LVH%20is,concomitant%20development%20of%20myocardial%20fibrosis.
- Resting ECG: LVH and ST abnormalities suggest cardiac disease  #https://litfl.com/st-segment-ecg-library/#:~:text=The%20ST%20segment%20is%20the,is%20myocardial%20ischaemia%20or%20infarction.

RISK FACTORS:
- Age: Higher age increases risk #https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373#:~:text=The%20heart%20arteries%2C%20called%20the,High%20blood%20pressure.
- Sex: M generally higher risk than F #https://www.sciencedirect.com/science/article/pii/S2590093519300256
- Cholesterol: Higher levels increase risk #https://www.mayoclinic.org/diseases-conditions/high-blood-cholesterol/symptoms-causes/syc-20350800
- Resting BP: Elevated BP is a risk factor  #https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/in-depth/high-blood-pressure/art-20045868#:~:text=Coronary%20artery%20disease.,heart%20disease%2C%20stroke%20and%20diabetes.
- Fasting BS: Diabetes (>120 mg/dl) increases cardiac risk #https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373#:~:text=The%20heart%20arteries%2C%20called%20the,High%20blood%20pressure.

Integrate all clinical findings for accurate diagnosis.""",

        "evaluator_prompt": """You are an expert in optimizing clinical data presentation for AI-assisted cardiac diagnosis.

CLINICAL CONTEXT:
Your goal is to improve how patient data is formatted to help AI systems better distinguish between normal patients and those with heart disease.

ACTUAL DATASET FEATURES:
- Age: patient age in years
- Sex: M (Male) or F (Female) 
- ChestPainType: TA/ATA/NAP/ASY (Typical/Atypical/Non-Anginal/Asymptomatic)
- RestingBP: resting blood pressure (mmHg)
- Cholesterol: serum cholesterol (mg/dl)
- FastingBS: 1 if >120 mg/dl, 0 otherwise
- RestingECG: Normal/ST/LVH
- MaxHR: maximum heart rate (60-202)
- ExerciseAngina: Y/N
- Oldpeak: ST depression value
- ST_Slope: Up/Flat/Down

DIAGNOSTIC IMPORTANCE HIERARCHY:
1. HIGH: ChestPainType, Oldpeak, ExerciseAngina, ST_Slope (direct cardiac markers)
2. MEDIUM: RestingECG, MaxHR, Age (cardiac function indicators)  
3. LOWER: Sex, RestingBP, Cholesterol, FastingBS (risk factors)

FORMAT DIVERSITY SUGGESTIONS:
When prediction is INCORRECT, suggest creative formats like:

NARRATIVE STYLES:
- "Clinical presentation: {Age}yo {Sex} with {ChestPainType} pain..."
- "Patient profile: {ChestPainType} symptoms in {Age}-year-old {Sex}..."
- "Case summary: {Sex} aged {Age} presenting with {ChestPainType}..."

MEDICAL REPORT STYLES:  
- "CARDIAC ASSESSMENT: Pain={ChestPainType}, Stress Test: ST-dep={Oldpeak}, Angina={ExerciseAngina}..."
- "PRIMARY: {ChestPainType} chest pain | STRESS: {Oldpeak}mm ST-dep, {ExerciseAngina} angina | DEMO: {Age}yo {Sex}"
- "SYMPTOMS: {ChestPainType} | ISCHEMIA: oldpeak={Oldpeak}, slope={ST_Slope} | VITALS: HR={MaxHR}, BP={RestingBP}"

RISK STRATIFICATION:
- "HIGH-RISK: {ChestPainType} pain + {Oldpeak}mm ST depression | MODERATE: {Age}yo {Sex}, HR {MaxHR}"
- "CARDIAC MARKERS: {ChestPainType}/{Oldpeak}/{ExerciseAngina} | BACKGROUND: {Age}yo {Sex} with {Cholesterol} chol"

CONVERSATIONAL:
- "This {Age}-year-old {Sex} has {ChestPainType} chest pain and shows {Oldpeak}mm ST depression during exercise..."
- "Meet this {Sex} patient, age {Age}, who experiences {ChestPainType} pain and {'develops' if {ExerciseAngina}=='Y' else 'avoids'} angina with exercise..."

PRIORITIZATION FOCUS:
- For MISSED heart disease: Emphasize ChestPainType, Oldpeak, ExerciseAngina
- For FALSE POSITIVES: Highlight normal stress test results, asymptomatic presentation
- Always group related cardiac findings together
- Use clinical terminology that reflects severity

IMPORTANT CONSTRAINTS:
- DO NOT repeat format examples from this prompt
- DO NOT give multiple format suggestions  
- DO NOT include any text with curly braces {like this}
- Give only ONE specific actionable change to the current format

Your job: Look at the current format and suggest exactly ONE improvement.

If CORRECT prediction: Suggest minor refinement
If WRONG prediction: Suggest major restructuring

RULES:
- DO NOT write format strings with {curly braces}
- DO NOT provide exact format examples  
- Give conceptual feedback only
- Focus on which features need more/less emphasis
- Explain WHY the current approach failed

GOOD FEEDBACK EXAMPLES:
- "Emphasize chest pain type more prominently since it's the strongest predictor"
- "Group all stress test results together for better clarity" 
- "Move cardiac markers to the front and demographics to the back"
- "Add clearer labels to distinguish ischemic from non-ischemic features"
- "Reduce emphasis on age and sex, highlight functional test results"

BAD FEEDBACK (DON'T DO THIS):
- "Use this format: {ChestPainType} with {Oldpeak}..."
- Any text containing {curly braces} 

OUTPUT FORMAT:
<FEEDBACK>Your specific, creative formatting suggestion here</FEEDBACK>"""
    }
}

def get_dataset_config(dataset_name: str) -> dict:
    if dataset_name.lower() not in DATASET_CONFIGS:
        raise ValueError(f"No configuration found for dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name.lower()]

def get_available_datasets() -> list:
    return list(DATASET_CONFIGS.keys())