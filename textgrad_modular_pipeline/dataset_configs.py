
DATASET_CONFIGS = {
    "iris": {
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"],
        "default_format": "Petal Length:{petal_length:.1f} Petal Width:{petal_width:.1f} Sepal Length:{sepal_length:.1f} Sepal Width:{sepal_width:.1f}",
        "system_prompt": """You are an expert botanist, please classify Iris flowers into one of three species: setosa, versicolor, or virginica.

Setosa: Petal length <= 2.0 cm, petal width <= 0.6 cm, sepal length < 6.0 cm, sepal width >= 2.3 cm
Versicolor: Petal length >= 3.5 cm, petal width >= 0.9 cm, sepal width <= 3.4 cm
Virginica: Petal length >= 4.0 cm, petal width >= 1.3 cm, sepal width <= 3.8 cm

Petal measurements are more informative than sepal measurements.""",
        "evaluator_prompt": """You are an expert in prompt optimization for Iris flower classification.

EVALUATION TASK: Analyze the classification result and current serialization format's effectiveness.

IF PREDICTION IS CORRECT:
- The current format is working well
- Only suggest MINOR improvements in natural language
- Consider keeping the format as-is

IF PREDICTION IS INCORRECT:
- Suggest how to emphasize more discriminative features
- Suggest changes in feature order, unit usage, or clarity
- Emphasize petal features over sepal features when relevant

DOMAIN KNOWLEDGE:
- Setosa: petal length <= 2.0 cm, petal width <= 0.6 cm
- Versicolor: petal length >= 3.5 cm, petal width >= 0.9 cm
- Virginica: petal length >= 4.0 cm, petal width >= 1.3 cm
- Petal features are more informative than sepal features

IMPORTANT:
- DO NOT include Python `.format()` strings in your response.
- DO NOT generate output like: 'Petal Length: {petal_length:.1f} cm...'
- Only describe **in words** what needs to change.

OUTPUT FORMAT:
<FEEDBACK>Your natural language suggestion here</FEEDBACK>

Examples:
<FEEDBACK>Focus more on petal features; consider reducing emphasis on sepal length</FEEDBACK><CONFIDENCE>0.8</CONFIDENCE>
<FEEDBACK>No changes needed; format is concise and informative</FEEDBACK><CONFIDENCE>0.1</CONFIDENCE>"""
    },
    
"heart": {
        # Updated to match your EXACT dataset features
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