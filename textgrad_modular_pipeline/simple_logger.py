import json
import re
from datetime import datetime

class SimpleLogger:
    def __init__(self):
        self.feedback_log = []
        
    def log_step(self, epoch, batch, sample, formatted_input, prediction, 
                 true_label, prompt_before, raw_feedback, prompt_after, correct):
        
        
        feedback_match = re.search(r'<FEEDBACK>(.*?)</FEEDBACK>', raw_feedback, re.DOTALL)
        extracted_feedback = feedback_match.group(1).strip() if feedback_match else raw_feedback
        
        entry = {
            "epoch": epoch,
            "batch": batch, 
            "sample": sample,
            "formatted_input": formatted_input,
            "prediction": prediction,
            "true_label": true_label,
            "system_prompt_before": prompt_before, 
            #"serialization_format_before": format_before,
            #"raw_feedback": raw_feedback,
            "extracted_feedback": extracted_feedback,
            "correct": correct,
            "system_prompt_after": prompt_after
            #"serialization_format_after": format_after
        }
        self.feedback_log.append(entry)
        
    def save(self, provider):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"textgrad_{provider}_feedback_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.feedback_log, f, indent=2)
        print(f"Log saved to: {filename}")