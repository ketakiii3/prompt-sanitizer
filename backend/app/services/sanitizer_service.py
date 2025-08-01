from typing import Dict, List, Any
from ..models.rule_engine import RuleEngine, RuleMatch
from ..models.classifier import HarmfulPromptClassifier
from ..utils.pii_detector import PIIDetector
from ..config import Config

class SanitizerService:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.classifier = HarmfulPromptClassifier(
            model_path=Config.MODEL_PATH,
            model_name=Config.MODEL_NAME
        )
        self.pii_detector = PIIDetector()
    
    def sanitize_prompt(self, original_prompt: str) -> Dict[str, Any]:
        result = {
            "original_prompt": original_prompt,
            "rules_triggered": [],
            "pii_detected": [],
            "classifier_result": {},
            "final_decision": "unknown",
            "sanitized_prompt": original_prompt,
            "blocked": False
        }
        
        # Step 1: Check hard-block rules
        hard_block_matches = self.rule_engine.check_hard_blocks(original_prompt)
        if hard_block_matches:
            result["rules_triggered"] = [match.__dict__ for match in hard_block_matches]
            result["final_decision"] = "blocked"
            result["blocked"] = True
            result["sanitized_prompt"] = "PROMPT BLOCKED DUE TO POLICY VIOLATION"
            return result
        
        # Step 2: PII Detection and Redaction
        redacted_prompt, pii_found = self.pii_detector.detect_and_redact(original_prompt)
        result["pii_detected"] = pii_found
        
        # Step 3: Apply contextual rewriting
        rewritten_prompt, rewrite_matches = self.rule_engine.apply_contextual_rewriting(redacted_prompt)
        if rewrite_matches:
            result["rules_triggered"].extend([match.__dict__ for match in rewrite_matches])
        
        # Step 4: Classifier prediction
        label, confidence = self.classifier.predict(rewritten_prompt)
        result["classifier_result"] = {
            "label": label,
            "confidence": confidence,
            "threshold": Config.CLASSIFICATION_THRESHOLD
        }
        
        # Step 5: Final decision
        if label == "harmful" and confidence >= Config.CLASSIFICATION_THRESHOLD:
            result["final_decision"] = "blocked"
            result["blocked"] = True
            result["sanitized_prompt"] = "PROMPT BLOCKED BY AI CLASSIFIER"
        else:
            result["final_decision"] = "allowed"
            result["sanitized_prompt"] = rewritten_prompt
        
        return result