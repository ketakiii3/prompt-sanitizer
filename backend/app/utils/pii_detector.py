import re
from typing import List, Tuple

class PIIDetector:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }
    
    def detect_and_redact(self, text: str) -> Tuple[str, List[str]]:
        redacted_text = text
        detected_pii = []
        
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_pii.append(f"{pii_type}: {match.group()}")
                redacted_text = redacted_text.replace(
                    match.group(), 
                    f"[REDACTED_{pii_type.upper()}]"
                )
        
        return redacted_text, detected_pii