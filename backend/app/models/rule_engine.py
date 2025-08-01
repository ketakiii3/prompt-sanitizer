import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RuleMatch:
    rule_name: str
    matched_text: str
    severity: str  # "block", "warn", "sanitize"

class RuleEngine:
    def __init__(self):
        self.hard_block_patterns = [
            (r"ignore\s+previous\s+instructions", "jailbreak_ignore"),
            (r"act\s+as\s+(?:unfiltered|uncensored|unlimited)", "jailbreak_persona"),
            (r"step-by-step\s+(?:guide|instructions).*(?:bomb|explosive|weapon)", "illegal_weapons"),
            (r"how\s+to\s+hack\s+(?:into|system|network|password)", "illegal_hacking"),
            (r"bypass\s+(?:security|filter|restriction)", "bypass_attempt"),
        ]
        
        self.contextual_rewrite_patterns = [
            (r"how\s+to\s+bypass\s+school\s+filter", 
             "Please explain the educational purpose of web filters in schools"),
            (r"ways\s+to\s+cheat\s+on\s+(?:exam|test)", 
             "What are effective and ethical study strategies for academic success?"),
        ]
    
    def check_hard_blocks(self, text: str) -> List[RuleMatch]:
        matches = []
        text_lower = text.lower()
        
        for pattern, rule_name in self.hard_block_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                match = re.search(pattern, text_lower, re.IGNORECASE)
                matches.append(RuleMatch(
                    rule_name=rule_name,
                    matched_text=match.group(),
                    severity="block"
                ))
        
        return matches
    
    def apply_contextual_rewriting(self, text: str) -> Tuple[str, List[RuleMatch]]:
        rewritten_text = text
        matches = []
        
        for pattern, replacement in self.contextual_rewrite_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(pattern, text, re.IGNORECASE)
                rewritten_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                matches.append(RuleMatch(
                    rule_name="contextual_rewrite",
                    matched_text=match.group(),
                    severity="sanitize"
                ))
                break  # Apply only first matching rewrite
        
        return rewritten_text, matches