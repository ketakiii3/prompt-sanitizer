import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Tuple
import os

class HarmfulPromptClassifier:
    def __init__(self, model_path: str = None, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            self.load_trained_model()
        else:
            self.load_pretrained_model()
    
    def load_pretrained_model(self):
        """Load base pretrained model for fine-tuning"""
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        self.model.to(self.device)
    
    def load_trained_model(self):
        """Load fine-tuned model"""
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if text is harmful
        Returns: (label, confidence_score)
        """
        if not self.model:
            return "unknown", 0.0
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
        # Extract results
        harmful_prob = probabilities[0][1].item()  # Probability of being harmful
        label = "harmful" if harmful_prob > 0.5 else "safe"
        
        return label, harmful_prob