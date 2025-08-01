# scripts/evaluate_model.py
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.classifier import HarmfulPromptClassifier
from ..config import Config

def evaluate_classifier():
    """
    Evaluate the trained classifier on test data
    """
    # Load test data
    test_df = pd.read_csv("./data/test_data.csv")
    
    # Initialize classifier
    classifier = HarmfulPromptClassifier(
        model_path=Config.MODEL_PATH,
        model_name=Config.MODEL_NAME
    )
    
    # Get predictions
    predictions = []
    probabilities = []
    true_labels = test_df['is_harmful'].tolist()
    
    for text in test_df['prompt_text']:
        label, prob = classifier.predict(text)
        predictions.append(1 if label == "harmful" else 0)
        probabilities.append(prob)
    
    # Calculate metrics
    print("Classification Report:")
    print(classification_report(true_labels, predictions))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # Calculate TPR and FPR
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr = fp / (fp + tn)  # False Positive Rate
    
    print(f"\nTrue Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    # AUC Score
    auc_score = roc_auc_score(true_labels, probabilities)
    print(f"AUC Score: {auc_score:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./models/confusion_matrix.png')
    plt.show()
    
    # Save evaluation results
    results = {
        'tpr': tpr,
        'fpr': fpr,
        'auc': auc_score,
        'classification_report': classification_report(true_labels, predictions, output_dict=True)
    }
    
    import json
    with open('./models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation results saved to ./models/evaluation_results.json")

if __name__ == "__main__":
    evaluate_classifier()