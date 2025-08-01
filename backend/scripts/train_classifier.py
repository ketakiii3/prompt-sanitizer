# backend/scripts/train_classifier.py

# Import necessary libraries
import pandas as pd  # Used for reading and manipulating our data, especially from CSV files.
import torch  # The main library for all things PyTorch (tensors, neural networks, etc.).
from torch.utils.data import Dataset, DataLoader  # Tools to create and manage datasets and load data in batches.
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW  # Hugging Face tools for the specific model we're using.
from sklearn.model_selection import train_test_split  # A utility to easily split our data into training and validation sets.
from sklearn.metrics import accuracy_score, classification_report  # Tools to evaluate our model's performance.
import numpy as np  # A library for numerical operations, used here for handling predictions.
from tqdm import tqdm  # A great library for creating smart progress bars to monitor long-running loops.


# This class defines how our dataset is structured and how individual data points are processed.
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        # This is the constructor for our dataset. It gets called when we create an instance of PromptDataset.
        self.texts = texts  # The list of all our prompt texts (e.g., "this is a comment").
        self.labels = labels  # The corresponding list of labels (0 for safe, 1 for harmful).
        self.tokenizer = tokenizer  # The tokenizer object we'll use to convert text into numbers.
        self.max_length = max_length  # The maximum length of a tokenized prompt. Prompts longer than this will be cut short.
    
    def __len__(self):
        # This method returns the total number of samples in the dataset.
        return len(self.texts)
    
    def __getitem__(self, idx):
        # This method gets a single data sample at a specific index 'idx'.
        text = str(self.texts[idx])  # Get the text for the given index.
        label = self.labels[idx]  # Get the label for the given index.
        
        # Here, we use the tokenizer to process the text.
        encoding = self.tokenizer(
            text,
            truncation=True,  # This will cut off any text that is longer than max_length.
            padding='max_length',  # This will add 'padding' tokens to any text shorter than max_length, making all samples the same size.
            max_length=self.max_length,  # Sets the maximum length for truncation and padding.
            return_tensors='pt'  # This tells the tokenizer to return the output as PyTorch Tensors.
        )
        
        # The method returns a dictionary containing the processed data for one sample.
        return {
            'input_ids': encoding['input_ids'].flatten(),  # The tokenized text as a flat tensor of numbers.
            'attention_mask': encoding['attention_mask'].flatten(),  # A tensor that tells the model which tokens to pay attention to (and which are just padding).
            'labels': torch.tensor(label, dtype=torch.long)  # The label for this sample, converted to a PyTorch tensor.
        }

def train_classifier():
    # This is the main function that runs the entire training process.
    
    # --- Data Loading and Preparation ---
    print("Loading prepared data from './data/training_data.csv'...")
    # This line reads the training data that our 'prepare_data.py' script created.
    df = pd.read_csv('./data/training_data.csv')
    
    # This line converts the 'prompt_text' column from the DataFrame into a Python list.
    texts = df['prompt_text'].tolist()
    # This line converts the 'is_harmful' column into a list of labels.
    labels = df['is_harmful'].tolist()
    
    print("Splitting data into training and validation sets...")
    # This function splits our data: 80% for training the model, and 20% for validating its performance.
    # 'stratify=labels' is important: it ensures that the training and validation sets have the same percentage of harmful/safe prompts.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- Model and Tokenizer Initialization ---
    print("Initializing tokenizer and model from 'distilbert-base-uncased'...")
    # This line downloads and loads the pre-trained tokenizer for DistilBERT.
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # This line downloads and loads the pre-trained DistilBERT model.
    # 'num_labels=2' tells the model that we have two possible output classes: 0 (safe) and 1 (harmful).
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2
    )
    
    # --- Dataset and DataLoader Creation ---
    print("Creating PyTorch datasets and data loaders...")
    # This line creates an instance of our custom PromptDataset for the training data.
    train_dataset = PromptDataset(train_texts, train_labels, tokenizer)
    # This line does the same for our validation data.
    val_dataset = PromptDataset(val_texts, val_labels, tokenizer)
    
    # The DataLoader takes our dataset and prepares it to be fed to the model in batches.
    # 'batch_size=16' means the model will see 16 samples at a time.
    # 'shuffle=True' is important for training: it shuffles the data at every epoch to make the model more robust.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # --- Training Setup ---
    # This line checks if a GPU is available (like on a powerful server). If not, it will use the CPU.
    # Inside our Docker container, this will be 'cpu'.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # This line moves our model to the selected device (CPU in our case).
    model.to(device)
    
    # This line creates the optimizer, which is responsible for updating the model's weights during training.
    # 'lr=2e-5' is the learning rate, a small number that controls how much the weights are adjusted at each step.
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # --- The Training Loop ---
    print("Starting model training...")
    # This sets the model to "training mode".
    model.train()
    # We will loop over our entire dataset 3 times (3 epochs).
    for epoch in range(3):
        total_loss = 0
        # This tqdm line creates a progress bar for our training loop, which is very helpful for monitoring.
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        # This inner loop iterates over each batch of data from our train_loader.
        for batch in progress_bar:
            # This line clears any old gradients before calculating new ones.
            optimizer.zero_grad()
            
            # These lines move the data for the current batch to our selected device (CPU).
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # This is the forward pass: we feed the data to the model.
            outputs = model(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          labels=labels)
            
            # This line gets the loss (a measure of how wrong the model's predictions were).
            loss = outputs.loss
            # This is the backward pass: it calculates how much each model weight contributed to the loss.
            loss.backward()
            # This line tells the optimizer to update the weights based on the calculated gradients.
            optimizer.step()
            
            # These lines are for reporting the loss in our progress bar.
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader)}')
        
        # --- Validation after each epoch ---
        print("Running validation...")
        # This sets the model to "evaluation mode".
        model.eval()
        val_predictions = []
        val_true = []
        
        # We don't need to calculate gradients for validation, so we use 'torch.no_grad()' to save memory and computations.
        with torch.no_grad():
            # This loop iterates over each batch in our validation data.
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # We get the model's output logits (raw scores).
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # We take the index of the highest score as our prediction (0 or 1).
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # We store the predictions and true labels to evaluate performance later.
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        # This line calculates the overall accuracy of our model on the validation set.
        accuracy = accuracy_score(val_true, val_predictions)
        print(f'Validation Accuracy: {accuracy:.4f}')
        # This prints a detailed report with precision, recall, and F1-score for each class.
        print(classification_report(val_true, val_predictions))
        
        # We set the model back to "training mode" for the next epoch.
        model.train()
    
    # --- Save the Final Model ---
    print("Training complete. Saving the final model...")
    # This saves the fine-tuned model's weights and configuration.
    model.save_pretrained('./models/trained_classifier')
    # This saves the tokenizer, which is important for ensuring we process future text the same way.
    tokenizer.save_pretrained('./models/trained_classifier')
    print("Model saved successfully!")

if __name__ == "__main__":
    # This is the entry point of the script. It calls our main function when you run 'python scripts/train_classifier.py'.
    train_classifier()