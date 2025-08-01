# backend/scripts/prepare_data.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io

def download_and_process_jigsaw_data():
    """
    Downloads the Jigsaw Toxic Comment Classification dataset from a direct URL,
    processes it, and returns a pandas DataFrame.
    This method avoids using the huggingface `datasets` library.
    """
    # This is a direct download link to the zip file from the original Kaggle competition.
    url = "https://storage.googleapis.com/paparoma-data/jigsaw-toxic-comment-classification-challenge.zip"
    
    print("Downloading Jigsaw dataset from direct link...")
    
    try:
        # This line sends a request to the URL to get the file.
        r = requests.get(url, stream=True)
        # This line will raise an error if the download fails (e.g., 404 Not Found).
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        # This line exits the script if the download fails, preventing further errors.
        exit(1)

    print("Download complete. Processing in memory...")

    try:
        # This line opens the downloaded zip file directly from memory without saving it to disk.
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # This block opens the 'train.csv' file from within the zip archive
        # and loads it into a pandas DataFrame.
        with z.open('train.csv') as f:
            df = pd.read_csv(f)
            
    except Exception as e:
        print(f"Error processing the zip file or reading the CSV: {e}")
        exit(1)

    print("Dataset loaded successfully. Preparing data...")

    # A comment is considered harmful if any of the toxicity labels are 1.
    # This line creates a new column 'is_harmful' with a value of 1 if any of the toxicity columns are 1, and 0 otherwise.
    df['is_harmful'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

    # Rename 'comment_text' to 'prompt_text' to match the rest of the project's code.
    df = df.rename(columns={'comment_text': 'prompt_text'})

    # We only need the 'prompt_text' and 'is_harmful' columns for our model.
    df = df[['prompt_text', 'is_harmful']]

    # Separate the safe and harmful prompts to create a balanced dataset.
    harmful_df = df[df['is_harmful'] == 1]
    safe_df = df[df['is_harmful'] == 0]

    # To avoid a heavily biased model, we sample a smaller portion of the non-toxic comments.
    # We'll take twice the number of harmful comments to have a good amount of safe examples.
    safe_df_sampled = safe_df.sample(n=len(harmful_df) * 2, random_state=42)

    # Combine the harmful comments and the sampled safe comments.
    balanced_df = pd.concat([harmful_df, safe_df_sampled])

    # Shuffle the dataset to ensure the data is not ordered in any way before training.
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def prepare_training_data():
    """
    Prepare and split training data.
    """
    # This line ensures the 'data' directory exists where we will save our final CSV files.
    os.makedirs("./data", exist_ok=True)

    df = download_and_process_jigsaw_data()

    # Split the DataFrame into training (80%) and testing (20%) sets.
    # stratify=df['is_harmful'] ensures both sets have the same proportion of harmful/safe comments.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_harmful']
    )

    # Save the prepared data to CSV files, which the training script will use.
    train_df.to_csv("./data/training_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)

    print("-" * 50)
    print(f"Training data saved: {len(train_df)} samples")
    print(f"Test data saved: {len(test_df)} samples")
    print(f"Harmful samples in training: {train_df['is_harmful'].sum()}")
    print(f"Safe samples in training: {len(train_df) - train_df['is_harmful'].sum()}")
    print("-" * 50)


if __name__ == "__main__":
    prepare_training_data()