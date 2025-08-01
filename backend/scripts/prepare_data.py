# scripts/prepare_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def download_and_process_jigsaw_data():
    """
    Downloads the Jigsaw Toxic Comment Classification dataset from Hugging Face,
    processes it, and returns a pandas DataFrame.
    """
    # Load the dataset from Hugging Face
    print("Downloading Jigsaw Toxic Comment Classification dataset...")
    dataset = load_dataset("jigsaw_toxicity_pred", "toxic_comment_classification")
    print("Dataset downloaded successfully.")

    # Convert the training split to a pandas DataFrame
    df = dataset['train'].to_pandas()

    # Create the 'is_harmful' column. A comment is considered harmful if any of the toxicity labels are 1.
    df['is_harmful'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

    # Rename 'comment_text' to 'prompt_text' to match the rest of the project's code
    df = df.rename(columns={'comment_text': 'prompt_text'})

    # Keep only the 'prompt_text' and 'is_harmful' columns
    df = df[['prompt_text', 'is_harmful']]

    # Separate the safe and harmful prompts to balance the dataset
    harmful_df = df[df['is_harmful'] == 1]
    safe_df = df[df['is_harmful'] == 0]

    # Sample a smaller portion of the safe comments to create a more balanced dataset
    # This is done because the original dataset has a large number of non-toxic comments
    safe_df_sampled = safe_df.sample(n=len(harmful_df) * 2, random_state=42) # sampling twice the number of harmful comments

    # Concatenate the harmful and sampled safe comments into a single DataFrame
    balanced_df = pd.concat([harmful_df, safe_df_sampled])

    # Shuffle the dataset to ensure the data is not ordered in any way
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def prepare_training_data():
    """
    Prepare and split training data.
    """
    # Create the data directory if it doesn't already exist
    os.makedirs("./data", exist_ok=True)

    # Download and process the Jigsaw dataset
    df = download_and_process_jigsaw_data()

    # Split the DataFrame into training and testing sets
    # test_size=0.2 means 20% of the data will be used for testing
    # random_state=42 ensures that the split is the same every time you run the script
    # stratify=df['is_harmful'] ensures that the proportion of harmful and safe prompts is the same in both the training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_harmful']
    )

    # Save the training and testing DataFrames to CSV files
    train_df.to_csv("./data/training_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)

    # Print out some information about the created datasets
    print(f"Training data saved: {len(train_df)} samples")
    print(f"Test data saved: {len(test_df)} samples")
    print(f"Harmful samples in training: {train_df['is_harmful'].sum()}")
    print(f"Safe samples in training: {len(train_df) - train_df['is_harmful'].sum()}")

if __name__ == "__main__":
    # This block of code will only run when you execute the script directly
    prepare_training_data()