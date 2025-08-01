# backend/scripts/prepare_data.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_process_local_data():
    """
    Loads the Jigsaw training data from a local CSV file, processes it,
    and returns a pandas DataFrame.
    """
    # This line defines the path to your manually downloaded CSV file.
    # The path is relative to the `backend` directory, which is our working directory in the Docker container.
    local_csv_path = "./jigsaw_toxicity_pred/train.csv"
    
    print(f"Loading dataset from local file: {local_csv_path}")
    
    try:
        # This line reads the CSV file into a pandas DataFrame.
        df = pd.read_csv(local_csv_path)
    except FileNotFoundError:
        # This is a safety check. If the file isn't found, the script will exit with a helpful message.
        print(f"Error: The file was not found at {local_csv_path}")
        print("Please make sure the 'jigsaw_toxicity_pred' directory containing 'train.csv' is placed inside the 'backend' directory.")
        exit(1)

    print("Dataset loaded successfully. Preparing data...")

    # A comment is considered harmful if any of the toxicity labels are 1.
    # This line creates a new column 'is_harmful' with a value of 1 if any toxicity column is 1, and 0 otherwise.
    df['is_harmful'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

    # This line renames 'comment_text' to 'prompt_text' to match the column name expected by the rest of the code.
    df = df.rename(columns={'comment_text': 'prompt_text'})

    # We only need the 'prompt_text' and 'is_harmful' columns for training our model.
    df = df[['prompt_text', 'is_harmful']]

    # We need to balance the dataset because most comments in the original data are not toxic.
    # First, separate the harmful and safe comments.
    harmful_df = df[df['is_harmful'] == 1]
    safe_df = df[df['is_harmful'] == 0]

    # Now, take a random sample of the safe comments.
    # We are sampling a number of safe comments equal to twice the number of harmful comments to create a more balanced set.
    safe_df_sampled = safe_df.sample(n=len(harmful_df) * 2, random_state=42)

    # Finally, combine the harmful comments and the sampled safe comments into one DataFrame.
    balanced_df = pd.concat([harmful_df, safe_df_sampled])

    # This line shuffles the dataset to ensure the data is not ordered in any way before splitting.
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def prepare_training_data():
    """
    Prepares and splits the data into training and testing sets.
    """
    # This line ensures the 'data' directory exists where we will save our final CSV files for training.
    os.makedirs("./data", exist_ok=True)

    # This line calls our function to load and process the local data.
    df = load_and_process_local_data()

    # This line splits the DataFrame into a training set (80%) and a testing set (20%).
    # stratify=df['is_harmful'] ensures both sets have the same proportion of harmful vs. safe comments.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_harmful']
    )

    # These lines save the prepared data to CSV files, which the training script will then use.
    train_df.to_csv("./data/training_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)

    # This block prints a summary of the data we've prepared.
    print("-" * 50)
    print(f"Training data saved: {len(train_df)} samples")
    print(f"Test data saved: {len(test_df)} samples")
    print(f"Harmful samples in training: {train_df['is_harmful'].sum()}")
    print(f"Safe samples in training: {len(train_df) - train_df['is_harmful'].sum()}")
    print("-" * 50)


if __name__ == "__main__":
    # This line calls the main function to start the data preparation process when the script is run.
    prepare_training_data()