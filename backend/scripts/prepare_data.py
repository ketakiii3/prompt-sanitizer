# backend/scripts/prepare_data.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_process_local_data():
    """
    This function loads the Jigsaw training data from a local CSV file that you have already downloaded.
    It then processes the data to get it ready for training.
    """
    # This line defines the exact path to your 'train.csv' file inside the Docker container.
    # The path is relative to the `backend` directory, which is our working directory.
    local_csv_path = "./jigsaw_toxicity_pred/train.csv"
    
    print(f"Attempting to load dataset from local file: {local_csv_path}")
    
    try:
        # This line reads your local CSV file into a pandas DataFrame.
        df = pd.read_csv(local_csv_path)
    except FileNotFoundError:
        # This is a crucial error check. If the file isn't found, the script will stop and tell you exactly why.
        print("-" * 60)
        print(f"FATAL ERROR: The file was not found at '{local_csv_path}'")
        print("Please ensure the 'jigsaw_toxicity_pred' directory containing 'train.csv' is placed directly inside the 'backend' directory.")
        print("-" * 60)
        # This command stops the script to prevent further errors.
        exit(1)

    print("Local dataset loaded successfully. Preparing data for the model...")

    # The original dataset has multiple toxicity columns (toxic, severe_toxic, etc.).
    # We create a single 'is_harmful' column. If any of the toxicity columns have a 1, 'is_harmful' will be 1. Otherwise, it's 0.
    df['is_harmful'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

    # The training script expects the text column to be named 'prompt_text'.
    # This line renames 'comment_text' to 'prompt_text' to match what the model trainer expects.
    df = df.rename(columns={'comment_text': 'prompt_text'})

    # For our purpose, we only need the text of the prompt and our new 'is_harmful' label.
    # This line selects just those two columns and discards the rest.
    df = df[['prompt_text', 'is_harmful']]

    # The original dataset is heavily unbalanced (many more non-toxic comments than toxic ones).
    # To fix this, we first separate the harmful comments from the safe ones.
    harmful_df = df[df['is_harmful'] == 1]
    safe_df = df[df['is_harmful'] == 0]

    # Now, we take a random sample of the safe comments. This prevents the model from just learning to always say "safe".
    # We are sampling a number of safe comments equal to twice the number of harmful comments to have a good mix.
    safe_df_sampled = safe_df.sample(n=len(harmful_df) * 2, random_state=42)

    # Finally, we combine the harmful comments and the newly sampled safe comments back into one big DataFrame.
    balanced_df = pd.concat([harmful_df, safe_df_sampled])

    # This line shuffles all the rows in the DataFrame randomly. This is important for training a robust model.
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def prepare_training_data():
    """
    This is the main function that prepares and splits the data into the final training and testing files.
    """
    # This line makes sure the './data' directory exists. We will save our final files there.
    os.makedirs("./data", exist_ok=True)

    # This line calls our function above to get the processed and balanced data.
    df = load_and_process_local_data()

    # This line splits our balanced data into a training set (which the model learns from) and a testing set (which we use to evaluate the model).
    # We are using 80% for training and 20% for testing.
    # 'stratify' ensures that both the training and testing sets have the same percentage of harmful/safe comments.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['is_harmful']
    )

    # These lines save our final, prepared data into two new CSV files. The training script will read these files.
    train_df.to_csv("./data/training_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)

    # This block prints a summary of what we've done, so you can see the result.
    print("-" * 50)
    print("Data preparation complete!")
    print(f"Training data saved to ./data/training_data.csv ({len(train_df)} samples)")
    print(f"Test data saved to ./data/test_data.csv ({len(test_df)} samples)")
    print(f"Harmful samples in training set: {train_df['is_harmful'].sum()}")
    print("-" * 50)


if __name__ == "__main__":
    # This is the entry point of the script. When you run `python scripts/prepare_data.py`, this function is called.
    prepare_training_data()