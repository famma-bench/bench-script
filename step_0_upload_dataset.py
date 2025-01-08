import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os
import json
from famma_runner.utils.data_const import DatasetColumns as DC


def download_from_google_sheet(sheet_id, sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    print(df.head())

    # save to csv
    df.to_csv(f"{sheet_name}.csv", index=False)
    return df



def prepare_dataset(csv_path):
    """
    Prepare dataset from CSV and convert it to HuggingFace format.
    Splits are determined by unique values in the 'release' column (e.g., 'release_v1', 'release_v2').
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    def process_row(row):
        # Process main images
        images = [row[col] for col in DC.image_columns() if pd.notna(row[col])]
        # Process answer images
        ans_images = [row[col] for col in DC.answer_image_columns() if pd.notna(row[col])]
        
        # Convert options from string to dict if it exists
        options = json.loads(row[DC.OPTIONS]) if pd.notna(row[DC.OPTIONS]) else None

        # question id = language + main_question_id + sub_question_id + releave_version
        question_id = f"{row[DC.LANGUAGE]}_{row[DC.MAIN_QUESTION_ID]}_{row[DC.SUB_QUESTION_ID]}_{row[DC.RELEASE]}"

        sample = {
            DC.QUESTION_ID: question_id,
            DC.CONTEXT: row[DC.CONTEXT],
            DC.QUESTION: row[DC.QUESTION],
            DC.OPTIONS: options,
            'images': images,
            'answer_images': ans_images,
            DC.IMAGE_TYPE: row[DC.IMAGE_TYPE],
            DC.ANSWERS: row[DC.ANSWERS],
            DC.EXPLANATION: row[DC.EXPLANATION],
            DC.TOPIC_DIFFICULTY: row[DC.TOPIC_DIFFICULTY],
            DC.QUESTION_TYPE: row[DC.QUESTION_TYPE],
            DC.SUBFIELD: row[DC.SUBFIELD],
            DC.LANGUAGE: row[DC.LANGUAGE],
            DC.MAIN_QUESTION_ID: row[DC.MAIN_QUESTION_ID],
            DC.SUB_QUESTION_ID: row[DC.SUB_QUESTION_ID],
            DC.RELEASE: row[DC.RELEASE]
        }
        
        # Validate the sample
        DC.validate_sample(sample)
        return sample

    # Get unique release versions
    release_versions = df[DC.RELEASE].unique()
    
    # Process each release version
    splits = {}
    for release in release_versions:
        release_df = df[df[DC.RELEASE] == release]
        release_data = [process_row(row) for _, row in release_df.iterrows()]
        splits[release] = Dataset.from_list(release_data, features=DC.get_features())
        print(f"Split {release}: {len(release_data)} samples")
    
    # Create DatasetDict with all splits
    dataset_dict = DatasetDict(splits)
    
    return dataset_dict

def upload_to_hub(dataset_dict, repo_name, version):
    """
    Upload dataset to HuggingFace Hub
    """
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=False
        )
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")
    
    # Push dataset to hub
    dataset_dict.push_to_hub(
        repo_id=repo_name,
        token=os.environ.get("HF_TOKEN"),  # Make sure to set your HF_TOKEN environment variable
        tag=version
    )

def main():
    # Configuration
    CSV_PATH = "path/to/your/dataset.csv"
    REPO_NAME = "your-username/dataset-name"  # e.g., "username/famma-dataset"
    VERSION = "v1.0.0"  # or any version tag you want to use
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_dict = prepare_dataset(CSV_PATH)
    
    # Upload to HuggingFace Hub
    print("Uploading to HuggingFace Hub...")
    upload_to_hub(dataset_dict, REPO_NAME, VERSION)
    
    print(f"Dataset uploaded successfully to {REPO_NAME} with tag {VERSION}")
    print("You can now load the dataset using:")
    print(f"dataset = load_dataset('{REPO_NAME}', tag='{VERSION}')")

if __name__ == "__main__":
    main()
