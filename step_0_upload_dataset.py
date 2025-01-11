import pandas as pd
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi
import os
import json
from pathlib import Path
from omegaconf import OmegaConf
from famma_runner.utils.data_const import DatasetColumns as DC, ReleaseVersion
from easyllm_kit.utils import get_logger 

logger = get_logger('dataset_maker')

def prepare_dataset(csv_dir):
    """
    Prepare dataset from CSV and convert it to HuggingFace format.
    Splits are determined by unique values in the 'release' column (e.g., 'release_v2406', 'release_v2501').
    """
    # Read CSV file
    df = pd.read_csv(csv_dir, header=0)
    
    def process_row(row):
        # Convert options from string to dict if it exists
        options = json.loads(row[DC.OPTIONS]) if pd.notna(row[DC.OPTIONS]) else None
        
        # Get short release name
        release_short = ReleaseVersion.to_short_name(row[DC.RELEASE])
        
        # question id = language + main_question_id + sub_question_id + release_short
        question_id = f"{row[DC.LANGUAGE]}_{row[DC.MAIN_QUESTION_ID]}_{row[DC.SUB_QUESTION_ID]}_{release_short}"

        sample = {
            DC.QUESTION_ID: question_id,
            DC.CONTEXT: row[DC.CONTEXT],
            DC.QUESTION: row[DC.QUESTION],
            DC.OPTIONS: options,
        }
        
        # Add image columns
        for i in range(1, 8):
            image_key = f'image_{i}'
            image_path = row[image_key]
            # If image path exists, keep it, otherwise empty string
            sample[image_key] = str(image_path) if pd.notna(image_path) else ""
        
        sample[DC.IMAGE_TYPE] = row[DC.IMAGE_TYPE]
        
        sample[DC.ANSWER] = row[DC.ANSWER]
        sample[DC.EXPLANATION] = row[DC.EXPLANATION]
        sample[DC.TOPIC_DIFFICULTY] = row[DC.TOPIC_DIFFICULTY]
        sample[DC.QUESTION_TYPE] = row[DC.QUESTION_TYPE]
        sample[DC.SUBFIELD] = row[DC.SUBFIELD]
        sample[DC.LANGUAGE] = row[DC.LANGUAGE]
        sample[DC.MAIN_QUESTION_ID] = row[DC.MAIN_QUESTION_ID]
        sample[DC.SUB_QUESTION_ID] = row[DC.SUB_QUESTION_ID]

        # Add answer image columns
        for i in range(1, 4):
            ans_image_key = f'ans_image_{i}'
            image_path = row[ans_image_key]
            # If image path exists, keep it, otherwise empty string
            sample[ans_image_key] = str(image_path) if pd.notna(image_path) else ""
        
        sample[DC.RELEASE] = row[DC.RELEASE]
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
        # Use short name for split key
        split_key = ReleaseVersion.to_short_name(release)
        splits[split_key] = Dataset.from_list(release_data, features=DC.get_features())
        print(f"Split {split_key} (from {release}): {len(release_data)} samples")
    
    # Create DatasetDict with all splits
    dataset_dict = DatasetDict(splits)
    
    return dataset_dict

def upload_to_hub(dataset_dict, repo_name, version):
    """
    Upload dataset to HuggingFace Hub
    """
    api = HfApi()
    
    # Push dataset to hub
    dataset_dict.push_to_hub(
        repo_id=repo_name,
        token=os.environ.get("HF_TOKEN"),  # Make sure to set your HF_TOKEN environment variable
        tag=version
    )

def main():
    # Configuration
    config = OmegaConf.load("make_data_config.yaml")
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
