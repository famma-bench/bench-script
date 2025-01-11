import pandas as pd
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi
import os
import json
from pathlib import Path
from omegaconf import OmegaConf
from easyllm_kit.utils import get_logger 
from typing import Optional
from famma_runner.utils import find_image_file, DC, ReleaseVersion

logger = get_logger('dataset_maker')


def prepare_dataset(csv_dir, image_parent_dir):
    """
    Prepare dataset from CSV and convert it to HuggingFace format.
    Splits are determined by unique values in the 'release' column.
    Images are only attached to the first sub-question of each main question.
    
    Args:
        csv_dir: Path to the CSV file
        image_parent_dir: Root directory containing all images
    """
    # Read CSV file and sort by main_question_id and sub_question_id
    df = pd.read_csv(csv_dir, header=0)
    df = df.sort_values([DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID])
    image_parent_dir = Path(image_parent_dir)
    
    def process_row(row, is_first_subquestion):
        """Process a single row, attaching images only if it's the first sub-question"""
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
        
        # Add image columns - only if this is the first sub-question
        for i in range(1, 8):
            image_key = f'image_{i}'
            if is_first_subquestion:
                image_name = row[image_key]
                if pd.notna(image_name):
                    # Try to find image with either jpg or png extension
                    full_path = find_image_file(image_parent_dir, image_name)
                    if full_path is not None:
                        sample[image_key] = str(full_path)
                    else:
                        logger.warning(f"Image not found: {image_name}.[jpg|png]")
                        sample[image_key] = None
                else:
                    sample[image_key] = None
            else:
                sample[image_key] = None
        
        sample[DC.IMAGE_TYPE] = row[DC.IMAGE_TYPE]
        
        sample[DC.ANSWER] = row[DC.ANSWER]
        sample[DC.EXPLANATION] = row[DC.EXPLANATION]
        sample[DC.TOPIC_DIFFICULTY] = row[DC.TOPIC_DIFFICULTY]
        sample[DC.QUESTION_TYPE] = row[DC.QUESTION_TYPE]
        sample[DC.SUBFIELD] = row[DC.SUBFIELD]
        sample[DC.LANGUAGE] = row[DC.LANGUAGE]
        sample[DC.MAIN_QUESTION_ID] = row[DC.MAIN_QUESTION_ID]
        sample[DC.SUB_QUESTION_ID] = row[DC.SUB_QUESTION_ID]

        # Add answer image columns - only if this is the first sub-question
        for i in range(1, 4):
            ans_image_key = f'ans_image_{i}'
            if is_first_subquestion:
                image_name = row[ans_image_key]
                if pd.notna(image_name):
                    # Try to find image with either jpg or png extension
                    full_path = find_image_file(image_parent_dir, image_name)
                    if full_path is not None:
                        sample[ans_image_key] = str(full_path)
                    else:
                        logger.warning(f"Answer image not found: {image_name}.[jpg|png]")
                        sample[ans_image_key] = None
                else:
                    sample[ans_image_key] = None
            else:
                sample[ans_image_key] = None

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
        
        # Process rows, tracking first sub-question for each main question
        current_main_id = None
        release_data = []
        
        for _, row in release_df.iterrows():
            is_first_subquestion = row[DC.MAIN_QUESTION_ID] != current_main_id
            if is_first_subquestion:
                current_main_id = row[DC.MAIN_QUESTION_ID]
            
            processed_sample = process_row(row, is_first_subquestion)
            release_data.append(processed_sample)
        
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
    """
    Main function to prepare and upload dataset to HuggingFace Hub.
    Reads configuration from make_data_config.yaml.
    """
    # Load configuration
    config = OmegaConf.load("make_data_config.yaml")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset_dict = prepare_dataset(
        csv_dir=config.data.csv_path,
        image_parent_dir=config.data.image_dir
    )
    
    # Upload to HuggingFace Hub
    logger.info("Uploading to HuggingFace Hub...")
    upload_to_hub(
        dataset_dict=dataset_dict,
        repo_name=config.hf.repo_name,
        version=config.hf.version
    )
    
    logger.info(f"Dataset uploaded successfully to {config.hf.repo_name} with tag {config.hf.version}")
    logger.info("You can now load the dataset using:")
    logger.info(f"dataset = load_dataset('{config.hf.repo_name}', tag='{config.hf.version}')")

if __name__ == "__main__":
    main()
