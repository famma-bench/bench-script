import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import ast
from omegaconf import OmegaConf
from easyllm_kit.utils import get_logger, read_json
from famma_runner.utils import RDC, LANGUAGE_ORDER
import re

logger = get_logger('dataset_maker', '../question_maker.log')


def prepare_dataset(json_dir, version, source_release):
    """
    Prepare dataset from json and convert it to HuggingFace format.
    
    Args:
        json_dir: Path to json file
        version: Version to use as split name
        source_release: Source release to use as split name
    """
    # Read json file
    json_data = read_json(json_dir)

    df = pd.DataFrame(json_data).transpose()

    # Create a new column for sorting languages
    df['language_order'] = df[RDC.LANGUAGE].map(LANGUAGE_ORDER)

    df[RDC.MAIN_QUESTION_ID] = df[RDC.MAIN_QUESTION_ID].astype(int)
    df[RDC.SUB_QUESTION_ID] = df[RDC.SUB_QUESTION_ID].astype(int)
    # Sort DataFrame with language order first
    df = df.sort_values(['language_order', RDC.MAIN_QUESTION_ID, RDC.SUB_QUESTION_ID])

    # Drop the temporary language_order column
    df = df.drop('language_order', axis=1)

    # Add index column
    df[RDC.INDEX] = range(len(df))

    def process_row(row):
        """Process a single row"""
        # Convert options from list to string if it exists and question is multiple-choice
        options = None
        if row[RDC.QUESTION_TYPE] == 'multiple-choice':
            options_value = row[RDC.OPTIONS]
            
            # Check if options are in the format like 'A. A. Fixed', 'B. B. Floating', etc.
            if isinstance(options_value, list):
                cleaned_options = []
                for option in options_value:
                    # Check if option starts with a pattern like 'A. A.', 'B. B.', etc.
                    match = re.match(r'^([A-D])\.\s+\1\.\s+(.+)$', option)
                    if match:
                        # Remove the duplicate prefix and reformat
                        prefix, content = match.groups()
                        cleaned_options.append(f'{prefix}. {content}')
                    else:
                        cleaned_options.append(option)
                options = cleaned_options
            else:
                # Handle non-list options
                try:
                    if pd.notna(options_value):
                        options_str = str(options_value).strip()
                        if options_str:
                            # Try to parse as a list
                            parsed_options = ast.literal_eval(options_str)
                            if isinstance(parsed_options, list):
                                options = parsed_options
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse OPTIONS for question {row[RDC.QUESTION_ID]}: {options_value}")

        sample = {
            RDC.INDEX: row[RDC.INDEX],
            RDC.QUESTION_ID: row[RDC.QUESTION_ID],
            RDC.SOURCE_RELEASE: source_release,
            RDC.CONTEXT: row[RDC.CONTEXT],
            RDC.QUESTION: row[RDC.QUESTION],
            RDC.OPTIONS: options,
            RDC.ANSWER: row[RDC.ANSWER],
            RDC.THINKING_TRAJECTORY: row[RDC.THINKING_TRAJECTORY],
            RDC.STRUCTURED_THINKING_TRAJECTORY: row[RDC.STRUCTURED_THINKING_TRAJECTORY],
            RDC.TOPIC_DIFFICULTY: row[RDC.TOPIC_DIFFICULTY],
            RDC.QUESTION_TYPE: row[RDC.QUESTION_TYPE],
            RDC.SUBFIELD: row[RDC.SUBFIELD],
            RDC.LANGUAGE: row[RDC.LANGUAGE],
            RDC.MAIN_QUESTION_ID: row[RDC.MAIN_QUESTION_ID],
            RDC.SUB_QUESTION_ID: row[RDC.SUB_QUESTION_ID],
            RDC.IS_ARITHMETIC: row[RDC.IS_ARITHMETIC],
            RDC.RELEASE: version
        }
        
        # we ignore the image columns for now

        return sample

    # Process all rows
    data = []

    for _, row in df.iterrows():
        processed_sample = process_row(row)
        data.append(processed_sample)

    # Create dataset with single split using the version
    splits = {
        version: Dataset.from_list(data, features=RDC.get_features())
    }
    logger.info(f"Created split {version} with {len(data)} samples")

    # Create DatasetDict with the version split
    dataset_dict = DatasetDict(splits)

    return dataset_dict


def upload_to_hub(dataset_dict, repo_name, version, token):
    """
    Upload dataset to HuggingFace Hub
    
    Args:
        dataset_dict: Dataset dictionary to upload
        repo_name: Name of the repository on HuggingFace
        version: Version tag for the dataset
        token: HuggingFace API token
    """
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=False,
            token=token,
            exist_ok=True
        )
        logger.info(f"Created or verified repository: {repo_name}")
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        raise

    # Push the dataset to the hub
    dataset_dict.push_to_hub(
        repo_id=repo_name,
        token=token,
        commit_message=f"Upload dataset version {version}"
    )


def main():
    """
    Main function to prepare and upload dataset to HuggingFace Hub or save locally.
    Reads configuration from data_config.yaml.
    """
    # Load configuration
    config = OmegaConf.load("../configs/reason_data_config.yaml")

    # Initialize DatasetDict
    dataset_dict = DatasetDict()

    # Process each file in source_csv_dir
    for entry in config.data.source_json_dir:
        version = entry['version']
        json_dir = entry['path']
        source_release = entry['source_release']
        logger.info(f"Preparing dataset for version: {version}")
        dataset = prepare_dataset(
            json_dir=json_dir,
            version=version,
            source_release=source_release
        )

        # Add to DatasetDict with version as split name
        dataset_dict[version] = dataset[version]

    # Upload to HuggingFace Hub
    logger.info("Uploading to HuggingFace Hub...")
    upload_to_hub(
        dataset_dict=dataset_dict,
        repo_name=config.hf.repo_name,
        version=config.hf.version,
        token=config.hf.token
    )

    logger.info(f"Dataset process completed. Check {config.hf.repo_name} for results.")


if __name__ == "__main__":
    main()
