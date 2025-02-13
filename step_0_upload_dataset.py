import pandas as pd
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi
import ast
from pathlib import Path
from omegaconf import OmegaConf
from easyllm_kit.utils import get_logger
from famma_runner.utils import find_image_file, DC, ReleaseVersion, LANGUAGE_ORDER
from PIL import Image

logger = get_logger('dataset_maker', 'question_maker.log')


def validate_question_id(df):
    """
    Validate question IDs in the dataset.
    
    Checks:
    1. main_question_id should be monotonically increasing and continuous
       e.g., [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6...]
    2. sub_question_id should be strictly monotonically increasing within each main_question_id
       e.g., [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6...]
    """
    # Convert IDs to integers for comparison
    df = df.copy()
    df[DC.MAIN_QUESTION_ID] = df[DC.MAIN_QUESTION_ID].astype(int)
    df[DC.SUB_QUESTION_ID] = df[DC.SUB_QUESTION_ID].astype(int)

    # Group by release and language to validate each release separately
    for (release, language), release_df in df.groupby([DC.RELEASE, DC.LANGUAGE]):
        logger.info(f"Validating {len(release_df)} question IDs for release: {release} and language: {language}")

        # Sort by main_question_id to check monotonicity
        release_df = release_df.sort_values([DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID])

        # Check main_question_id continuity
        unique_main_ids = sorted(release_df[DC.MAIN_QUESTION_ID].unique())
        expected_main_ids = list(range(1, len(unique_main_ids) + 1))
        if unique_main_ids != expected_main_ids:
            missing_ids = set(expected_main_ids) - set(unique_main_ids)
            raise ValueError(f"Non-continuous main_question_ids in {release}. Missing IDs: {missing_ids}")

        # Check sub_question_id monotonicity within each main_question_id
        for main_id, group in release_df.groupby(DC.MAIN_QUESTION_ID):
            sub_ids = group[DC.SUB_QUESTION_ID].tolist()

            # Check if sub_question_ids start from 1
            if sub_ids[0] != 1:
                raise ValueError(f"Sub-question IDs for main_question_id {main_id} in {release} don't start from 1")

            # Check if sub_question_ids are continuous and increasing
            expected_sub_ids = list(range(1, len(sub_ids) + 1))
            if sub_ids != expected_sub_ids:
                raise ValueError(
                    f"Invalid sub_question_ids for main_question_id {main_id} in {release}. "
                    f"Expected {expected_sub_ids}, got {sub_ids}"
                )

    logger.info("Question ID validation passed successfully")
    return True


def validate_columns(df):
    """
    Validate column values in the dataset.
    
    Checks:
    1. TOPIC_DIFFICULTY must be in ['easy', 'medium', 'hard']
    2. QUESTION_TYPE must be in ['open question', 'multiple-choice']
    3. LANGUAGE must be in ['english', 'chinese', 'french']
    4. IMAGE_TYPE must not contain any null values
    5. For multiple-choice questions, OPTIONS must be a valid list of strings
    """
    # check if all values in TOPIC_DIFFICULTY are in ['easy', 'medium', 'hard']
    if not df[DC.TOPIC_DIFFICULTY].isin(['easy', 'medium', 'hard']).all():
        raise ValueError(f"Invalid values in {DC.TOPIC_DIFFICULTY} column")

    # check if all values in QUESTION_TYPE are in ['open question', 'multiple-choice']
    if not df[DC.QUESTION_TYPE].isin(['open question', 'multiple-choice']).all():
        raise ValueError(f"Invalid values in {DC.QUESTION_TYPE} column")

    # check if all values in LANGUAGE are in ['english', 'chinese', 'french']
    if not df[DC.LANGUAGE].isin(['english', 'chinese', 'french']).all():
        raise ValueError(f"Invalid values in {DC.LANGUAGE} column")

    # check if all values in IMAGE_TYPE are not null
    if df[DC.IMAGE_TYPE].isna().any():
        raise ValueError(f"Null values found in {DC.IMAGE_TYPE} column")

    # check OPTIONS format for multiple-choice questions
    multiple_choice_df = df[df[DC.QUESTION_TYPE] == 'multiple-choice']
    for idx, row in multiple_choice_df.iterrows():
        options = row[DC.OPTIONS]
        if pd.isna(options) or not isinstance(options, str):
            raise ValueError(f"Missing or invalid OPTIONS for multiple-choice question at index {idx}")

        # Clean and parse options
        try:
            # First try: direct parsing after basic cleaning
            options_cleaned = (options
                               .replace("'", "'")  # Replace curly apostrophe
                               .replace("'", "'")  # Replace another curly apostrophe
                               .replace(""", '"')  # Replace curly quotes
                .replace(""", '"')  # Replace another curly quotes
                               )
            # make the options = ['A. xxx', 'B. xxx', 'C. xxx', 'D. xxx']
            options_cleaned = eval(options_cleaned)  # read it as a list
            parsed_options = [f"{chr(65 + i)}. {option.strip()}" for i, option in enumerate(options_cleaned)]

        except (ValueError, SyntaxError):
            # Second try: manual parsing
            try:
                # Remove brackets and split by comma, handling quotes properly
                options_text = options.strip('[]')
                parts = []
                current_part = []
                in_quotes = False
                quote_char = None

                for char in options_text:
                    if char in ["'", '"'] and (not quote_char or char == quote_char):
                        in_quotes = not in_quotes
                        quote_char = char if in_quotes else None
                    elif char == ',' and not in_quotes:
                        parts.append(''.join(current_part).strip())
                        current_part = []
                    else:
                        current_part.append(char)

                if current_part:  # Add the last part
                    parts.append(''.join(current_part).strip())

                # Clean each part
                parsed_options = []
                for part in parts:
                    cleaned = part.strip()
                    # Remove surrounding quotes if they match
                    if (cleaned.startswith("'") and cleaned.endswith("'")) or \
                            (cleaned.startswith('"') and cleaned.endswith('"')):
                        cleaned = cleaned[1:-1]
                    if cleaned:  # Only add non-empty strings
                        parsed_options.append(cleaned)

                if not parsed_options:
                    raise ValueError("No valid options found")

            except Exception as e:
                logger.error(f"Failed to parse options at index {idx}: {options}")
                logger.error(f"Error: {str(e)}")
                raise ValueError(f"Invalid format in OPTIONS at index {idx}: {options}")

        # Validate parsed options
        if not isinstance(parsed_options, list) or not all(isinstance(item, str) for item in parsed_options):
            raise ValueError(f"OPTIONS must be a list of strings at index {idx}")

        # Store cleaned options back in the DataFrame
        df.at[idx, DC.OPTIONS] = str(parsed_options)

    # For open questions, OPTIONS should be empty or null
    open_question_df = df[df[DC.QUESTION_TYPE] == 'open question']
    if not open_question_df[DC.OPTIONS].isna().all():
        non_null_idx = open_question_df[~open_question_df[DC.OPTIONS].isna()].index
        raise ValueError(f"Open questions should not have OPTIONS at indices: {list(non_null_idx)}")

    return True


def prepare_dataset(csv_path, image_parent_dir, version):
    """
    Prepare dataset from CSV and convert it to HuggingFace format.
    
    Args:
        csv_path: Path to CSV file
        image_parent_dir: Path to image directory
        version: Version to use as split name
    """
    # Read CSV file
    df = pd.read_csv(csv_path, header=0)

    # Create a new column for sorting languages
    df['language_order'] = df[DC.LANGUAGE].map(LANGUAGE_ORDER)

    # Sort DataFrame with language order first
    df = df.sort_values(['language_order', DC.RELEASE, DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID])

    # Drop the temporary language_order column
    df = df.drop('language_order', axis=1)

    # Validate question IDs
    validate_question_id(df)
    logger.info("Question IDs validated successfully")

    # Validate columns
    validate_columns(df)
    logger.info("Columns validated successfully")

    # Add index column
    df[DC.INDEX] = range(len(df))

    def process_row(row, is_first_subquestion):
        """Process a single row, attaching images only if it's the first sub-question"""
        # Convert options from string to list if it exists and question is multiple-choice
        options = None
        if row[DC.QUESTION_TYPE] == 'multiple-choice' and pd.notna(row[DC.OPTIONS]):
            try:
                options = ast.literal_eval(row[DC.OPTIONS])
            except (ValueError, SyntaxError):
                logger.warning(f"Invalid format in OPTIONS for question {row[DC.QUESTION_ID]}")
                options = None

        # Get short release name
        release_short = ReleaseVersion.to_short_name(row[DC.RELEASE])

        # question id = language + main_question_id + sub_question_id + release_short
        question_id = f"{row[DC.LANGUAGE]}_{row[DC.MAIN_QUESTION_ID]}_{row[DC.SUB_QUESTION_ID]}_{release_short}"

        sample = {
            DC.INDEX: row[DC.INDEX],
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
                    image_name = image_name.split('.')[0]
                    # Construct full image path by joining parent_dir + subdir + filename
                    image_path = Path(image_parent_dir + row['question_image_parent_dir'][3:]) / image_name
                    # Try to find image with either jpg or png extension
                    full_path = find_image_file(image_path.parent, image_path.name)
                    if full_path is not None:
                        # Read the image file into PIL Image
                        img = Image.open(full_path)
                        # Store the PIL Image object directly
                        sample[image_key] = img
                    else:
                        logger.warning(f"Image not found: {image_path}.[jpg|png]")
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
        for i in range(1, 7):
            ans_image_key = f'ans_image_{i}'
            if is_first_subquestion:
                image_name = row[ans_image_key]
                if pd.notna(image_name):
                    image_name = image_name.split('.')[0]
                    # Construct full image path by joining parent_dir + subdir + filename
                    image_path = Path(image_parent_dir + row['ans_image_parent_dir'][3:]) / image_name
                    # Try to find image with either jpg or png extension
                    full_path = find_image_file(image_path.parent, image_path.name)
                    if full_path is not None:
                        # Read the image file into PIL Image
                        img = Image.open(full_path)
                        # Store the PIL Image object directly
                        sample[ans_image_key] = img
                    else:
                        logger.warning(f"Answer image not found: {image_path}.[jpg|png]")
                        sample[ans_image_key] = None
                else:
                    sample[ans_image_key] = None
            else:
                sample[ans_image_key] = None

        sample[DC.RELEASE] = row[DC.RELEASE]

        # Validate the sample
        DC.validate_sample(sample)
        return sample

    # Process all rows
    current_main_id = None
    data = []

    for _, row in df.iterrows():
        is_first_subquestion = row[DC.MAIN_QUESTION_ID] != current_main_id
        if is_first_subquestion:
            current_main_id = row[DC.MAIN_QUESTION_ID]

        processed_sample = process_row(row, is_first_subquestion)
        data.append(processed_sample)

    # Create dataset with single split using the version
    splits = {
        version: Dataset.from_list(data, features=DC.get_features())
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


def save_dataset_locally(dataset_dict, local_path):
    """
    Save dataset locally to the specified path.
    
    Args:
        dataset_dict: Dataset dictionary to save
        local_path: Path to save the dataset locally
    """
    dataset_dict.save_to_disk(local_path)
    logger.info(f"Dataset saved locally at {local_path}")


def main():
    """
    Main function to prepare and upload dataset to HuggingFace Hub or save locally.
    Reads configuration from data_config.yaml.
    """
    # Load configuration
    config = OmegaConf.load("configs/data_config.yaml")

    # Initialize DatasetDict
    dataset_dict = DatasetDict()

    # Process each file in source_csv_dir
    for entry in config.data.source_csv_dir:
        version = entry['version']
        csv_path = entry['path']

        logger.info(f"Preparing dataset for version: {version}")
        dataset = prepare_dataset(
            csv_path=csv_path,
            image_parent_dir=config.data.source_image_dir,
            version=version
        )

        # Add to DatasetDict with version as split name
        dataset_dict[version] = dataset[version]

    # Check if local caching is enabled
    if config.data.local_cache:
        save_dataset_locally(dataset_dict, config.data.local_cache_dir)
    else:
        # Upload to HuggingFace Hub
        logger.info("Uploading to HuggingFace Hub...")
        upload_to_hub(
            dataset_dict=dataset_dict,
            repo_name=config.hf.repo_name,
            version=config.hf.version,
            token=config.hf.token
        )

    logger.info(f"Dataset process completed. Check {config.hf.repo_name} or {config.data.local_cache_dir} for results.")


if __name__ == "__main__":
    main()
