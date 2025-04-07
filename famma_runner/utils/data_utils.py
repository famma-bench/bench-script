import os
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from datasets import load_dataset
from easyllm_kit.utils import save_json
import numpy as np
import pandas as pd
import base64
from famma_runner.utils.data_const import DatasetColumns as DC


def convert_to_json_list(dataset, save_dir="./hf_data", release_version="release_v2406", decode_answer=False):
    """
    Convert data in Dataset format to list format.
    Saves images locally and returns their paths.
    
    Args:
        dataset: HuggingFace dataset
        save_dir: Base directory to save images
        release_version: Version string to append to images folder
        decode_answer: If True, decode base64-encoded answers
    """
    json_list = []
    # Create images directory if it doesn't exist
    images_dir = os.path.join(save_dir, f"images_{release_version}")
    os.makedirs(images_dir, exist_ok=True)

    for idx, sample in enumerate(dataset):
        sample_res = {}
        for key, value in sample.items():
            if isinstance(value, (Image.Image, JpegImageFile, PngImageFile)):
                # Create a unique filename for the image
                image_filename = f"{sample['question_id']}_{key}.jpg"
                image_path = os.path.join(images_dir, image_filename)

                # Convert RGBA to RGB if needed
                if value.mode == 'RGBA':
                    value = value.convert('RGB')

                # Save the image locally
                value.save(image_path, format="JPEG")

                # Store the relative path in the JSON
                sample_res[key] = os.path.join(f"images_{release_version}", image_filename)
            elif decode_answer and (key == 'answers' or key == 'explanation') and isinstance(value, str):
                # Decode base64-encoded answers if flag is set
                try:
                    decoded_value = base64.b64decode(value).decode('utf-8')
                    sample_res[key] = decoded_value
                except:
                    # If decoding fails, keep the original value
                    sample_res[key] = value
            else:
                sample_res[key] = value
        json_list.append(sample_res)
    return json_list


def download_data(hf_dir, split=None, save_dir="./hf_data", from_local=False, decode_answer=False):
    """
    Download dataset from HuggingFace repo and convert to JSON files.
    Images are saved locally in {save_dir}/images/.
    
    Args:
        hf_dir (str): HuggingFace repository name (e.g., 'weaverbirdllm/famma')
        split (str, optional): Specific split to download. If None, downloads all splits.
        save_dir (str): Directory to save the JSON files and images
        from_local (bool): If True, load from local cache instead of HuggingFace
        decode_answer (bool): If True, decode base64-encoded answers
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        if split:
            if from_local:
                from datasets import load_from_disk
                # Load from local cache
                dataset = load_from_disk(hf_dir)
            else:
                # Load from HuggingFace
                dataset = load_dataset(hf_dir, split=split, cache_dir=save_dir)
            json_list = convert_to_json_list(dataset, save_dir=save_dir, release_version=split,
                                             decode_answer=decode_answer)

            # Save to JSON file
            split_path = os.path.join(save_dir, f"{split}.json")
            save_json(json_list, split_path)
            print(f"Saved {split} split to {split_path}")

        else:
            dataset = load_dataset(hf_dir)
            for split_name in dataset.keys():
                json_list = convert_to_json_list(dataset[split_name], save_dir=save_dir, release_version=split_name,
                                                 decode_answer=decode_answer)

                # Save to JSON file
                split_path = os.path.join(save_dir, f"{split_name}.json")
                save_json(json_list, split_path)
                print(f"Saved {split_name} split to {split_path}")

        print(f"\nDataset downloaded and saved to {save_dir}")
        print(f"Images are saved in {os.path.join(save_dir, f'images_{split}')}")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False


def order_by_language(df, language_order, main_question_col, sub_question_col, language_col):
    """Prepares the dataset by sorting it based on language order and converting
    specified columns to integers.

    Args:
        df (pd.DataFrame): The DataFrame to be prepared.
        language_order (dict): A dictionary mapping languages to their order.
        main_question_col (str): The column name for main question IDs.
        sub_question_col (str): The column name for sub question IDs.
        language_col (str): The column name for language.

    Returns:
        pd.DataFrame: A sorted and prepared DataFrame.
    """
    # Create a new column for sorting languages
    df['language_order'] = df[language_col].map(language_order)

    # Convert main_question_id, sub_question_id to int
    df[main_question_col] = df[main_question_col].astype(int)
    df[sub_question_col] = df[sub_question_col].astype(int)

    # Sort DataFrame with language order first
    df.sort_values(['language_order', main_question_col, sub_question_col], inplace=True)


def sample_questions(df, num_english_main_questions=40, num_chinese_main_questions=20, num_french_main_questions=10):
    """
    Sample questions from the dataset.
    """

    def sample_language_questions(df, language, num_questions):
        main_ids = df[df[DC.LANGUAGE] == language][DC.MAIN_QUESTION_ID].unique()
        sampled_main_ids = np.random.choice(main_ids, num_questions, replace=False)
        sampled_df = df[df[DC.LANGUAGE] == language][df[DC.MAIN_QUESTION_ID].isin(sampled_main_ids)]
        print(f"Sampled {len(sampled_df)} {language} questions")
        return sampled_df

    # Sample questions for each language
    language_samples = {
        'english': num_english_main_questions,
        'chinese': num_chinese_main_questions,
        'french': num_french_main_questions
    }

    sampled_dfs = [sample_language_questions(df, lang, n) for lang, n in language_samples.items()]
    sampled_df = pd.concat(sampled_dfs)

    print(f"Sampled {len(sampled_df)} questions")

    # the selected question ids are unique
    selected_question_ids = sampled_df[DC.QUESTION_ID].tolist()

    res_df = df[~df[DC.QUESTION_ID].isin(selected_question_ids)]
    print(f"Remaining {len(res_df)} questions")

    # as we extract some questions, the main question ids become non-continuous
    # we need to reindex the main question ids
    # Process each dataframe separately
    for df_to_process in [sampled_df, res_df]:
        # Process each language separately
        for language in df_to_process[DC.LANGUAGE].unique():
            # Get subset for this language
            language_mask = df_to_process[DC.LANGUAGE] == language
            language_df = df_to_process[language_mask]

            # Get unique main question IDs in sorted order
            unique_main_ids = sorted(language_df[DC.MAIN_QUESTION_ID].unique())

            # Create mapping from old to new main question IDs
            main_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_main_ids, 1)}

            # Update main question IDs
            df_to_process.loc[language_mask, DC.MAIN_QUESTION_ID] = \
                df_to_process.loc[language_mask, DC.MAIN_QUESTION_ID].map(main_id_mapping)

            # For each main question, reindex the subquestions
            for main_id in df_to_process.loc[language_mask, DC.MAIN_QUESTION_ID].unique():
                main_q_mask = (df_to_process[DC.MAIN_QUESTION_ID] == main_id) & language_mask
                # Sort by existing subquestion IDs to maintain order
                sorted_indices = df_to_process[main_q_mask].sort_values(DC.SUB_QUESTION_ID).index
                # Assign new sequential subquestion IDs starting from 1
                df_to_process.loc[sorted_indices, DC.SUB_QUESTION_ID] = \
                    range(1, len(sorted_indices) + 1)

    return sampled_df, res_df


def encode_answer(text):
    """Encode text to base64"""
    if pd.isna(text):
        return ""
    return base64.b64encode(str(text).encode('utf-8')).decode('utf-8')


def decode_answer(text):
    """Decode text from base64"""
    if pd.isna(text):
        return ""
    return base64.b64decode(text).decode('utf-8')
