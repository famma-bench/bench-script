import base64
import json
import os
import ast
import string
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from io import BytesIO
from datasets import load_dataset
from easyllm_kit.utils import save_json


def image_to_base64(image):
    """
    Convert image to base64 format
    """
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def remove_json_format(json_str):
    json_str = json_str.strip().lstrip('```json').strip()
    json_str = json_str.rstrip('```').strip()
    return json_str


def parse_list(value):
    """
    Convert a comma-separated string to a list.
    """
    return value.split(',')


def format_options(options_str):
    """
    Parse the options string and format it with labels.
    """
    # Parse the options string into a list
    options = ast.literal_eval(options_str.strip().replace('\n', ' '))
    # Generate labels (A, B, C, etc.) for each option
    labels = string.ascii_uppercase[:len(options)]
    # Create the options dictionary with labels as keys
    options_dict = {label: option for label,
    option in zip(labels, options)}
    # Format the options into a string with each option on a new line, prefixed by its label
    formatted_options = "\n".join(
        [f"{label}: {option}" for label, option in zip(labels, options)])

    return {
        "labels": labels,
        "options_dict": options_dict,
        "formatted_options": formatted_options
    }


def convert_to_json_list(dataset, save_dir="./hf_data"):
    """
    Convert data in Dataset format to list format.
    Saves images locally and returns their paths.
    
    Args:
        dataset: HuggingFace dataset
        save_dir: Base directory to save images
    """
    json_list = []
    # Create images directory if it doesn't exist
    images_dir = os.path.join(save_dir, "images")
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
                sample_res[key] = os.path.join("images", image_filename)
            else:
                sample_res[key] = value
        json_list.append(sample_res)
    return json_list


def download_data(hf_dir, split=None, save_dir="./hf_data"):
    """
    Download dataset from HuggingFace repo and convert to JSON files.
    Images are saved locally in {save_dir}/images/.
    
    Args:
        hf_dir (str): HuggingFace repository name (e.g., 'weaverbirdllm/famma')
        split (str, optional): Specific split to download. If None, downloads all splits.
        save_dir (str): Directory to save the JSON files and images
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        if split:
            dataset = load_dataset(hf_dir, split=split, cache_dir=save_dir)
            json_list = convert_to_json_list(dataset, save_dir=save_dir)

            # Save to JSON file
            split_path = os.path.join(save_dir, f"{split}.json")
            save_json(json_list, split_path)
            print(f"Saved {split} split to {split_path}")

        else:
            dataset = load_dataset(hf_dir)
            for split_name in dataset.keys():
                json_list = convert_to_json_list(dataset[split_name], save_dir=save_dir)

                # Save to JSON file
                split_path = os.path.join(save_dir, f"{split_name}.json")
                save_json(json_list, split_path)
                print(f"Saved {split_name} split to {split_path}")

        print(f"\nDataset downloaded and saved to {save_dir}")
        print(f"Images are saved in {os.path.join(save_dir, 'images')}")
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
