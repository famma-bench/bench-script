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


default_response = {
    "english": "I don't know",
    "chinese": "我不知道",
    "french": "Je ne sais pas"
}


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


def convert_to_json_list(dataset):
    """
    Convert data in Dataset format to list format.
    Handles PIL Image objects by converting them to base64.
    """
    json_list = []
    for sample in dataset:
        sample_dict = dict(sample)
        for key, value in sample_dict.items():
            if isinstance(value, (Image.Image, JpegImageFile, PngImageFile)):
                # Convert any PIL Image type to base64
                sample_dict[key] = image_to_base64(value)
        json_list.append(sample_dict)
    return json_list


def download_data(hf_dir, split=None, save_dir="./hf_data"):
    """
    Download dataset from HuggingFace repo and convert to JSON files.
    
    Args:
        hf_dir (str): HuggingFace repository name (e.g., 'weaverbirdllm/famma')
        split (str, optional): Specific split to download. If None, downloads all splits.
        save_dir (str): Directory to save the JSON files
        
    Example:
        >>> download_data('weaverbirdllm/famma', split='release_v2406')
        >>> download_data('weaverbirdllm/famma')  # downloads all splits
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        if split:
            # Download specific split
            # hf_dir = 'mmmu/mmmu'
            # split = 'dev'
            dataset = load_dataset(hf_dir, split=split, cache_dir=save_dir)
            json_list = convert_to_json_list(dataset)
            
            # Save to JSON file
            split_path = os.path.join(save_dir, f"{split}.json")
            save_json(split_path, json_list)
            print(f"Saved {split} split to {split_path}")
            
        else:
            # Download all splits
            dataset = load_dataset(hf_dir)
            for split_name in dataset.keys():
                json_list = convert_to_json_list(dataset[split_name])
                
                # Save to JSON file
                split_path = os.path.join(save_dir, f"{split_name}.json")
                save_json(split_path, json_list)
                print(f"Saved {split_name} split to {split_path}")
        
        print(f"\nDataset downloaded and saved to {save_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False
