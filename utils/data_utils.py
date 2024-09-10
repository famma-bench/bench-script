import base64
import json
import os
import ast
import string
from PIL import Image
from io import BytesIO
from datasets import load_dataset

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


def save_json(filename, ds):
    """ 
    Save data as JSON to the specified path
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    return json_data


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
    Convert data in Dataset format to list format
    """
    json_list = []
    for sample in dataset:
        sample_dict = dict(sample)
        for key, value in sample_dict.items():
            if isinstance(value, Image.Image):
                sample_dict[key] = image_to_base64(value)
        json_list.append(sample_dict)
    return json_list


def download_data(hf_dir, subset_name, save_dir):
    """
    Download from huggingface repo and convert all data files into json files
    """
    if not subset_name:
        subsets = ["English", "Chinese", "French"]
    else:
        subsets = [subset_name]

    splits = ["validation", "test"]

    for subset in subsets:
        dataset = load_dataset(hf_dir, subset)

        for split in splits:
            json_list = convert_to_json_list(dataset[split])

            split_path = os.path.join(save_dir, subset, f"{split}.json")
            os.makedirs(os.path.dirname(split_path), exist_ok=True)

            save_json(split_path, json_list)
            print(f"Saved {split} split of {subset} subset to {split_path}")

    return
