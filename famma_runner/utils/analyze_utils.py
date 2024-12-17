import os
import pandas as pd

from models import get_client_fn
from utils.data_utils import read_json, save_json
from utils.prompt_utils import AnalyzePrompt
from collections import defaultdict
from tqdm import tqdm


def get_error_type(sample, model_api):
    """
    Get the error type of a sample.
    """
    context = sample["context"]
    question = sample["question"]
    model_answer = sample["model_answer"]
    model_explanation = sample["model_explanation"]
    answer = sample["answers"]

    images = []
    last_used_index = 0
    # Collect unique images from the samples
    for i in range(1, 8):
        image_key = f"image_{i}"
        tag = f"<image_{i}>"
        if tag in sample['context'] and pd.notna(sample.get(image_key)):
            if sample[image_key] not in images:
                images.append(sample[image_key])
                last_used_index = i
    for i in range(last_used_index + 1, 8):
        image_key = f"image_{i}"
        if pd.notna(sample.get(image_key)):
            images.append(sample[image_key])

    input_prompt = AnalyzePrompt().get_prompt(
        context=context,
        question=question,
        model_answer=model_answer,
        model_explanation=model_explanation,
        answer=answer
    )
    model_output = model_api(input_prompt, images).lower()

    return model_output


def prepare_dataset_for_analysis(data_dir):
    """
    Prepare the dataset by reading JSON files named 'correction_result.json' from the specified directory. 
    Filter and return the data where 'is_correct' is false.
    """
    file_path = os.path.join(data_dir, 'correction_result.json')
    data = read_json(file_path)

    incorrect_data = {key: value for key,
                      value in data.items() if value.get('is_correct') == False}

    return incorrect_data


def analyze_incorrect_ans(model_name, api_key, data_dir, save_dir):
    """
    Evaluates model performance by analyzing the error types in the incorrect answers dataset,
    calculates error type counts and percentages, and saves the results to a specified directory.
    """
    # Initialize API client
    Client = get_client_fn(model_name)
    model_api = Client(api_key, model_name=model_name, temperature=0.0)

    # Load incorrect datasets from the data directory
    incorrect_data = prepare_dataset_for_analysis(data_dir)

    # Initialize counters for error types
    error_type_counts = defaultdict(int)

    # Analyze each incorrect data entry
    for key, value in tqdm(incorrect_data.items()):
        error_type = get_error_type(value, model_api).replace(' ', '_').lower()
        error_type_counts[error_type] += 1

    # Calculate total number of incorrect entries
    total_incorrect = sum(error_type_counts.values())

    # Calculate percentage of each error type
    error_type_percentages = {error_type: (
        count / total_incorrect) * 100 for error_type, count in error_type_counts.items()}

    # Prepare result data
    result_data = {
        'error_counts': dict(error_type_counts),
        'error_percentages': error_type_percentages,
    }

    # Save result data to a JSON file
    save_path = os.path.join(save_dir, 'error_analysis_results.json')
    save_json(save_path, result_data)

    print(f"Error analysis results saved to {save_path}")
