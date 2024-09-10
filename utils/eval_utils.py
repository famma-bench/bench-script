import logging
import os

import pandas as pd
import time

from tqdm import tqdm
from models import get_client_fn
from utils.descriptive_utils import postprocess
from utils.prompt_utils import JudgePrompt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

score_dict = {
    "easy": 1,
    "medium":1.5,
    "hard": 2
}


def evaluate_multiple_choice(sample):
    """
    Evaluates whether the model's answer to a multiple-choice question matches any of the standard answers, 
    considering it correct if there is an exact match
    """
    gold_i = sample["answers"]
    pred_i = sample["model_answer"]
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_open_question(sample, model_api):
    """
    Evaluates if the model's open-ended answer is correct by checking if "incorrect" appears in the model's response
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

    input_prompt = JudgePrompt().get_prompt(
        context=context,
        question=question,
        model_answer=model_answer,
        model_explanation=model_explanation,
        answer=answer
    )
    model_output = model_api(input_prompt, images).lower()

    if "unable to answer" in model_output:
        return "unable to answer"
    if "incorrect" in model_output:
        return False
    if "correct" in model_output:
        return True

    return False


def prepare_dataset_for_evaluation(data_dir):
    """
    Read and return a list of DataFrames from all CSV files in the specified directory
    """
    result = []

    # Get all CSV files in the directory
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(
        data_dir) if file.endswith(".csv")]

    for file in csv_files:
        # Read each CSV file into a DataFrame and append to result list
        df = pd.read_csv(file)
        result.append(df)

    return result


def eval_ans(model_name, api_key, data_dir, save_dir):
    """
    Evaluates model performance on a dataset by calculating accuracy, real score, and normalized score, 
    then saves the results and evaluation data to specified directories
    """
    # Initialize API client
    Client = get_client_fn(model_name)
    model_api = Client(api_key, model_name=model_name, temperature=0.0)

    # Load all datasets from the data directory
    eval_datasets = prepare_dataset_for_evaluation(data_dir)

    for data_df in eval_datasets:
        data = data_df.to_dict(orient='index')
        # Initialize score-related variables
        total_count = len(data)
        correct_count = 0
        unable_to_answer_count = 0
        total_score = 0
        max_score = sum(score_dict.get(
            sample["topic_difficulty"], 0) for sample in data.values())

        # Get the context from the first sub_question
        first_sample = next(iter(data.values()))
        response_model_name = first_sample.get("model_name", "")

        logging.info(f"Calculating {response_model_name} score")

        for _, question_data in tqdm(data.items()):
            difficulty_score = score_dict.get(
                question_data["topic_difficulty"], 0)
            is_correct = (evaluate_multiple_choice(question_data)
                          if question_data["question_type"] == "multiple-choice"
                          else evaluate_open_question(question_data, model_api))

            # Ensure the correct status is saved
            question_data["is_correct"] = is_correct

            # Model is unable to answer, not a judge unable to answer
            if is_correct == "unable to answer":
                unable_to_answer_count += 1
            elif is_correct:
                correct_count += 1
                total_score += difficulty_score

            # Avoid hitting rate limits or overwhelming the API
            time.sleep(2)

        postprocess(response_model_name, save_dir, total_count,
                    correct_count, unable_to_answer_count, total_score, max_score, data)
