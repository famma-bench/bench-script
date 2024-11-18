import os

import pandas as pd

from tqdm import tqdm
from utils.descriptive_utils import postprocess
from utils.prompt_utils import JudgePrompt

from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json, save_json
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config

logger = get_logger('famma', 'famma.log')


def evaluate_multiple_choice(sample):
    """
    Evaluates whether the model's answer to a multiple-choice question matches any of the standard answers, 
    considering it correct if there is an exact match
    """
    gold_i = sample["answers"]
    pred_i = sample["model_extract_answer"]
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


def evaluate_open_question(sample, model):
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
    # we use litellm api for all the models
    msg = [{
        "type": "text",
        "text": input_prompt,
    }
    ]
    for image in images:
        msg.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}"
            },
        })

    model_output = model.generate(msg).lower()

    if "unable to answer" in model_output:
        return "unable to answer"
    if "incorrect" in model_output:
        return False
    if "correct" in model_output:
        return True

    return False


def prepare_dataset_for_evaluation(pred_json_dir, gold_json_dir):
    """
    Read and return a list of DataFrames from all CSV files in the specified directory
    """

    # read all json files in the directory
    pred_jsons = read_json(pred_json_dir)
    gold_jsons = read_json(gold_json_dir)

    # make gold_jsons a dataframe
    gold_df = pd.DataFrame(gold_jsons)
    gold_df['model_name'] = None
    gold_df['model_extract_answer'] = None
    gold_df['model_answer'] = None
    gold_df['model_explanation'] = None

    # look up main_question_id and sub_question_id in the gold json file
    # add answers to the pred_jsons
    for k, pred_json in pred_jsons.items():
        language, main_question_id = k.split("_")
        # find question_id in gold_jsons given the main_question_id and language
        question_set = gold_df[
            (gold_df["main_question_id"] == int(main_question_id)) & (gold_df["language"] == language)]
        question_ids = question_set["question_id"].sort_values().tolist()
        # insert answers in pred_json to gold_jsons, with each answer in the order of question_ids
        for idx, question_id in enumerate(question_ids):
            mask = gold_df["question_id"] == question_id
            for key, value in pred_json[idx].items():
                gold_df.loc[mask, key] = value
    return gold_df


def eval_ans(config_dir, gen_data_dir, gold_data_dir, save_dir, output_db_name='eval'):
    """
    Evaluates model performance on a dataset by calculating accuracy, real score, and normalized score, 
    then saves the results and evaluation data to specified directories
    """
    # Load configuration from YAML file
    model_config = Config.build_from_yaml_file(config_dir)

    # Build the LLM model
    model = LLM.build_from_config(model_config)

    # Load all datasets from the data directory
    eval_df = prepare_dataset_for_evaluation(gen_data_dir, gold_data_dir)

    database = initialize_database(output_db=output_db_name)

    # Initialize score-related variables
    total_count = len(eval_df)
    correct_count = 0
    unable_to_answer_count = 0
    total_score = 0

    # Get the context from the first sub_question
    response_model_name = eval_df.loc[0, 'model_name']

    logger.info(f"Calculating {response_model_name} score")

    for _, question_data in eval_df.iterrows():
        is_correct = (evaluate_multiple_choice(question_data)
                      if question_data["question_type"] == "multiple-choice"
                      else evaluate_open_question(question_data, model))

        # Ensure the correct status is saved
        question_data["is_correct"] = is_correct

        # Model is unable to answer, not a judge unable to answer
        if is_correct == "unable to answer":
            unable_to_answer_count += 1
        elif is_correct:
            correct_count += 1

        key = question_data['question_id']
        if key not in database:
            # convert question_data to dict
            input_dict = question_data.to_dict()
            write_to_database(output_db_name, key, input_dict)

    postprocess(response_model_name, save_dir, total_count,
                correct_count, unable_to_answer_count, total_score, eval_df)
