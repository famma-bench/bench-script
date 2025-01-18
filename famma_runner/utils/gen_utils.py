import json
import os
import random
import re
import base64
import numpy as np

from easyllm_kit.utils.io_utils import write_to_database
from easyllm_kit.utils import get_logger, extract_json_from_text

logger = get_logger('famma', 'famma.log')


def extract_choice_from_response(response, all_choices, choice_ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match
    index_ans = True
    candidates = []

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 0 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 0:
        for index, ans in choice_ans.items():
            ans_pattern = f" {ans.strip()} "
            if ans_pattern.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(choice_ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def generate_response_from_llm(model, input_prompt, images):
    """
    Get answers for either multiple-choice or open-ended questions from the DataFrame.
    """
    # we use litellm api for openai / claude / gemini
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

    model_output = model.generate(msg)

    output = safe_parse_response(model_output)

    return output


def safe_parse_response(response):
    """
    Parse the response string as JSON or extracts data using regex if JSON parsing fails.
    """
    try:
        return extract_json_from_text(response)
    except (json.JSONDecodeError, TypeError):
        # Revised regular expression to extract the data
        pattern = r'"?sub-question-(\d+)"?:\s*\{\s*"?answer"?:\s*"([^"]*)"\s*,?\s*"?explanation"?:\s*"([^"]*)"\s*\}'
        matches = re.findall(pattern, response)
        # Check if matches were found and create dict
        if matches:
            response = {
                f"sub-question-{match[0]}": {
                    "answer": match[1],
                    "explanation": match[2]
                }
                for match in matches
            }
            return response
        else:
            logger.warning(
                "Response is not valid JSON or is malformed. Response: %s", response)
        return {}


def collect_images_from_first_subquestion(sub_question_set_df, parent_dir):
    """
    Collects unique images from the first sub-question in the question set and returns them as a list.
    """
    images = []
    sub_question_set_df.sort_values(by='sub_question_id', inplace=True)
    if not sub_question_set_df.empty:
        first_row = sub_question_set_df.iloc[0]
        # Check images referenced in context
        for i in range(1, 8):
            image_key = f"image_{i}"
            if first_row.get(image_key) is not None and first_row[image_key] != 'None':
                image_dir = os.path.join(parent_dir, first_row[image_key])
                # encode image to base64
                with open(image_dir, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    images.append(encoded_string)

    return images


def save_to_local_ddb(results_df, target_db_name, key):
    """
    Save the current results to the local DictDatabase.
    """
    columns = ["model_answer", "model_extract_answer", "model_explanation", "model_name"]
    ddb_samples = results_df[columns]
    input_dict = ddb_samples.to_dict(orient='records')
    write_to_database(target_db_name, key, input_dict)


def save_output_samples(output_samples, model_name, save_dir):
    """
    Save output samples to CSV files.
    """
    folder_name = f"{model_name}_model_answers"
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    output_file_name = f"{folder_name}.csv"
    output_file_path = os.path.join(folder_path, output_file_name)

    # Save the DataFrame to a CSV file
    output_samples.to_csv(output_file_path, index=False,
                          encoding='utf_8_sig', header=True)
    logger.info("Saved output samples to %s", output_file_path)
