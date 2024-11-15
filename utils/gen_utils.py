import json
import os
import random
import re
import numpy as np
import pandas as pd
import logging
import time
import dictdatabase as DDB

from tqdm import tqdm
from models import get_client_fn
from utils.data_utils import format_options, read_json, remove_json_format, default_response
from utils.prompt_utils import MultipleChoiceQuestionPrompt, OpenQuestionPrompt
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_dataset_for_generation(data_dir, subset_name=None, question_ids=None):
    """
    Prepare the dataset by reading JSON files from a specified subset directory or the entire data directory,
    returning the dataset as a DataFrame. If question_ids are provided, filter the DataFrame to return only the valid questions.
    """
    # Create a list to store the results
    dataset_list = []

    # Traverse all files in the directory, including subdirectories
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Read JSON data from file
                data = read_json(file_path)
                # Append data to the dataset list
                dataset_list.extend(data)

    # Convert list to DataFrame
    dataset_df = pd.DataFrame(dataset_list)

    # If subset_name is provided, filter the DataFrame
    if subset_name:
        dataset_df = dataset_df[dataset_df['language'].str.capitalize(
        ) == subset_name.capitalize()]

    # If question_ids are provided, filter the DataFrame
    if question_ids:
        dataset_df = dataset_df[dataset_df['question_id'].isin(question_ids)]
        if dataset_df.empty:
            raise ValueError(
                "None of the provided question_ids were found in the dataset.")

    return dataset_df


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


def get_question_response(model, context, question_df, images, question_type="multiple-choice"):
    """
    Get answers for either multiple-choice or open-ended questions from the DataFrame.
    """
    sub_questions = question_df["question"].tolist()

    if question_type == "multiple-choice":
        input_prompt = MultipleChoiceQuestionPrompt().get_prompt(
            context=context, sub_questions=sub_questions)
    else:
        input_prompt = OpenQuestionPrompt().get_prompt(
            context=context, sub_questions=sub_questions)

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
    
    model_output = model.generate(msg)
    return model_output


def safe_parse_response(response):
    """
    Parse the response string as JSON or extracts data using regex if JSON parsing fails.
    """
    try:
        response = re.sub(r'\n', ' ', response)
        return json.loads(remove_json_format(response))
    except (json.JSONDecodeError, TypeError):
        # Regular expression to extract the data
        pattern = r'"?sub-question-(\d+)"?:\s*\{\s*"?answer-(\d+)"?:\s*"([^"]*)"\s*,?\s*"?explanation-(\d+)"?:\s*"([^"]*)"\s*\}'
        matches = re.findall(pattern, response)
        # Check if matches were found and create dict
        if matches:
            response = {
                f"sub-question-{match[0]}": {
                    f"answer-{match[0]}": match[2],
                    f"explanation-{match[0]}": match[4]
                }
                for match in matches
            }
            return response
        else:
            logging.warning(
                "Response is not valid JSON or is malformed. Response: %s", response)
        return {}


def process_response(model_name, data_df, response, is_multiple_choice=False):
    """
    Extracts relevant information from the model's response for both open-ended and multiple-choice questions,
    formats it, and updates the DataFrame with the model's answers and explanations.
    """
    response = safe_parse_response(response)
    data_df = data_df.reset_index(drop=False)

    # Prepare columns for answers and explanations
    data_df['model_answer'] = ""
    data_df['model_explanation'] = ""
    data_df['model_extract_answer'] = ""
    data_df['model_name'] = model_name

    for index in data_df.index:
        language = data_df.loc[index, 'language']
        response_entry = response.get(f"sub-question-{index + 1}", {})
        response_answer = response_entry.get(
            f"answer-{index + 1}", default_response.get(language))

        data_df.at[index, "model_answer"] = response_answer  
        data_df.at[index, "model_explanation"] = response_entry.get(
            f"explanation-{index + 1}", default_response.get(language))
        
        if is_multiple_choice:
            options = format_options(data_df.at[index, "options"])
            labels = options["labels"]
            options_dict = options["options_dict"]
            data_df.at[index, "model_extract_answer"] = extract_choice_from_response(
                response_answer, labels, options_dict)

    return data_df


def generate_answer_by_model(model, sub_question_set_df, target_db_name, key):
    """
    Generates model answer and explanation for a subset of questions, including both multiple-choice 
    and open-ended types.
    """
    # Get the context from the first sub_question
    context = sub_question_set_df.iloc[0].get("context", "")

   # Collect unique images
    images = []
    last_used_index = 0
    for _, row in sub_question_set_df.iterrows():
        for i in range(1, 8):
            image_key = f"image_{i}"
            tag = f"<image_{i}>"
            if tag in row['context'] and pd.notna(row.get(image_key)):
                if row[image_key] not in images:
                    images.append(row[image_key])
                    last_used_index = i
        # Traverse the remaining image fields starting from the last used index + 1
        # special case to note
        # image place holder in sub-questions should not have duplicates
        # i.e, sub-question-2: "xxxx <image_3>", sub-question-3: "xxxx <image_4>"
        for i in range(last_used_index + 1, 8):
            image_key = f"image_{i}"
            if pd.notna(row.get(image_key)):
                images.append(row[image_key])

    # Process multiple-choice_df questions
    multiple_choice_df = sub_question_set_df.query(
        'question_type == "multiple-choice"').copy()
    if not multiple_choice_df.empty:
        # Format the 'question' column to follow the structure:
        # "question: xxx\n options: A. xxx\n B. xxx...". Applies this formatting to each row in the DataFrame.
        def format_question(row):
            if isinstance(row['options'], str):
                options = format_options(row['options'])
                return f"{row['question'].strip()}\noptions:\n{options['formatted_options']}"
            else:
                return row['question']

        multiple_choice_df['question'] = multiple_choice_df.apply(
            format_question, axis=1)
        multiple_choice_response = get_question_response(
            model, context, multiple_choice_df, images, question_type="multiple-choice")
        multiple_choice_result = process_response(
            model.model_name, multiple_choice_df, multiple_choice_response, is_multiple_choice=True)
    else:
        multiple_choice_result = pd.DataFrame()

    # Process open-ended questions
    open_question_df = sub_question_set_df.query(
        'question_type == "open question"').copy()
    if not open_question_df.empty:
        open_question_response = get_question_response(
            model, context, open_question_df, images, question_type="open question")
        open_question_result = process_response(
            model.model_name, open_question_df, open_question_response, is_multiple_choice=False)
    else:
        open_question_result = pd.DataFrame()

    # Save the current sub_question_set answers to the local DDB to prevent data loss
    ddb_samples = pd.concat([multiple_choice_result, open_question_result], ignore_index=True)[
        ["model_answer", "model_extract_answer", "model_explanation", "model_name"]]
    ddb_samples_dict = ddb_samples.to_dict(orient='records')

    with DDB.at(target_db_name).session() as (sess, obj):
        obj[key] = [{"model_answer": row["model_answer"], "model_extract_answer": row["model_extract_answer"], "model_explanation": row["model_explanation"], "model_name": row["model_name"]}
                    for row in ddb_samples_dict]
        sess.write()

    output_samples_subset = pd.concat(
        [multiple_choice_result, open_question_result], ignore_index=True, sort=False)

    return output_samples_subset


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
    logging.info("Saved output samples to %s", output_file_path)


def generate_ans(config_dir, data_dir, save_dir, question_ids):
    """
    Generate dataset answers with a model, save the output as JSON files.
    """
    # Load configuration from YAML file
    model_config = Config.build_from_yaml_file(config_dir)

    # Build the LLM model
    model = LLM.build_from_config(model_config)

    model_name = model.model_name

    # Initialize the DDB
    target_db_name = f'{model_name}_DDB'
    init_db = DDB.at(target_db_name).read()
    if init_db is None:
        DDB.at(target_db_name).create()

    # 2. Load the dataset
    dataset = prepare_dataset_for_generation(
        data_dir, question_ids)

    # 3. Process dataset and feed into the model
    output_samples = pd.DataFrame()
    grouped = dataset.groupby(['language', 'main_question_id'])

    for (language, main_question_id), sub_question_set_df in tqdm(grouped):
        key = str(f"{language}_{main_question_id}")
        try:
            # Attempt to read existing data from the DDB
            existing_data = pd.DataFrame(DDB.at(target_db_name, key=key).read()) if DDB.at(
                target_db_name, key=key).exists() else pd.DataFrame()

            if not existing_data.empty:
                existing_data = existing_data[[
                    "model_answer", "model_extract_answer", "model_explanation", "model_name"]].reset_index(drop=True)
                sub_question_set_df = sub_question_set_df.reset_index(
                    drop=True).copy()
                sub_question_set_df.loc[:, [
                    "model_answer", "model_extract_answer", "model_explanation", "model_name"]] = existing_data
                output_samples = pd.concat(
                    [output_samples, sub_question_set_df], ignore_index=True)
            else:
                output_samples_subset = generate_answer_by_model(
                    model, sub_question_set_df, target_db_name, key
                )
                output_samples = pd.concat(
                    [output_samples, output_samples_subset], ignore_index=True, sort=False)

        except Exception as e:
            logging.error(
                "Error processing main_question_id %s: %s", main_question_id, str(e))
            time.sleep(2)
            continue

        # Avoid hitting rate limits or overwhelming the API
        time.sleep(2)

    logging.info('Generation complete')  # logger
    save_output_samples(output_samples, model_name, save_dir)
