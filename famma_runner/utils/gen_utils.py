import json
import os
import random
import re
import base64
from easyllm_kit.utils import get_logger, extract_json_from_text
from typing import Optional, List, Union, Dict
from pathlib import Path

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


def generate_response_from_llm(model, input_prompt, images=None):
    """
    Get answers for either multiple-choice or open-ended questions from the DataFrame.
    """
    if model.model_name in ['gpt4o', 'claude_35_sonnet', 'gemini-1.5']:
        # we use litellm api for openai / claude / gemini
        msg = [{
            "type": "text",
            "text": input_prompt,
        }
        ]
        if images is not None:
            for image in images:
                msg.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                })
        model_output = model.generate(msg)
    elif model.model_name in ['qwen', 'qwen_vl']:
        # we use openai api for qwen
        msg = [{
            "type": "text",
            "text": input_prompt,
        }]
        if images is not None:
            for image in images:
                msg.append({
                    "type": "image_url",
                    "image_url": f"data:image/base64,{image}"
                })
        model_output = model.generate(msg)
        model_output = json.loads(model_output)["choices"][0]["message"]["content"]
    elif model.model_name in ['custom_llm']:
        images = "\n".join(images)
        model_output = model.generate(input_prompt + "\n" + images)
    else:
        # message = _prepare_litellm_message(input_prompt, images)
        # return model.generate(message)
        return None


def safe_parse_response(response_text_all, question_id_list,model_name):
    """
    Parse the response string as JSON or extracts data using regex if JSON parsing fails.
    Args:
        response_text: The text response from the model
        question_id_list: List of question IDs to look for
    Returns:
        Dictionary mapping question IDs to their answers and explanations
    """
    # First try to parse as JSON
    if model_name in ['custom_llm']: # TODO:逻辑关系不对
        response_dict_all =json.loads(response_text_all) # TODO:对于r1，这里是custum_model，先load进所有内容，response_text
        response_text = response_dict_all['content']
        reasoning_text = response_dict_all['reasoning']
    else:
        response_text = response_text_all
        reasoning_text = ""

    try:
        response_dict = json.loads(response_text)
    except json.JSONDecodeError:
        response_dict = extract_json_from_text(response_text)
        response_dict['reasoning'] = reasoning_text

    if response_dict.get('result', None) == 'error parsing':
        # If JSON parsing fails, use regex
        logger.info('Start to using regex to extract answers.')
        parsed_response = {}

        for question_id in question_id_list:
            # Pattern to match: "q1": {"answer": "some answer", "explanation": "some explanation"}
            pattern = rf'"{question_id}"\s*:\s*\{{\s*"answer"\s*:\s*"(.*?)"\s*,\s*"explanation"\s*:\s*"(.*?)"\s*\}}'

            match = re.search(pattern, response_text)
            if match:
                answer, explanation = match.groups()
                parsed_response[question_id] = {
                    "answer": answer.strip(),
                    "explanation": explanation.strip()
                }
            else:
                logger.warning(f"Could not find match for question {question_id} in response")
                parsed_response[question_id] = {
                    "answer": "",
                    "explanation": ""
                }

        if not parsed_response:
            logger.warning(
                "Could not parse any responses. Response text: %s", response_text)

        return parsed_response
    else:
        return response_dict


def collect_images_from_first_subquestion(sub_question_set_df, parent_dir, model_name):
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
                if model_name in ['custom_llm']:#TODO:model_name,custom_llm,逻辑关系不对
                    ocr_text = paddle_ocr(image_dir)
                    images.append(ocr_text)
                else:
                    with open(image_dir, 'rb') as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        images.append(encoded_string)
    return images

#新增OCR处理函数
from paddleocr import PaddleOCR
def paddle_ocr(img_path):
    ocr_model = PaddleOCR(show_log=False)
    result = ocr_model.ocr(img_path)
    return '\n'.join([line[1][0] for res in result for line in res])
