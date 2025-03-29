import json
import os
import re
import base64
from easyllm_kit.utils import get_logger, extract_json_from_text
from typing import Optional, List, Union, Dict
from pathlib import Path
import json_repair

logger = get_logger('famma', 'famma.log')


def _prepare_litellm_message(prompt: str, images: Optional[List[str]] = None) -> List[Dict]:
    """Helper function to prepare message for LiteLLM-based models."""
    message = [{"type": "text", "text": prompt}]
    if images:
        message.extend([
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            } for img in images
        ])
    return message


def _handle_ocr(ocr_model, images: List[str], prompt: str) -> str:
    """Process images with OCR and append results to prompt."""
    for i, img in enumerate(images):
        # Convert base64 to bytes and save as temporary jpg
        img_bytes = base64.b64decode(img)
        temp_img_path = f"temp_img_{i}.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(img_bytes)

        # Perform OCR on jpg file
        ocr_result = ocr_model.ocr(temp_img_path)
        ocr_text = f'\n <image_{i + 1}> OCR result: '.join([line[1][0] for res in ocr_result for line in res])
        prompt = f"{prompt}\n{ocr_text}"

        # Clean up temporary file
        os.remove(temp_img_path)
    return prompt


def generate_response_from_llm(
        model,
        input_prompt: str,
        images: Optional[Union[List[str], List[Path]]] = None,
        use_ocr: bool = False,
        ocr_model=None
) -> str:
    """
    Generate responses from various LLM models with optional image input and OCR processing.

    Args:
        model: The language model instance to use for generation
        input_prompt (str): The text prompt to send to the model
        images (Optional[Union[List[str], List[Path]]]): List of image paths or base64 encoded images
        use_ocr (bool): Whether to use OCR processing on the images
        ocr_model: OCR model instance (required if use_ocr is True)

    Returns:
        str: The generated response from the model

    Raises:
        ValueError: If OCR is requested but no OCR model is provided
        ValueError: If the model name is not supported
        NotImplementedError: If the model type is not implemented
    """
    if not hasattr(model, 'model_name'):
        raise ValueError("Model must have 'model_name' attribute")

    if use_ocr:
        if ocr_model is None:
            raise ValueError("ocr_model is required when use_ocr is True")
        if not images:
            raise ValueError("Images are required when use_ocr is True")
        input_prompt = _handle_ocr(ocr_model, images, input_prompt)
        return model.generate(input_prompt)

    if model.model_name in ['qwen', 'qwen_vl']:
        response = model.generate(input_prompt, image_dir=images)
        try:
            return json_repair.loads(response)["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Qwen model response: {e}")
    elif model.model_name == 'gemini' and not model.model_config.use_litellm_api:
        message = [input_prompt, images]
        return model.generate(message)
    else:
        message = _prepare_litellm_message(input_prompt, images)
        return model.generate(message)


def safe_parse_response(response, question_id_list):
    """
    Parse the response string as JSON or extracts data using regex if JSON parsing fails.
    Args:
        response: The text response from the model, can be a string or a dictionary
        question_id_list: List of question IDs to look for
    Returns:
        Dictionary mapping question IDs to their answers and explanations
    """
    # Initialize response dictionary
    response_dict = {}

    # If reasoning is attached, response should be a dictionary
    if isinstance(response, dict):
        response_text = response.get('content', '')
        reasoning = response.get('reasoning_content', '')
        response_dict['reasoning'] = reasoning
    else:
        response_text = response
    
    if response_text == '':
        response_dict['result'] = 'error parsing'

    # Try to parse as JSON
    try:
        parsed_json = json_repair.loads(response_text)
        response_dict.update(parsed_json)
    except json.JSONDecodeError:
        try:
            extracted_json = extract_json_from_text(response_text)
            response_dict.update(extracted_json)
        except Exception as e:
            logger.warning(f"Error extracting JSON: {e}")
            response_dict['result'] = 'error parsing'
    
    # If JSON parsing fails, use regex
    if response_dict.get('result', None) == 'error parsing':
        logger.info('Starting to use regex to extract answers.')
        parsed_response = {}

        for idx, question_id in enumerate(question_id_list):
            # Pattern to match: "q1": {"answer": "some answer", "explanation": "some explanation"}
            pattern = rf'"{question_id}"\s*:\s*\{{\s*"answer"\s*:\s*"(.*?)"\s*,\s*"explanation"\s*:\s*"(.*?)"\s*\}}'

            match = re.search(pattern, response_text)
            if match:
                answer, explanation = match.groups()
                parsed_response[question_id] = {
                    "answer": answer.strip(),
                    "explanation": explanation.strip(),
                }
            else:
                logger.warning(f"Could not find match for question {question_id} in response")
                if idx == 0:
                    logger.warning(f"Save the unparsed text in 'explanation' for question {question_id} in response")
                    parsed_response[question_id] = {
                        "answer": "",
                        "explanation": response_text
                    }
                else:
                    parsed_response[question_id] = {
                        "answer": "",
                        "explanation": ""
                    }

        if not parsed_response:
            logger.warning(
                "Could not parse any responses. Response text: %s", response_text)

        return parsed_response
        
    return response_dict


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
