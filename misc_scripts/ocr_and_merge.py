import os
import pandas as pd
from paddleocr import PaddleOCR
from easyllm_kit.utils import read_json, save_json, get_logger
from tqdm import tqdm

logger = get_logger('ocr_and_merge')

def get_paddle_language(language):
    """
    Map dataset language to PaddleOCR language code.
    
    Args:
        language: Language from the dataset (english, chinese, french)
        
    Returns:
        PaddleOCR language code
    """
    language_map = {
        'english': 'en',
        'chinese': 'ch',
        'french': 'fr'
    }
    return language_map.get(language, 'en')

def perform_ocr(image_path, ocr_model):
    """
    Perform OCR on the given image and return the extracted text.
    
    Args:
        image_path: Path to the image file.
        ocr_model: PaddleOCR model instance
        
    Returns:
        Extracted text from the image.
    """
    try:
        result = ocr_model.ocr(image_path, cls=True)
        
        # Extract text from OCR results
        texts = []
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                if len(line) >= 2 and line[1] and len(line[1]) >= 1:
                    texts.append(line[1][0])
        
        return " ".join(texts)
    except Exception as e:
        logger.error(f"Error performing OCR on {image_path}: {e}")
        return ""

def merge_ocr_text_into_dataset(json_path, image_base_path, output_csv_path):
    """
    Process all questions, perform OCR on images, and merge text into context.
    Output the result as a CSV file.
    
    Args:
        json_path: Path to the JSON file.
        image_base_path: Base path where images are stored.
        output_csv_path: Path to save the updated CSV file.
    """
    # Read the JSON data
    data = read_json(json_path)
    
    # Create a list to store all processed questions
    all_questions = []
    
    # Initialize OCR models for different languages
    ocr_models = {
        'english': PaddleOCR(use_angle_cls=True, lang='en'),
        'chinese': PaddleOCR(use_angle_cls=True, lang='ch'),
        'french': PaddleOCR(use_angle_cls=True, lang='fr')
    }
    
    # Group questions by language and main_question_id
    questions_by_group = {}
    for question_data in data:
        language = question_data.get('language', 'english')
        main_question_id = question_data.get('main_question_id')
            
        group_key = f"{language}_{main_question_id}"
        if group_key not in questions_by_group:
            questions_by_group[group_key] = []
        
        questions_by_group[group_key].append(question_data)
    
    # Process each group of questions
    for group_key, question_group in questions_by_group.items():
        language = question_group[0].get('language', 'english')
        ocr_model = ocr_models.get(language, ocr_models['english'])
        
        # Get all unique images from the first subquestion
        first_subquestion = None
        for question in question_group:
            if question.get('sub_question_id') == '1' or question.get('sub_question_id') == 1:
                first_subquestion = question
                break
        
        if not first_subquestion:
            first_subquestion = question_group[0]  # Fallback to first question if no sub_question_id 1
            
        # Extract OCR text from images in the first subquestion
        ocr_texts = []
        for i in range(1, 8):  # Assuming up to 7 images
            image_key = f'image_{i}'
            image_path = first_subquestion.get(image_key)
            
            if image_path and image_path != "None":
                full_image_path = os.path.join(image_base_path, image_path)
                if os.path.exists(full_image_path):
                    ocr_text = perform_ocr(full_image_path, ocr_model)
                    if ocr_text:
                        ocr_texts.append(f"image_{i} ocr text: {ocr_text}")
        
        # If we have OCR text, update only the first subquestion's context
        if ocr_texts:
            combined_ocr = "/n".join(ocr_texts)
            
            for question in question_group:
                if question.get('sub_question_id') == '1' or question.get('sub_question_id') == 1:
                    original_context = question.get('context', '')
                    if original_context == 'nan' or pd.isna(original_context) or not original_context:
                        question['context'] = f"<ocr>{combined_ocr}</ocr>"
                    else:
                        question['context'] = f"{original_context} /n <ocr>{combined_ocr}</ocr>"
        
        # Add all questions from this group to our list
        all_questions.extend(question_group)
        logger.info(f'{group_key} processed')
    
    # Convert to DataFrame
    df = pd.DataFrame(all_questions)
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False, header=True)
    logger.info(f"Updated dataset saved to {output_csv_path}")
    
    # Also save as JSON for reference
    json_output_path = output_csv_path.replace('.csv', '.json')
    save_json(data, json_output_path)
    logger.info(f"Updated JSON saved to {json_output_path}")

if __name__ == "__main__":
    json_path = "../hf_data/release_basic.json"
    image_base_path = "../hf_data/"
    output_csv_path = "../ddb_storage/release_basic_txt.csv"
    
    merge_ocr_text_into_dataset(json_path, image_base_path, output_csv_path) 