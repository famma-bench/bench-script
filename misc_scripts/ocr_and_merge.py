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
    
    # Process each question
    for main_key, main_value in tqdm(data.items(), desc="Processing main questions"):
        for question_id, question_data in main_value.items():
            # Get the language for OCR
            language = question_data.get('language', 'english')
            ocr_model = ocr_models.get(language, ocr_models['english'])
            
            # Initialize OCR text for this question
            ocr_texts = []
            
            # Check if there are images to process
            for i in range(1, 8):  # Assuming up to 7 images
                image_key = f'image_{i}'
                image_path = question_data.get(image_key)
                
                if image_path and image_path != "None":
                    full_image_path = os.path.join(image_base_path, image_path)
                    if os.path.exists(full_image_path):
                        ocr_text = perform_ocr(full_image_path, ocr_model)
                        if ocr_text:
                            ocr_texts.append(ocr_text)
            
            # Merge OCR text into the context
            if ocr_texts:
                combined_ocr = " ".join(ocr_texts)
                original_context = question_data.get('context', '')
                if original_context == 'nan' or not original_context:
                    question_data['context'] = f"<ocr>{combined_ocr}</ocr>"
                else:
                    question_data['context'] = f"{original_context} <ocr>{combined_ocr}</ocr>"
            
            # Add the processed question to our list
            all_questions.append(question_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_questions)
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Updated dataset saved to {output_csv_path}")
    
    # Also save as JSON for reference
    json_output_path = output_csv_path.replace('.csv', '.json')
    save_json(data, json_output_path)
    logger.info(f"Updated JSON saved to {json_output_path}")

if __name__ == "__main__":
    json_path = "../ddb_storage/o1_ans_release_v2406_with_arithmetic_flag.json"
    image_base_path = "../images_release_v2406"
    output_csv_path = "../ddb_storage/o1_ans_release_v2406_with_ocr.csv"
    
    merge_ocr_text_into_dataset(json_path, image_base_path, output_csv_path) 