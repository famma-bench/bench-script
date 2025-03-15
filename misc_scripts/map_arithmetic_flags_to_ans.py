import os
from easyllm_kit.utils import read_json, save_json, get_logger
from tqdm import tqdm

logger = get_logger('map_arithmetic_flags_to_ans', 'map_arithmetic_flags_to_ans.log')

def update_ans_with_arithmetic_flag(dataset_dir, old_ans_dir, output_dir):
    """
    Update answer database with arithmetic flags from the dataset.
    
    Args:
        dataset_dir: Path to the dataset JSON file
        old_ans_dir: Path to the answer database JSON file
        output_dir: Path to the output JSON file
    """
    # Read the dataset and answer files
    dataset_dict = read_json(dataset_dir)
    ans_dict = read_json(old_ans_dir)
    
    # Create a mapping of question_ids to arithmetic flags
    arithmetic_map = {}
    for item in dataset_dict:
        question_id = item.get('question_id')
        is_arithmetic = item.get('is_arithmetic', False)
        arithmetic_map[question_id] = is_arithmetic
    
    # Update the answer database
    updated_count = 0
    for question_id, value_dict in ans_dict.items():
        if question_id in arithmetic_map:
            value_dict['is_arithmetic'] = arithmetic_map[question_id]
            updated_count += 1
        else:
            logger.warning(f"Question ID {question_id} not found in dataset")

    logger.info(f"Updated {updated_count} questions with arithmetic flags")
    
    # Save the updated answer database
    save_json(ans_dict, output_dir)
    logger.info(f"Saved updated answer database to {output_dir}")

if __name__ == "__main__":
    # You can modify these paths as needed
    dataset_dir = "../hf_data/release_v2406.json"
    old_ans_dir = "../ddb_storage/deepseek_r1_evaluated_by_gemini.json"

    output_dir = "../ddb_storage/r1_v2406_evaluated_by_gemini_with_arithmetic_flag.json"

    update_ans_with_arithmetic_flag(dataset_dir, old_ans_dir, output_dir)
