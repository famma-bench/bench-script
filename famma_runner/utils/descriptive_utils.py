import os
import tiktoken

from collections import defaultdict
from easyllm_kit.utils import save_json, read_json


def get_first_sub_question_id(question_id):
    """
    Get the first sub-question id from the question id
    """
    question_id_parts = question_id.split("_")
    question_id_parts[2] = "1"
    return "_".join(question_id_parts)

def get_context(data, question_id):
    """
    Get the context of a question from the database
    """
    question_id = get_first_sub_question_id(question_id)

    for item in data:
        if item["question_id"] == question_id:
            return item["context"]
    raise ValueError(f"Context not found for question id: {question_id}")



def get_dataset_statistics(data_dir):
    """
    Calculates and saves dataset statistics
    """
    # Initialize the dictionary for statistical data
    stats = {
        "total_count": 0,
        "total_main_question_count": 0,
        "unique_main_question_ids": set(),
        "language_count": defaultdict(int),
        "question_type_count": defaultdict(int),
        "image_type_count": defaultdict(int),
        "image_type_set": set(),
        "subfield_count": defaultdict(int),
        "subfield_set": set(),
        "topic_difficulty_count": defaultdict(int),
        "explanation_count": 0,
        "multiple_images_count": 0,
        "arithmetic_count": 0,
        "arithmetic_by_language": defaultdict(int),
        "arithmetic_by_difficulty": defaultdict(int),
        "token_counts": defaultdict(lambda: {"total": 0, "count": 0, "average_total": 0, "average_question": 0, "average_context": 0}),
        "subfield_difficulty_count": defaultdict(lambda: defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0})),
        "total_token_sum": {"total": 0, "count": 0, "average_total": 0, "average_question": 0, "average_context": 0},
        "language_difficulty_count": defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0}),
        "question_type_difficulty_count": defaultdict(
            lambda: {"easy": 0, "medium": 0, "hard": 0}
        ),
        
        # New fields for token counts by category
        "token_counts_by_question_type": defaultdict(lambda: {"total": 0, "count": 0, "average_total": 0, "average_question": 0, "average_context": 0}),
        "token_counts_by_difficulty": defaultdict(lambda: {"total": 0, "count": 0, "average_total": 0, "average_question": 0, "average_context": 0}),
        "token_counts_by_subfield": defaultdict(lambda: {"total": 0, "count": 0, "average_total": 0, "average_question": 0, "average_context": 0}),
    }

    # read the json files in the data_dir
    data = read_json(data_dir)

    for item in data:
        stats["total_count"] += 1
        if "main_question_id" in item:
            stats["unique_main_question_ids"].add(item["main_question_id"])
        stats["language_count"][item["language"]] += 1
        stats["question_type_count"][item["question_type"]] += 1
        # Check for image-related columns
        if "image_type" in item:
            stats["image_type_count"][item["image_type"]] += 1
            stats["image_type_set"].add(item["image_type"])
        
        # Check for image_1 and image_2 fields
        if "image_1" in item and item["image_1"] != 'None':
            stats["image_count"] = stats.get("image_count", 0) + 1
        stats["subfield_count"][item["subfield"]] += 1
        stats["subfield_set"].add(item["subfield"])
        stats["topic_difficulty_count"][item["topic_difficulty"]] += 1
        if "explanation" in item and item["explanation"]:
            stats["explanation_count"] += 1
        if "image_2" in item and item["image_2"] != 'None':
            stats["multiple_images_count"] += 1

        # Count arithmetic questions
        if item.get("is_arithmetic") == '1':  # Default to False if field doesn't exist
            stats["arithmetic_count"] += 1
            stats["arithmetic_by_language"][item["language"]] += 1
            stats["arithmetic_by_difficulty"][item["topic_difficulty"]] += 1

        # Calculate the token count of content + question
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Calculate question tokens
        question_tokens = len(tokenizer.encode(item["question"]))
        
        # Store context for first sub-question of each main question
        main_id = get_first_sub_question_id(item["question_id"])
        
        context = get_context(data, main_id)
        context_tokens = len(tokenizer.encode(context))
         
        # Total tokens is context tokens (from first sub-question) + current question tokens
        total_tokens = context_tokens + question_tokens
        
        # Update token counts by language
        stats["token_counts"][item["language"]]["total"] += total_tokens
        stats["token_counts"][item["language"]]["count"] += 1
        stats["token_counts"][item["language"]]["question_tokens"] = stats["token_counts"][item["language"]].get("question_tokens", 0) + question_tokens
        stats["token_counts"][item["language"]]["context_tokens"] = stats["token_counts"][item["language"]].get("context_tokens", 0) + context_tokens
        
        stats["total_token_sum"]["total"] += total_tokens
        stats["total_token_sum"]["count"] += 1
        stats["total_token_sum"]["question_tokens"] = stats["total_token_sum"].get("question_tokens", 0) + question_tokens
        stats["total_token_sum"]["context_tokens"] = stats["total_token_sum"].get("context_tokens", 0) + context_tokens

        # Update token counts by question type
        question_type = item["question_type"]
        stats["token_counts_by_question_type"][question_type]["total"] += total_tokens
        stats["token_counts_by_question_type"][question_type]["count"] += 1
        stats["token_counts_by_question_type"][question_type]["question_tokens"] = stats["token_counts_by_question_type"][question_type].get("question_tokens", 0) + question_tokens
        stats["token_counts_by_question_type"][question_type]["context_tokens"] = stats["token_counts_by_question_type"][question_type].get("context_tokens", 0) + context_tokens

        # Update token counts by difficulty
        difficulty = item["topic_difficulty"]
        stats["token_counts_by_difficulty"][difficulty]["total"] += total_tokens
        stats["token_counts_by_difficulty"][difficulty]["count"] += 1
        stats["token_counts_by_difficulty"][difficulty]["question_tokens"] = stats["token_counts_by_difficulty"][difficulty].get("question_tokens", 0) + question_tokens
        stats["token_counts_by_difficulty"][difficulty]["context_tokens"] = stats["token_counts_by_difficulty"][difficulty].get("context_tokens", 0) + context_tokens
        
        # Update token counts by arithmeticity
        is_arithmetic = item.get("is_arithmetic") == '1'
        arithmetic_key = "arithmetic" if is_arithmetic else "non_arithmetic"
        
        # Ensure the arithmetic key exists in token_counts_by_arithmeticity
        if "token_counts_by_arithmeticity" not in stats:
            stats["token_counts_by_arithmeticity"] = {}
        if arithmetic_key not in stats["token_counts_by_arithmeticity"]:
            stats["token_counts_by_arithmeticity"][arithmetic_key] = {
                "total": 0,
                "count": 0,
                "question_tokens": 0,
                "context_tokens": 0
            }
        
        # Update token counts for arithmeticity
        stats["token_counts_by_arithmeticity"][arithmetic_key]["total"] += total_tokens
        stats["token_counts_by_arithmeticity"][arithmetic_key]["count"] += 1
        stats["token_counts_by_arithmeticity"][arithmetic_key]["question_tokens"] += question_tokens
        stats["token_counts_by_arithmeticity"][arithmetic_key]["context_tokens"] += context_tokens

        # Update token counts by subfield
        subfield = item["subfield"]
        stats["token_counts_by_subfield"][subfield]["total"] += total_tokens
        stats["token_counts_by_subfield"][subfield]["count"] += 1
        stats["token_counts_by_subfield"][subfield]["question_tokens"] = stats["token_counts_by_subfield"][subfield].get("question_tokens", 0) + question_tokens
        stats["token_counts_by_subfield"][subfield]["context_tokens"] = stats["token_counts_by_subfield"][subfield].get("context_tokens", 0) + context_tokens

        # Update subfield difficulty counts
        subfield = item["subfield"]
        difficulty = item["topic_difficulty"]
        stats["subfield_difficulty_count"][item["language"]][subfield][difficulty] += 1

        # Update language difficulty counts
        stats["language_difficulty_count"][item["language"]][difficulty] += 1

        # Update question type difficulty counts
        question_type = item["question_type"]
        stats["question_type_difficulty_count"][question_type][difficulty] += 1

    # Calculate the average token count for each language
    for language, token_data in stats["token_counts"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        question_tokens = token_data.get("question_tokens", 0)
        context_tokens = token_data.get("context_tokens", 0)
        
        token_data["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
        
        del token_data["total"]
        del token_data["count"]
        del token_data["question_tokens"]
        del token_data["context_tokens"]

    # Calculate the average token count for all languages
    total_tokens = stats["total_token_sum"]["total"]
    total_questions = stats["total_token_sum"]["count"]
    question_tokens = stats["total_token_sum"].get("question_tokens", 0)
    context_tokens = stats["total_token_sum"].get("context_tokens", 0)
    
    stats["total_token_sum"]["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
    stats["total_token_sum"]["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
    stats["total_token_sum"]["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
    
    del stats["total_token_sum"]["total"]
    del stats["total_token_sum"]["count"]
    del stats["total_token_sum"]["question_tokens"]
    del stats["total_token_sum"]["context_tokens"]

    # Calculate averages for token counts by question type
    for question_type, token_data in stats["token_counts_by_question_type"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        question_tokens = token_data.get("question_tokens", 0)
        context_tokens = token_data.get("context_tokens", 0)
        
        token_data["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
        
        del token_data["total"]
        del token_data["count"]
        del token_data["question_tokens"]
        del token_data["context_tokens"]

    # Calculate averages for token counts by difficulty
    for difficulty, token_data in stats["token_counts_by_difficulty"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        question_tokens = token_data.get("question_tokens", 0)
        context_tokens = token_data.get("context_tokens", 0)
        
        token_data["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
        
        del token_data["total"]
        del token_data["count"]
        del token_data["question_tokens"]
        del token_data["context_tokens"]

    # Calculate averages for token counts by arithmeticity
    for arithmeticity, token_data in stats["token_counts_by_arithmeticity"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        question_tokens = token_data.get("question_tokens", 0)
        context_tokens = token_data.get("context_tokens", 0)
        
        token_data["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
        
        del token_data["total"]
        del token_data["count"]
        del token_data["question_tokens"]
        del token_data["context_tokens"]
        
    # Calculate averages for token counts by subfield
    for subfield, token_data in stats["token_counts_by_subfield"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        question_tokens = token_data.get("question_tokens", 0)
        context_tokens = token_data.get("context_tokens", 0)
        
        token_data["average_total"] = round(total_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_question"] = round(question_tokens / total_questions, 2) if total_questions > 0 else 0
        token_data["average_context"] = round(context_tokens / total_questions, 2) if total_questions > 0 else 0
        
        del token_data["total"]
        del token_data["count"]
        del token_data["question_tokens"]
        del token_data["context_tokens"]

    # Ensure all counts are integers before saving
    for language, subfields in stats["subfield_difficulty_count"].items():
        for subfield, difficulties in subfields.items():
            for difficulty, count in difficulties.items():
                # Convert count to integer
                stats["subfield_difficulty_count"][language][subfield][difficulty] = int(count)

    for language, counts in stats["language_difficulty_count"].items():
        for difficulty, count in counts.items():
            stats["language_difficulty_count"][language][difficulty] = int(count)

    for question_type, counts in stats["question_type_difficulty_count"].items():
        for difficulty, count in counts.items():
            stats["question_type_difficulty_count"][question_type][difficulty] = int(count)

    # Update the total_main_question_count with the size of the set
    stats["total_main_question_count"] = len(stats["unique_main_question_ids"])

    os.makedirs("statistics", exist_ok=True)
    # Path to save the statistics file
    stats_file_path = os.path.join("statistics", "summary.json")

    # Save the statistics to a JSON file
    save_json(stats, stats_file_path)

    # Return the statistics
    return stats

