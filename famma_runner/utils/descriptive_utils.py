import os
import tiktoken
import pandas as pd

from collections import defaultdict
from utils.data_utils import save_json, read_json


def get_dataset_statistics(data_dir):
    """
    Calculates and saves dataset statistics
    """
    # Initialize the dictionary for statistical data
    stats = {
        "total_count": 0,
        "language_count": defaultdict(int),
        "split_count": defaultdict(int),
        "question_type_count": defaultdict(int),
        "image_type_count": defaultdict(int),
        "image_type_set": set(),
        "subfield_count": defaultdict(int),
        "subfield_set": set(),
        "topic_difficulty_count": defaultdict(int),
        "explanation_count": 0,
        "multiple_images_count": 0,
        "token_counts": defaultdict(lambda: {"total": 0, "count": 0, "average": 0}),
        "subfield_difficulty_count": defaultdict(lambda: defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0})),
        "total_token_sum": {"total": 0, "count": 0, "average": 0},
        "language_difficulty_count": defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0}),
        "question_type_difficulty_count": defaultdict(
            lambda: {"easy": 0, "medium": 0, "hard": 0}
        ),
    }

    # Traverse directories and files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                data = read_json(file_path)
                # Get the language from the directory name
                language = os.path.basename(root)
                # Get the split type from the file name without the extension
                split = os.path.splitext(file)[0]

                for item in data:
                    stats["total_count"] += 1
                    stats["language_count"][language] += 1
                    stats["split_count"][split] += 1
                    stats["question_type_count"][item["question_type"]] += 1
                    stats["image_type_count"][item["image_type"]] += 1
                    stats["image_type_set"].add(item["image_type"])
                    stats["subfield_count"][item["subfield"]] += 1
                    stats["subfield_set"].add(item["subfield"])
                    stats["topic_difficulty_count"][item["topic_difficulty"]] += 1
                    if item["explanation"]:
                        stats["explanation_count"] += 1
                    if item["image_2"]:
                        stats["multiple_images_count"] += 1

                    # Calculate the token count of content + question
                    content_question = item["context"] + item["question"]
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                    token_count = len(tokenizer.encode(content_question))
                    stats["token_counts"][language]["total"] += token_count
                    stats["token_counts"][language]["count"] += 1
                    stats["total_token_sum"]["total"] += token_count
                    stats["total_token_sum"]["count"] += 1

                    # Update subfield difficulty counts
                    subfield = item["subfield"]
                    difficulty = item["topic_difficulty"]
                    stats["subfield_difficulty_count"][language][subfield][difficulty] += 1

                    # Update language difficulty counts
                    stats["language_difficulty_count"][language][difficulty] += 1

                    # Update question type difficulty counts
                    question_type = item["question_type"]
                    stats["question_type_difficulty_count"][question_type][difficulty] += 1

    # Calculate the average token count for each language
    for language, token_data in stats["token_counts"].items():
        total_tokens = token_data["total"]
        total_questions = token_data["count"]
        token_data["average"] = (
            round(total_tokens / total_questions,
                  2) if total_questions > 0 else 0
        )
        del token_data["total"]
        del token_data["count"]

    # Calculate the average token count for all languages
    total_tokens = stats["total_token_sum"]["total"]
    total_questions = stats["total_token_sum"]["count"]
    stats["total_token_sum"]["average"] = (
        round(total_tokens / total_questions, 2) if total_questions > 0 else 0
    )
    del stats["total_token_sum"]["total"]
    del stats["total_token_sum"]["count"]

    # Convert set types to lists
    stats["image_type_set"] = list(stats["image_type_set"])
    stats["subfield_set"] = list(stats["subfield_set"])

    os.makedirs("statistics", exist_ok=True)
    # Path to save the statistics file
    stats_file_path = os.path.join("statistics", "summary.json")

    # Save the statistics to a JSON file
    save_json(stats_file_path, stats)

    # Return the statistics
    return stats

def _convert_to_df(sample_score_db):
    data_df = pd.DataFrame.from_dict(sample_score_db, orient='index')
    return data_df


def consolidate_result(sample_score_db):
    """
    Calculates and saves evaluation metrics.
    """
    # Initialize result dictionary
    sample_score_df = _convert_to_df(sample_score_db)
    # get the first row
    model_name = sample_score_df.iloc[0]["model_name"]
    total_count = sample_score_df.shape[0]
    correct_count = sample_score_df[sample_score_df["is_correct"] == True].shape[0]
    unable_to_answer_count = sample_score_df[sample_score_df["is_correct"] == "unable to answer"].shape[0]
    result_dict = {
        "model_name": model_name,
        "total_count": total_count,
        "correct_count": correct_count,
        "unable_to_answer_count": unable_to_answer_count,
        "accuracy_rate": round(correct_count / total_count, 3) if total_count > 0 else 0.0,
        "language_stats": defaultdict(lambda: {
            "total_count": 0,
            "correct_count": 0,
            "unable_to_answer_count": 0,
            "accuracy_rate": 0.0,
            "difficulty_stats": defaultdict(lambda: {
                "total_count": 0,
                "correct_count": 0,
                "unable_to_answer_count": 0,
                "accuracy_rate": 0.0,
            })
        }),
        "subfield_stats": defaultdict(lambda: {
            "total_count": 0,
            "correct_count": 0,
            "accuracy_rate": 0.0,
            "unable_to_answer_count": 0,
        }),
        "difficulty_stats": defaultdict(lambda: {
            "total_count": 0,
            "correct_count": 0,
            "accuracy_rate": 0.0,
            "unable_to_answer_count": 0,
        })
    }

    # Initialize per-language statistics
    language_stats = defaultdict(lambda: {
        "total_count": 0,
        "correct_count": 0,
        "unable_to_answer_count": 0,
        "total_score": 0,
        "difficulty_stats": defaultdict(lambda: {
            "total_count": 0,
            "correct_count": 0,
            "unable_to_answer_count": 0,
            "total_score": 0.0,
            "accuracy_rate": 0.0,
        })
    })

    # Initialize per-subfield statistics
    subfield_stats = defaultdict(lambda: {
        "total_count": 0,
        "correct_count": 0,
        "unable_to_answer_count": 0,
        "total_score": 0.0,
        "max_score": 0,
    })

    # Initialize per-difficulty statistics
    difficulty_stats = defaultdict(lambda: {
        "total_count": 0,
        "correct_count": 0,
        "unable_to_answer_count": 0,
        "total_score": 0.0,
        "max_score": 0,
    })

    # Compute per-language statistics
    for question_id, question_data in sample_score_db.items():
        language_key = question_data["language"].capitalize()
        subfield_key = question_data["subfield"]
        difficulty = question_data["topic_difficulty"]

        # Update language stats
        language_stats[language_key]["total_count"] += 1

        # Update language difficulty stats
        language_stats[language_key]["difficulty_stats"][difficulty]["total_count"] += 1

        # Update subfield stats
        subfield_stats[subfield_key]["total_count"] += 1

        # Update difficulty stats
        difficulty_stats[difficulty]["total_count"] += 1

        if question_data.get("is_correct") == "unable to answer":
            language_stats[language_key]["unable_to_answer_count"] += 1
            language_stats[language_key]["difficulty_stats"][difficulty]["unable_to_answer_count"] += 1
            subfield_stats[subfield_key]["unable_to_answer_count"] += 1
            difficulty_stats[difficulty]["unable_to_answer_count"] += 1
        elif question_data.get("is_correct") == True or question_data.get("is_correct") == 'True':
            language_stats[language_key]["correct_count"] += 1
            language_stats[language_key]["difficulty_stats"][difficulty]["correct_count"] += 1
            subfield_stats[subfield_key]["correct_count"] += 1
            difficulty_stats[difficulty]["correct_count"] += 1

    # Compute language-specific statistics
    for language, stats in language_stats.items():
        total_count = stats["total_count"]
        correct_count = stats["correct_count"]
        unable_to_answer_count = stats["unable_to_answer_count"]
        total_score = stats["total_score"]

        accuracy_rate = round(correct_count / total_count,
                              3) if total_count > 0 else 0.0

        result_dict["language_stats"][language] = {
            "total_count": total_count,
            "correct_count": correct_count,
            "unable_to_answer_count": unable_to_answer_count,
            "accuracy_rate": accuracy_rate,
            "difficulty_stats": {difficulty: {
                "total_count": stats["difficulty_stats"][difficulty]["total_count"],
                "correct_count": stats["difficulty_stats"][difficulty]["correct_count"],
                "unable_to_answer_count": stats["difficulty_stats"][difficulty]["unable_to_answer_count"],
                "accuracy_rate": round(
                    stats["difficulty_stats"][difficulty]["correct_count"] /
                    stats["difficulty_stats"][difficulty]["total_count"], 3)
                if stats["difficulty_stats"][difficulty]["total_count"] > 0 else 0.0,
            } for difficulty in stats["difficulty_stats"]}
        }

    # Compute subfield-specific statistics
    for subfield, stats in subfield_stats.items():
        total_count = stats["total_count"]
        correct_count = stats["correct_count"]
        unable_to_answer_count = stats["unable_to_answer_count"]
        total_score = stats["total_score"]

        accuracy_rate = round(correct_count / total_count,
                              3) if total_count > 0 else 0.0

        result_dict["subfield_stats"][subfield] = {
            "total_count": total_count,
            "correct_count": correct_count,
            "unable_to_answer_count": unable_to_answer_count,
            "accuracy_rate": accuracy_rate
        }

    # Compute subfield-specific statistics
    for difficulty, stats in difficulty_stats.items():
        total_count = stats["total_count"]
        correct_count = stats["correct_count"]
        unable_to_answer_count = stats["unable_to_answer_count"]
        total_score = stats["total_score"]

        accuracy_rate = round(correct_count / total_count,
                              3) if total_count > 0 else 0.0

        result_dict["difficulty_stats"][difficulty] = {
            "total_count": total_count,
            "correct_count": correct_count,
            "unable_to_answer_count": unable_to_answer_count,
            "accuracy_rate": accuracy_rate,
            "total_score": total_score
        }

    # Save results
    # save to current directory
    save_json('eval_result.json', result_dict)