import sys
import os
import pandas as pd

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project's root directory
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add the project's root directory to sys.path
sys.path.append(project_root)

from utils.data_utils import read_json
from utils.eval_utils import eval_ans

def test_eval_ans():
    model_name = "gpt-4o"
    api_key = "sk-xxx"
    # You can change the path of the data to be evaluated
    data_dir = "./test_result/gpt-4o_model_answers"
    save_dir = "./test_result/gpt-4o_model_answers"

    # Run the evaluation function
    eval_ans(model_name, api_key, data_dir, save_dir)

    # Check if the result directory exists
    result_dir = os.path.join(data_dir, "evaluation_result")
    assert os.path.exists(result_dir), "Result directory not generated"
    print("Result directory is generated")

    # Verify the score.json file
    score_file_path = os.path.join(result_dir, "score.json")
    assert os.path.exists(score_file_path), "Score file not generated"
    print("Score file is generated")

    score_data = read_json(score_file_path)
    # Verify the correction_result.json file
    correction_result_path = os.path.join(result_dir, "correction_result.csv")
    assert os.path.exists(
        correction_result_path), "Correction result file not generated"
    print("Correction result file is generated")

    # Load the correction data
    correction_data = pd.read_csv(correction_result_path)

    # Verify that each row in the DataFrame has been evaluated
    for index, row in correction_data.iterrows():
        assert "is_correct" in row, f"Row {index} has not been evaluated"

    print("Each question has been evaluated")

    # Verify scores
    total_count = score_data["total_count"]
    correct_count = score_data["correct_count"]
    total_score = score_data["total_score"]
    normalized_score = score_data["normalized_score"]

    # Ensure total_count matches the number of samples
    total_samples_count = len(correction_data)
    assert total_count == total_samples_count, f"Total count mismatch, expected {total_samples_count}, got {total_count}"
    print("The number of result is correct")

    # Ensure normalized score is within 0-100
    assert 0 <= normalized_score <= 100, f"Normalized score out of bounds, got {normalized_score}"
    print("The normalized score is correct")

    # Print final scores
    print(f"Total count: {total_count}")
    print(f"Correct count: {correct_count}")
    print(f"accuracy_rate: {round(correct_count / total_count, 3)}")
    print(f"Total score: {total_score}")
    print(f"Normalized score: {normalized_score}")

    print("Test passed!")


if __name__ == '__main__':
    test_eval_ans()
