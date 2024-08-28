import sys
import os
import pandas as pd

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project's root directory
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# Add the project's root directory to sys.path
sys.path.append(project_root)

from utils.gen_utils import generate_ans

def test_generate_ans():
    model_name = "gpt-4o"
    # You can change the path of the data to be generated
    data_dir = "./test_data"
    subset_name = None
    save_dir = "./test_result"
    question_ids = None

    generate_ans(model_name, "sk-xxx",
                 data_dir, subset_name, save_dir, question_ids)

    # Generate the save sub-directory based on model_name
    save_sub_dir = f"{model_name}_model_answers"

    # Verify if the output file is generated
    output_file_path = os.path.join(
        save_dir, save_sub_dir, f"{save_sub_dir}.csv")
    assert os.path.exists(output_file_path), "Output file not generated"
    print("The output file is generated")

    # Read the output file
    output_data = pd.read_csv(output_file_path)

    # Count total number of samples across all languages
    total_samples_count = len(output_data)
    assert total_samples_count == 6, f"Number of results incorrect, expected 6, got {total_samples_count}"
    print("Number of results correct")

    expected_context = ("A Sydney-based\nfixed-income\nportfolio manager is considering the following\n"
                        "Commonwealth of Australia government bonds traded on the ASX (Australian\n"
                        "Stock Exchange):\n\n<image_1>\n\nThe manager is considering portfolio strategies "
                        "based upon various interest rate\nscenarios over the next 12 months. She is considering "
                        "three long-only\ngovernment\nbond portfolio alternatives, as follows:\nBullet: Invest solely "
                        "in 4.5-year\ngovernment bonds\nBarbell: Invest equally in 2-year\nand 9-year\ngovernment "
                        "bonds\nEqual weights: Invest equally in 2-year,\n4.5-year,\nand 9-year\nbonds")

    has_context = False
    context = ""
    # Verify if the context is correct across all samples
    for index, row in output_data.iterrows():
        if "context" in row and expected_context in row["context"]:
            has_context = True
            context = row["context"]
            break

    assert has_context is True, f"Context is incorrect, expected '{expected_context}', got '{context}'"
    print("Context correct")

    print("Test passed!")


if __name__ == '__main__':
    test_generate_ans()
