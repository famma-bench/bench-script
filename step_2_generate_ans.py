import argparse
from utils.data_utils import parse_list
from utils.gen_utils import generate_ans

if __name__ == "__main__":
    """
    Generate answers from a specified model and save the results to files.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, default="./gen_config.yaml",
                        help="The dir of model config file.")

    parser.add_argument("--data_dir", type=str,
                        default="./data", help="The parent dir of dataset")

    parser.add_argument("--question_ids", type=parse_list, default=None,
                        help="list of question ids to query. If None, we will run over all the questions in the subset")

    parser.add_argument("--save_dir", type=str, default="./result",
                        help="The local dir to save the generation result")

    args = parser.parse_args()

    generate_ans(args.config_dir, 
                 args.data_dir,
                 args.save_dir, 
                 args.question_ids)
