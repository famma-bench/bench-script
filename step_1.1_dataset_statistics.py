import argparse
from utils.descriptive_utils import get_dataset_statistics

if __name__ == "__main__":
    """
    Process dataset and save statistics
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="./data", help="The parent dir of dataset")

    args = parser.parse_args()

    get_dataset_statistics(args.data_dir)
