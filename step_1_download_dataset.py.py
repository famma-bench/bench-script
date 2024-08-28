import argparse
from utils.data_utils import download_data

if __name__ == "__main__":
    """
    Download the dataset from huggingface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_dir", type=str, default="weaverbirdllm/famma",
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--subset_name", type=str, default=None,
                        help="English / Chinese / French, there are three subset. If None, download all the subset.")

    parser.add_argument("--save_dir", type=str, default="./data",
                        help="The local dir to save the dataset.",)

    args = parser.parse_args()

    download_data(args.hf_dir, args.subset_name, args.save_dir)
