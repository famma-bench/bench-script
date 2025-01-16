import argparse
from famma_runner.utils.data_utils import download_data

if __name__ == "__main__":
    """
    Download the dataset from huggingface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_dir", type=str, default="weaverbirdllm/famma",
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--split", type=str, default='release_v2406',
                        help="refer to the release version. If None, download all the splits.")

    parser.add_argument("--save_dir", type=str, default="./hf_data",
                        help="The local dir to save the dataset.",)

    args = parser.parse_args()

    download_data(args.hf_dir, args.split, args.save_dir)
