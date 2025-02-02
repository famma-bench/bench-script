import argparse
from easyllm_kit.utils import HFHelper

if __name__ == "__main__":
    """
    Download the dataset from huggingface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_repo", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct",
                        help="The dir of model repo on huggingface.")

    parser.add_argument("--save_dir", type=str, default="/data/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-72B-Instruct/",
                        help="The dir to save.")

    args = parser.parse_args()

    HFHelper.download_model_from_hf(args.model_repo, 
                                    args.save_dir)
