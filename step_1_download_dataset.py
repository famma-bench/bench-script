import argparse
from famma_runner.utils.data_utils import download_data

def main():
    """
    Download the dataset from HuggingFace and convert to JSON format
    """
    parser = argparse.ArgumentParser(description="Download dataset from HuggingFace")
    
    parser.add_argument(
        "--hf_dir", 
        type=str, 
        default="weaverbirdllm/famma",
        help="The HuggingFace repository name"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default=None,
        help="Specific split to download (e.g., 'release_v2406'). If not specified, downloads all splits."
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./hf_data",
        help="Directory to save the downloaded JSON files"
    )
    
    args = parser.parse_args()
    
    success = download_data(
        hf_dir=args.hf_dir,
        split=args.split,
        save_dir=args.save_dir
    )
    
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload failed!")

if __name__ == "__main__":
    main()
