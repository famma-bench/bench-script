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
        # default="./cache/release_v2501",
        help="The HuggingFace repository name"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default='release_livepro',
        help="Specific split to download (e.g., 'release_basic'). If not specified, downloads all splits."
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./hf_data",
        help="Directory to save the downloaded JSON files"
    )
    
    parser.add_argument(
        "--from_local", 
        type=bool, 
        default=False,
        help="If True, load from local cache instead of HuggingFace"
    )
    
    parser.add_argument(
        "--decode_answer", 
        type=bool, 
        default=True,
        help="If True, decode base64-encoded answers")
    
    args = parser.parse_args()
    
    success = download_data(
        hf_dir=args.hf_dir,
        split=args.split,
        save_dir=args.save_dir,
        from_local=args.from_local,
        decode_answer=args.decode_answer
    )
    
    if success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload failed!")

if __name__ == "__main__":
    main()
