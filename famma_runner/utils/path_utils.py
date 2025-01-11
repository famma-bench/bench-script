from pathlib import Path
from typing import Optional
    
from easyllm_kit.utils import ensure_dir

def get_cache_dir(model_repr:str, args) -> str:
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_dir(model_repr:str, args) -> str:
    n = args.n
    temperature = args.temperature
    language = args.language
    path = f"output/{model_repr}/{n}_{temperature}_{language}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_dir(model_repr:str, args) -> str:
    n = args.n
    temperature = args.temperature
    language = args.language
    path = f"output/{model_repr}/{n}_{temperature}_{language}_eval_all.json"
    return path

def find_image_file(parent_dir: Path, image_name: str) -> Optional[Path]:
    """
    Find image file with either .jpg or .png extension.
    
    Args:
        parent_dir: Root directory containing images
        image_name: Base name of the image without extension
        
    Returns:
        Path to image if found, None otherwise
    """
    for ext in ['.jpg', '.png']:
        image_path = parent_dir / f"{image_name}{ext}"
        if image_path.exists():
            return image_path
    return None