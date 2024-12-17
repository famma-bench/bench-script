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