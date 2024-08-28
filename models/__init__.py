def get_client_fn(model_name):
    if model_name in [
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
    ]:
        from .claude import ModelClient
    # gemini
    elif model_name in [
        "gemini-1.5-pro-001",
        "gemini-1.0-pro-vision-001",
        "gemini-1.5-flash-001",
    ]:
        from .gemini import ModelClient
    # gpt
    elif model_name in [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-mini-2024-07-18",
    ]:
        from .gpt import ModelClient
    # custom model
    else:
        raise ValueError(f"Model {model_name} not supported")
    return ModelClient
