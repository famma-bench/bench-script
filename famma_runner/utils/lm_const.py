from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Language(Enum):
    English = "English"
    Chinese = "Chinese"
    French = "French"

class LMStyle(Enum):
    """
    Code borrowed from LiveCodeBench
    ref: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/lm_styles.py
    """
    OpenAIChat = "OpenAIChat"
    OpenAIReason =  "OpenAIReason"
    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Gemini = "Gemini"
    DeepSeekAPI = "DeepSeekAPI"
    
    Qwen2 = "Qwen2"
    Qwen25 = "Qwen2.5"
    LLaMa32 = "LLaMa3.2"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    is_multi_modal: bool
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)


LanguageModelList: list[LanguageModel] = [
    LanguageModel(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        LMStyle.LLaMa32,
        True,
        datetime(2024, 9, 19),
        link="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct",
    ),
    LanguageModel(
        "gpt-4o-2024-08-06",
        "GPT-4O-2024-08-06",
        LMStyle.OpenAIChat,
        True,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "gpt-4o-mini-2024-07-18",
        "GPT-4O-mini-2024-07-18",
        LMStyle.OpenAIChat,
        True,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "o1-preview-2024-09-12",
        "O1-Preview-2024-09-12 (N=1)",
        LMStyle.OpenAIReason,
        True,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "o1-mini-2024-09-12",
        "O1-Mini-2024-09-12 (N=1)",
        LMStyle.OpenAIReason,
        True,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "claude-2",
        "Claude-2",
        LMStyle.Claude,
        True,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/claude-2",
    ),
    LanguageModel(
        "claude-3-opus-20240229",
        "Claude-3-Opus",
        LMStyle.Claude3,
        True,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-sonnet-20240229",
        "Claude-3-Sonnet",
        LMStyle.Claude3,
        True,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-5-sonnet-20240620",
        "Claude-3.5-Sonnet",
        LMStyle.Claude3,
        True,
        datetime(2024, 3, 31),
        link="https://www.anthropic.com/news/claude-3-5-sonnet",
    ),
    LanguageModel(
        "claude-3-haiku-20240307",
        "Claude-3-Haiku",
        LMStyle.Claude3,
        True,
        datetime(2023, 4, 30),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "gemini-1.5-pro-002",
        "Gemini-Pro-1.5-002",
        LMStyle.Gemini,
        True,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-flash-002",
        "Gemini-Flash-1.5-002",
        LMStyle.Gemini,
        True,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "Qwen/Qwen2-72B",
        "Qwen2-Base-72B",
        LMStyle.Qwen2,
        False,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-72B",
    ),
    LanguageModel(
        "Qwen/Qwen2-72B-Instruct",
        "Qwen2-Ins-72B",
        LMStyle.Qwen2,
        False,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-72B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2.5-7B",
        "Qwen2.5-Base-7B",
        LMStyle.Qwen25,
        False,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-7B",
    ),
    LanguageModel(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-Ins-7B",
        LMStyle.Qwen25,
        False,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen2-VL-Ins-72B",
        LMStyle.Qwen2,
        True,
        datetime(2023, 9, 17),
        link="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct",
    ),
]

LanguageModelStore: dict[str, LanguageModel] = {
    lm.model_name: lm for lm in LanguageModelList
}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))