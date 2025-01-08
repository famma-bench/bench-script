from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datasets import Features, Value, Sequence

class ReleaseVersion(str, Enum):
    """
    Maps release versions to their short names.
    """
    RELEASE_V1 = 'release_v2406'
    RELEASE_V2 = 'release_v2501'
    RELEASE_V3 = 'release_v2507'
    
    @classmethod
    def to_short_name(cls, release: str) -> str:
        """Convert release version to short name"""
        mapping = {
            cls.RELEASE_V1: 'r1',
            cls.RELEASE_V2: 'r2',
            cls.RELEASE_V3: 'r3',
        }
        return mapping.get(release, release)
    
    @classmethod
    def from_short_name(cls, short_name: str) -> str:
        """Convert short name back to release version"""
        mapping = {
            'r1': cls.RELEASE_V1,
            'r2': cls.RELEASE_V2,
            'r3': cls.RELEASE_V3,
        }
        return mapping.get(short_name, short_name)

class DatasetColumns(str, Enum):
    """
    Enum class for dataset column names.
    Inherits from str to allow direct string comparison and usage.
    """
    QUESTION_ID = 'question_id'
    CONTEXT = 'context'
    QUESTION = 'question'
    OPTIONS = 'options'
    IMAGE_1 = 'image_1'
    IMAGE_2 = 'image_2'
    IMAGE_3 = 'image_3'
    IMAGE_4 = 'image_4'
    IMAGE_5 = 'image_5'
    IMAGE_6 = 'image_6'
    IMAGE_7 = 'image_7'
    IMAGE_TYPE = 'image_type'
    ANSWERS = 'answers'
    EXPLANATION = 'explanation'
    TOPIC_DIFFICULTY = 'topic_difficulty'
    QUESTION_TYPE = 'question_type'
    SUBFIELD = 'subfield'
    LANGUAGE = 'language'
    MAIN_QUESTION_ID = 'main_question_id'
    SUB_QUESTION_ID = 'sub_question_id'
    ANS_IMAGE_1 = 'ans_image_1'
    ANS_IMAGE_2 = 'ans_image_2'
    ANS_IMAGE_3 = 'ans_image_3'
    RELEASE = 'release'

    @classmethod
    def image_columns(cls) -> list[str]:
        """Returns list of main image column names"""
        return [f'image_{i}' for i in range(1, 8)]
    
    @classmethod
    def answer_image_columns(cls) -> list[str]:
        """Returns list of answer image columns"""
        return [f'ans_image_{i}' for i in range(1, 4)]
    
    @classmethod
    def all_columns(cls) -> list[str]:
        """Returns list of all column names"""
        return [member.value for member in cls]

    @classmethod
    def get_features(cls) -> Features:
        """
        Returns the features dictionary compatible with Hugging Face datasets.
        """
        return Features({
            cls.QUESTION_ID: Value('string'),
            cls.CONTEXT: Value('string'),
            cls.QUESTION: Value('string'),
            cls.OPTIONS: Sequence(Value('string')),  # List of options
            'images': Sequence(Value('string')),  # List of base64 strings
            'answer_images': Sequence(Value('string')),  # List of base64 strings
            cls.IMAGE_TYPE: Value('string'),
            cls.ANSWERS: Value('string'),
            cls.EXPLANATION: Value('string'),
            cls.TOPIC_DIFFICULTY: Value('string'),
            cls.QUESTION_TYPE: Value('string'),
            cls.SUBFIELD: Value('string'),
            cls.LANGUAGE: Value('string'),
            cls.MAIN_QUESTION_ID: Value('string'),
            cls.SUB_QUESTION_ID: Value('string'),
            cls.RELEASE: Value('string')
        })

    @classmethod
    def validate_sample(cls, sample: Dict) -> bool:
        """
        Validates if a sample matches the expected schema.
        Returns True if valid, raises ValueError if invalid.
        """
        required_keys = {
            cls.QUESTION_ID, cls.CONTEXT, cls.QUESTION, 'images', 'answer_images',
            cls.IMAGE_TYPE, cls.ANSWERS, cls.EXPLANATION, cls.TOPIC_DIFFICULTY,
            cls.QUESTION_TYPE, cls.SUBFIELD, cls.LANGUAGE, cls.MAIN_QUESTION_ID,
            cls.SUB_QUESTION_ID, cls.RELEASE
        }
        
        missing_keys = required_keys - set(sample.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
            
        return True
