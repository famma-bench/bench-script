from easyllm_kit.utils import PromptTemplate


class SingleQuestionGRPOPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  
        
        Question Format:
        - Context: The given financial context.
        - question: the actual question text(question_id, question_text, question_type, etc)
        - Images: Image placeholders like '<image_1>', '<image_2>' refer to images. We use OCR to extract the text from the images.
        
        Answering Guidelines:
        For question, provide answer:
        - For multiple-choice questions, directly return the option index A, B, C, D, etc.
        - For open-ended questions, provide a short and precise answer.

        Please first have a step-by-step thinking process of how to approach the question, noted using <think> </think> and then provide answer with thinking process:
        
        <think>
        thinking process here
        </think>

        <answer>
        answer here
        </answer>
        
        Now please answer the following question:
        context: {{context}}
        question: {{question}}
        """

        return cls(
            template=_template,
            input_variables=["context", "question"]
        )
    
class QuestionPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  
        
        Question Format:
        - Context: The given financial context.
        - Sub_questions: A list of sub-questions, where each contains:
        id: unique identifier for the sub-question
        type: question type ('multiple-choice' or 'open-ended')
        question: the actual question text
        - Images: Image placeholders like '<image_1>', '<image_2>' refer to accompanying images. If images are mentioned, they will be included alongside the textual context. If no images are provided, answer based solely on the textual context.
        
        Answering Guidelines:
        For each sub_question, provide:        
        - Answer:
            For multiple-choice questions, return the option index A, B, C, D, etc.
            For open-ended questions, provide a concise and precise answer.
        - Explanation: Provide a clear and detailed explanation (maximum 200 words) for your answer in the same language as the question.
        - Explanation Format: Write explanations in clear, natural language without using special characters or symbols that could interfere with JSON parsing (avoid \n, \t, etc.). Keep explanations concise and focused.

        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
            "<question_id>": {
                "answer": "<answer>",
                "explanation": "<explanation>"
            },
            "<question_id>": {
                "answer": "<answer>",
                "explanation": "<explanation>"
            },
            ...
        }
        ```
        Ensure that the response strictly adheres to JSON syntax without any additional content. 
        Now please answer the following question:
        context: {{context}}
        {{sub_questions}} 
        """

        return cls(
            template=_template,
            input_variables=["context", "sub_questions"]
        )


class JudgePrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable expert and teacher in the finance domain. 
        You are reviewing a student's answers to financial questions. 
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question). '<image_1>, <image_2> ...' mentioned in the text of the context or question are sequential placeholders for images, which are fed at the same time as the textual information.
        You are given the context, the question, the student's answer and the student's explanation and the ground-truth answer. 
        Please use the given information and refer to the ground-truth answer to determine if the student's answer is correct.
        
        Question Format:
        {
            "question_id": "<unique_identifier>",
            "context": "<financial_context>",
            "type": "<multiple-choice|open-ended>",
            "question": "<question_text>",
            "student_answer": "<student's_answer>",
            "student_explanation": "<student's_explanation>",
            "ground_truth": "<correct_answer>"
        }

        Evaluation Guidelines:
        For multiple-choice questions:
        Correct if student's answer matches the ground truth content, regardless of format
        Example: If correct answer is "A. Stock market", both "A" and "Stock market" are considered correct
        Focus on whether the student selected the right concept/answer, not the format
        For open-ended questions:
        Compare key concepts and accuracy of student's response with ground truth
        Respond directly as either 'correct' or 'incorrect'.

        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
            "<question_id>": "correct" or "incorrect"
        }
        ```
        Now please evaluate the following response:
        {{question}}
        """

        return cls(
            template=_template,
            input_variables=["question"]
        )


class ProgramOfThoughtsQuestionPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert who solves problems step-by-step using code. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  
        
        Question Format:
        - Context: The given financial context.
        - Sub_questions: A list of sub-questions, where each contains:
        id: unique identifier for the sub-question
        type: question type ('multiple-choice' or 'open-ended')
        question: the actual question text
        - Images: Image placeholders like '<image_1>', '<image_2>' refer to accompanying images. If images are mentioned, they will be included alongside the textual context. If no images are provided, answer based solely on the textual context.
        
        Answering Guidelines:
        For each sub_question, provide:        
        - Answer: Write complete Python code that solves the problem. Extract all relevant values from the question as input variables and include the full computation. The code should be executable and produce the final answer.
        - Explanation: After the code, provide a clear explanation of how the code works and how it leads to the answer in the same language as the question. max 100 words.
        
        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
            "<question_id>": {
                "answer": "<complete_python_code>",
                "explanation": "<explanation_of_code_and_solution>"
            },
            "<question_id>": {
                "answer": "<complete_python_code>",
                "explanation": "<explanation_of_code_and_solution>"
            },
            ...
        }
        ```
        Ensure that the response strictly adheres to JSON syntax without any additional content. 
        Now please answer the following question:
        context: {{context}}
        {{sub_questions}} 
        """

        return cls(
            template=_template,
            input_variables=["context", "sub_questions"]
        )


class QuestionThinkPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  

        Question Format:
        - Context: The given financial context.
        - Sub_questions: A list of sub-questions, where each contains:
        id: unique identifier for the sub-question
        type: question type ('multiple-choice' or 'open-ended')
        question: the actual question text
        - Images: Image placeholders like '<image_1>', '<image_2>' refer to images. We use OCR to extract the text from the images.

        Answering Guidelines:
        Please first have a step-by-step thinking process of how to approach the question, noted using <think> and </think> and then provide answer:
        - For multiple-choice questions, return the option index A, B, C, D, etc.
        - For open-ended questions, provide a concise and precise answer.

        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
            "<question_id>": {
                "answer": "<answer>"
            },
            "<question_id>": {
                "answer": "<answer>"
            },
            ...
        }
        ```
        Ensure that the response strictly adheres to JSON syntax without any additional content. 
        Now please answer the following question:
        context: {{context}}
        {{sub_questions}} 
        """

        return cls(
            template=_template,
            input_variables=["context", "sub_questions"]
        )


class JsonResponsePrompt(PromptTemplate):
    @classmethod
    def init(cls) -> "JsonResponsePrompt":
        template = '''```json{{output_dict}}```
        '''
        return cls(
            template=template,
            input_variables=['output_dict']
        )
