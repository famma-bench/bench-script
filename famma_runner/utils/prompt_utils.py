from easyllm_kit.utils import PromptTemplate


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


class AnalyzePrompt:
    _template = """You are a highly skilled expert in error analysis for AI models in the finance domain. You are reviewing collected incorrect answers to financial questions.
    The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question). '<image_1>, <image_2> ...' mentioned in the text of the context or question are sequential placeholders for images, which are fed at the same time as the textual information.
    You are given the context, the question, the student's answer and the student's explanation and the ground-truth answer. 
    
    You need to classify these incorrect answers based on the provided categories: perceptual errors, lack of knowledge, reasoning errors, and other errors. 
    Here are the definitions for each error type:
    perceptual errors: these occur when the model misinterprets basic visual information. Perceptual errors involve misunderstandings of elementary visual details or lack of specialized knowledge.
    lack of knowledge: this type of error occurs due to insufficient specialized knowledge. The model may correctly identify visual elements but fail to interpret them accurately in context.
    reasoning errors: these errors arise when the model correctly understands text and images but fails to apply logical or mathematical reasoning effectively. 
    other errors: this category includes errors not covered by the above types, such as Textual Understanding Errors, Rejection to Answer, Annotation Errors, and Answer Extraction Errors. These involve challenges in interpreting complex text, limitations in response generation, inaccuracies in data annotation, and difficulties in extracting precise answers.

    The input is as follows; use these details to determine the primary error category.

    context: {context}
    question: {question}
    model's incorrect answer: {model_answer}
    model's explanation: {model_explanation}
    ground-truth answer: {answer}
    
    Now please provide the result directly, identifying the error category as one of: perceptual errors, lack of knowledge, reasoning errors, or other errors. 
    """

    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "question",
                             "model_answer", "model_explanation", "answer"],
            template=self._template,
        )

    def get_prompt(self, context="", question="", model_answer="", model_explanation="", answer=""):
        return self.prompt.format(
            context=context,
            question=question,
            model_answer=model_answer,
            model_explanation=model_explanation,
            answer=answer,
        )
