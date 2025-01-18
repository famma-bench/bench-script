from easyllm_kit.utils import PromptTemplate


class QuestionPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  
        
        Question Format:
        - Context: The given financial context.
        - Sub-Questions: A series of related sub-questions tied to the same context and images. Later questions may depend on the answers to earlier ones. Each sub-question has a seperate question type (multiple-choice or open-ended), indicated at the beginning of the sub-question.
        - Images: Image placeholders like '<image_1>', '<image_2>' refer to accompanying images. If images are mentioned, they will be included alongside the textual context. If no images are provided, answer based solely on the textual context.
        
        Answering Guidelines:
        For each sub-question, provide:        
        - Answer:
            For multiple-choice questions, return the option index.
            For open-ended questions, provide a concise and precise answer.
        - Explanation: Provide a clear and detailed explanation (maximum 200 words) for your answer in the same language as the question.

        Now consider the following question:
        context: {{context}}
        {{sub_questions}} 
        
        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
        subquestion-1: {
            answer: <answer-1>,
            explanation: <explanation-1>
        },
        subquestion-2: {
            answer: <answer-2>,
            explanation: <explanation-2>
        },
        subquestion-3: {
            answer: <answer-3>,
            explanation: <explanation-3>
        },
        ...
        }
        ```
        Ensure that the response strictly adheres to JSON syntax without any additional content. 
        """

        return cls(
                    template=_template,
                    input_variables=["context", "sub_questions"]
                )

class JudgePrompt:
    _template = """You are a highly knowledgeable expert and teacher in the finance domain. 
    You are reviewing a student's answers to financial questions. 
    The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question). '<image_1>, <image_2> ...' mentioned in the text of the context or question are sequential placeholders for images, which are fed at the same time as the textual information.
    You are given the context, the question, the student's answer and the student's explanation and the ground-truth answer. 
    Please use the given information and refer to the ground-truth answer to determine if the student's answer is correct.
    
    The input information is as follows：
    
    context: {context}
    question: {question}
    student's answer: {model_answer}
    student's explanation: {model_explanation}
    ground-truth answer: {answer}
    
    If the student's answer is empty or completely nonsensical, please respond with 'unable to answer'. 
    In other cases, please respond directly as either 'correct' or 'incorrect'.
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
