from langchain.prompts import PromptTemplate


class MultipleChoiceQuestionPrompt:
    _template = """You are a highly knowledgeable financial expert. Please answer multiple-choice questions in the finance domain. You are given context, images, questions and options.
    The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question). '<image_1>, <image_2> ...' mentioned in the text of the context or question are sequential placeholders for images, which are fed at the same time as the textual information. 
    If no image information is provided, you must answer based solely on the given information.
    Besides, the question may contain several sub-questions that share the same context, and the answer to each sub-question may depend on the answers to previous ones.
    The question format is
    
    context: <context>
    sub-question-1: <sub-question-1>
    sub-question-2: <sub-question-2>
    sub-question-3: <sub-question-3>
    ...

    Now consider the following question:
    context: {context} 
    {sub_questions}
    
    Please provide the chosen answer and a precise, detailed explanation of why this choice is correct. The explanation should be in the same language as the question and should not exceed 400 words.
    Your response must be in a standard JSON format:
    {{
       sub-question-1: {{
           answer-1: <answer-1>,
           explanation-1: <explanation-1>
       }},
       sub-question-2: {{
           answer-2: <answer-2>,
           explanation-2: <explanation-2>
       }},
       sub-question-3: {{
           answer-3: <answer-3>,
           explanation-3: <explanation-3>
       }},
       ...
    }}
    Ensure that the response strictly adheres to JSON syntax without any additional content. 
    """

    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "sub_questions"], template=self._template
        )

    def get_prompt(self, context="", sub_questions=[]):
        sub_questions_str = ""

        for i, sub_question in enumerate(sub_questions):
            sub_questions_str += f"sub-question-{i + 1}: {sub_question}\n"

        prompt = self.prompt.format(
            context=context,
            sub_questions=sub_questions_str.strip(),
        )

        return prompt


class OpenQuestionPrompt:
    _template = """You are a highly knowledgeable financial expert. Please answer open-ended questions in the finance domain. 
    The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question). '<image_1>, <image_2> ...' mentioned in the text of the context or question are sequential placeholders for images, which are fed at the same time as the textual information. 
    If no image information is provided, you must answer based solely on the given information.
    Besides, the question may contain several sub-questions that share the same context, and the answer to each sub-question may depend on the answers to previous ones.
    The question format is
    
    context: <context>
    sub-question-1: <sub-question-1>
    sub-question-2: <sub-question-2>
    sub-question-3: <sub-question-3>
    ...
    
    Now consider the following question:
    context: {context} 
    {sub_questions} 

    Please provide the answer and a precise, detailed explanation. The explanation should be in the same language as the question and should not exceed 400 words.
    Your answer must be in a standard JSON format:
    {{
       sub-question-1: {{
           answer-1: "answer-1",
           explanation-1: "explanation-1"
       }},
       sub-question-2: {{
           answer-2: "answer-2",
           explanation-2: "explanation-2"
       }},
       sub-question-3: {{
           answer-3: "answer-3",
           explanation-3: "explanation-3"
       }},
       ...
    }}
    Ensure that the response strictly adheres to JSON syntax without any additional content. 
    """

    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "sub_questions"], template=self._template
        )

    def get_prompt(self, context="", sub_questions=[]):
        sub_questions_str = ""

        for i, question in enumerate(sub_questions):
            sub_questions_str += f"sub-question-{i + 1}: {question}\n"

        return self.prompt.format(
            context=context,
            sub_questions=sub_questions_str.strip(),
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
