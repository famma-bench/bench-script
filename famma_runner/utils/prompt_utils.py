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


class QuestionPromptForReasoningFineTune(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. Please answer the questions in the finance domain. You are given context, images, questions and options.
        The questions are multilingual (either in English, Chinese, or French) and multimodal (containing images as part of the question).  

        Question Format:
        - Context: The given financial context.
        - question: the actual question text along with the question type, if it is a multiple-choice question, the options are also provided.

        Answering Guidelines:
        Please first provide a detailed step-by-step thinking process to solve the question, enclosed within <think> tags:
        <think>
        [Your step-by-step reasoning process here. Break down the problem, analyze the context, consider relevant financial concepts, and work through calculations methodically, verify the reasoning process is correct.]
        MUST use the same language as the question, if the question is in Chinese or French, your answer should also be in Chinese or French.
        </think>
        
        Then provide your final answer within <answer> tags:
        <answer>
        For multiple-choice questions: Provide the letter of the correct option (A, B, C, D, etc.)
        For open-ended questions: Provide a concise, precise answer based on your reasoning.
        MUST use the same language as the question, if the question is in Chinese or French, your answer should also be in Chinese or French.
        </answer>
        
        Now you are given the following question:
        context: {{context}}
        question: {{question}}
        
        Please think step by step and provide your final answer.
        """

        return cls(
            template=_template,
            input_variables=["context", "question"]
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


class QuestionParsingPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. 
        I will provide you with text content from financial questions and their corresponding answers. Your task is to analyze and structure this content.

        1. **Content Analysis**
        - Review the provided text content of questions and their corresponding answers
        - Identify and separate any sub-questions within main questions
        - Ensure you capture all relevant information from the text

        2. **Create a Structured Response for Each Question**
        - **Context:** Provide a clear, self-contained restatement of the scenario in question-form. This should allow a reader to understand the context without needing the original text.
        - **Question:** Quote the question exactly as provided (verbatim, with no omissions or re-phrasing).
        - **Answer:** Include the exact answer as provided in the text.

        Your response must be in a standard JSON format and should follow this structure:
        ```json
        {
            "question_id": start with 1, 2, 3, ...
            "question_type": "multiple-choice" or "open question", 
            "context": "<context>",
            "question": "<question>",
            "answer": "<answer>"
        }
        ```

        Please structure the questions and answers from the provided text content.
        """
        return cls(template=_template, input_variables=None)

class DistillationPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a highly knowledgeable financial expert. 
        I give you a screenshot of questions and a screenshot of the ground truth answers. Please first extract and understand the questions and the ground truth answers, then generate a deep thinking process of how to solve the questions.
        Please first have a step-by-step thinking process of how to approach the question, noted using <think> and </think> and then provide answer:
        - For multiple-choice questions, return the option index A, B, C, D, etc.
        - For open-ended questions, provide a concise and precise answer.

        Your response must be in a standard JSON format and should follow this structure:
        ```json 
        {
            "question": "<question>",
            "thinking": "<thinking>",
            "answer": "<answer>"
        }
        ```
        """
        return cls(template=_template, input_variables=None)


class ReasoningDistillationPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are a financial expert with deep expertise in financial markets, accounting principles, and investment strategies. Given a context and a question (which may be open-ended or multiple-choice), your task is to generate a detailed, natural reasoning process to solve the problem.

    Your reasoning should flow naturally, with different types of thinking interleaved as you work through the problem. Use the following tags as needed throughout your response:
    - `<think>...</think>`: Use this for your general reasoning, analysis, and reflections. This includes:
        * Identifying key variables and facts from the context
        * Recalling relevant financial formulas and principles
        * Structuring your approach to the problem
        * Evaluating different perspectives and alternatives
        * Verifying assumptions and checking your work
    - `<python>...</python>`: When calculations would be helpful, include Python code with:
        * Clear comments explaining your approach
        * Step-by-step implementation
        * The final output assigned to a variable named `result`
    - `<search>...</search>`: When you need external information that might not be in the context, specify:
        * What information you need to search for
        * Why this information is necessary for solving the problem
    - `<information>...</information>`: After a search request, present the information you would expect to find:
        * Relevant data points or insights
        * How this information connects to the problem
    IMPORTANT: These tags should be naturally interleaved throughout your response as your thinking evolves. For example, you might start with some general thinking, then realize you need to search for information, incorporate that information into more thinking, use Python for calculations, and continue with further analysis.

    - `<answer>...</answer>`: After your detailed reasoning process, provide your final answer. For multiple-choice questions, specify the option (e.g., "B"). For open-ended questions, give a concise, precise conclusion.

    IMPORTANT: Your thinking process MUST be at least 200 words long and explore multiple angles of the problem. Do not abbreviate your reasoning or skip steps. The more comprehensive and detailed your reasoning, the better. and MUST USE the tags <think>, </think>, <python>, </python>, <search>, </search>, <information>, </information>, <answer>, </answer> to structure your thinking process.

    Do EXTREMELY DETAILED and EXHAUSTIVELY LONG step-by-step the thinking process with interleaved <think>, </think>, <python>, </python>, <search>, </search>, <information>, </information>, <answer>, </answer> and return the final output in the following JSON format:
    ```json
    {
        "answer": "<concise final answer>"
    }
    ```

    Now please consider the following question and context:
    context: {{context}}
    question: {{question}}

        """
        return cls(template=_template, input_variables=['context', 'question'])

class ReasoningRewritePrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """You are an expert with deep expertise in finance. Given a context, a question (open-ended or multiple-choice), a reasoning process and an incomplete referral trajectory, your task is to refine and enhance the reasoning process to make it more natural, detailed, and analytical.

    Your refined reasoning should:
    1. Maintain the core insights from the original reasoning
    2. Add more depth and detail to each step
    3. Interleave different types of thinking naturally
    4. Include additional relevant financial concepts and principles
    5. Address any gaps or assumptions in the original reasoning

    Structure Your Thinking Naturally: Use the following tags to structure your refined reasoning process. These should be interleaved naturally as your thinking evolves:
    
    - `<think>...</think>`: Use for general reasoning, analysis, and reflections:
        * Deep analysis of the problem context
        * Application of financial principles
        * Evaluation of different approaches
        * Critical thinking and verification
        * Connecting different aspects of the problem
    
    - `<python>...</python>`: For arithmetic questions or when calculations are needed:
        * First determine if the question requires numerical calculations
        * Clear, well-commented code
        * Step-by-step implementation
        * Results stored in `result` variable
        * Explanation of calculations
    
    - `<search>...</search>`: Specify when additional information is needed:
        * What specific information to search for
        * Why this information is crucial
        * How it relates to the problem
    
    - `<information>...</information>`: Present found information:
        * Relevant data points
        * Key insights
        * How it impacts the analysis
    
    - `<answer>...</answer>`: Provide your final answer:
        * Clear and precise conclusion
        * Option letter for multiple-choice
        * Concise response for open-ended

    IMPORTANT GUIDELINES:
    1. Maintain a natural flow between different types of thinking
    2. Address all aspects of the original reasoning
    3. Ensure all calculations and assumptions are clearly explained
    4. Keep the XML-like tags properly nested and formatted
    5. Your refined reasoning should be comprehensive and detailed, no shorter than the original reasoning process
    6. Elaborate on financial concepts mentioned in the original reasoning
    7. Include step-by-step breakdowns of all calculations with intermediate values
    8. Explicitly state any assumptions made and justify them with financial principles
    9. Consider alternative approaches where applicable before arriving at the final answer
    10. Ensure the final answer is well-justified based on the preceding analysis
    11. MUST use the same language as the question, if the question is in Chinese or French, your refined reasoning process should also be in Chinese or French.

    Return your refined reasoning in the following JSON format:
    ```json
    {
        "rewrite_thinking_trajectory": "<refined reasoning process with all tags>"
    }
    ```

    Now please refine the following reasoning process:
    context: {{context}}
    question: {{question}}
    reasoning_process: {{reasoning_process}}
    reference_trajectory: {{reference_trajectory}}
    """
        return cls(template=_template, input_variables=['context', 'question',
                                                        'reasoning_process',
                                                        'reference_trajectory'])