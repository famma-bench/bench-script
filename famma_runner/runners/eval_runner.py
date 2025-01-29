from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json, extract_json_from_text, convert_to_dict
from easyllm_kit.models import LLM
import pandas as pd
import os

from famma_runner.runners.base_runner import Runner
from famma_runner.utils import generate_response_from_llm, DC, LANGUAGE_ORDER, order_by_language, JudgePrompt

logger = get_logger('eval_runner', 'eval_runner.log')


@Runner.register("evaluation")
class EvaluationRunner(Runner):
    def __init__(self, config):
        self.model_config = config["model"]
        self.generation_config = config["generation"]
        self.data_config = config["data"]
        self.config = config

        self.llm = self.setup_model()
        self.llm_name = self.llm.model_name

        self.answers_df, self.gold_df = self.setup_dataset()

        # Extract model name and judger name from the respective directories
        model_name = os.path.basename(self.data_config.data_dir).split('.')[
            0]  # Assuming the model name is part of the file name
        judger_name = self.model_config.model_name

        # Form the target database name using the extracted names
        self.target_db_name = f'{model_name}_evaluated_by_{judger_name}'
        self.target_db = initialize_database(output_db=self.target_db_name)

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.config.get('model', None),
                      'generation_config': self.config.get('generation', None), }
        llm = LLM.build_from_config(llm_config)

        return llm

    @staticmethod
    def json_to_df(json_dir):
        data = read_json(json_dir)
        data_list = []
        # loop over the keys
        for _, v in data.items():
            # look over the values
            for _, sub_question in v.items():
                data_list.append(sub_question)
        df = pd.DataFrame(data_list)
        order_by_language(df, LANGUAGE_ORDER, DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID, DC.LANGUAGE)
        return df

    def setup_dataset(self):
        # convert dataset to DataFrame for easy processing
        answers_df = self.json_to_df(self.data_config.data_dir)
        if self.data_config.gold_dir is None or self.data_config.gold_dir == self.data_config.data_dir:
            gold_df = answers_df.copy()
        else:
            gold_json = read_json(self.data_config.gold_dir)
            gold_df = pd.DataFrame(gold_json)
            order_by_language(gold_df, LANGUAGE_ORDER, DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID, DC.LANGUAGE)

        return answers_df, gold_df

    def get_gold_answer(self, question_id):
        return self.gold_df[self.gold_df[DC.QUESTION_ID] == question_id][DC.ANSWER].values[0]

    def run(self):
        # We use gold_df to judge the answers   
        gold_df = self.gold_df.copy()
        for _, group in gold_df.groupby(['language_order', DC.LANGUAGE, DC.MAIN_QUESTION_ID]):
            for idx in range(len(group)):
                row = group.iloc[idx]
                key = row['question_id']

                if key in self.target_db:
                    continue

                logger.info(f'start judging answers for {key}')

                student_row = self.answers_df[self.answers_df[DC.QUESTION_ID] == key]

                # Ensure student_row is not empty and make a copy of the row
                data_to_save = row.to_dict()
                if not student_row.empty:
                    student_row = student_row.iloc[0].copy()

                    data_to_save['model_answer'] = student_row['model_answer']
                    data_to_save['model_explanation'] = student_row['model_explanation']

                    judge_response = self.judge_answer_for_one_subquestion(data_to_save)

                    # Convert the row to a dictionary and add 'is_correct_by_model'
                    data_to_save['is_correct_by_model'] = judge_response[key]
                else:
                    logger.warning(f'No student row found for question_id: {key}, set is_correct_by_model to False')
                    data_to_save['model_answer'] = None
                    data_to_save['model_explanation'] = None
                    data_to_save['is_correct_by_model'] = False

                # Ensure all values in data_to_save are JSON serializable
                data_to_save = convert_to_dict(data_to_save)

                write_to_database(self.target_db_name, key, data_to_save)

        # Save the DataFrame to a file or database as needed
        gold_df.to_csv('output_samples.csv', index=False)

        logger.info('Judging complete')
        logger.info('Result saved to %s in json format', self.target_db_name)
        logger.info('Result saved to %s in csv format', 'output_samples.csv')

    def judge_answer_for_one_subquestion(self, gold_row):
        question = {
            'question_id': gold_row[DC.QUESTION_ID],
            'context': gold_row[DC.CONTEXT],
            'question_type': gold_row[DC.QUESTION_TYPE],
            'question': gold_row[DC.QUESTION] + gold_row[DC.OPTIONS] if gold_row[
                                                                            DC.QUESTION_TYPE] == 'multiple_choice' else
            gold_row[DC.QUESTION],
            'student_answer': gold_row['model_answer'],  # attach the model_answer to the question
            'student_explanation': gold_row['model_explanation'],  # attach the model_explanation to the question
            'ground_truth': gold_row[DC.ANSWER]
        }
        prompt = JudgePrompt.init().format(
            question=question
        )

        model_response = generate_response_from_llm(self.llm, prompt)
        model_response = extract_json_from_text(model_response)

        return model_response
