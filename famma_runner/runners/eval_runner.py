from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
import pandas as pd
import os
from famma_runner.runners.base_runner import Runner
from famma_runner.utils import generate_response_from_llm
from famma_runner.utils import JudgePrompt
from datetime import datetime

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

        # Initialize the DDB
        # The name is timestamped + model name
        self.target_db_name = f'{self.llm_name}_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.target_db = initialize_database(output_db=self.target_db_name)

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.config.get('model', None),
                      'generation_config': self.config.get('generation', None), }
        llm = LLM.build_from_config(llm_config)

        return llm

    def setup_dataset(self):
        # convert dataset to DataFrame for easy processing
        answers_data = read_json(self.data_config.data_dir)
        if self.data_config.gold_dir is None or self.data_config.gold_dir == self.data_config.data_dir:
            gold_data = answers_data.copy()
        else:
            gold_data = read_json(self.data_config.gold_dir)
        answers_df = pd.DataFrame(answers_data)
        gold_df = pd.DataFrame(gold_data)
        return answers_df, gold_df

    def run(self):
        dataset_df = self.dataset_df.copy()
        for (main_question_id, language), group in dataset_df.groupby(['main_question_id', 'language']):
            for idx in range(len(group)):
                row = group.iloc[idx]
                key = row['question_id']

                if key in self.target_db:
                    continue

                logger.info(f'start judging answers for {main_question_id}')
                model_response = self.judge_answer_for_one_subquestion(row)
                
                row['is_correct_by_model'] = model_response[key]
                
                write_to_database(self.target_db_name, key, model_response)
            

        # Save the DataFrame to a file or database as needed
        dataset_df.to_csv('output_samples.csv', index=False)

        logger.info('Generation complete')
        logger.info('Result saved to %s in json format', self.target_db_name)
        logger.info('Result saved to %s in csv format', 'output_samples.csv')

    def _run_single(self, prompt: str) -> list[str]:
        pass