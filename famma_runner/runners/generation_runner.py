from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
import pandas as pd
import os
from famma_runner.runners.base_runner import Runner
from famma_runner.utils import collect_images_from_first_subquestion, generate_response_from_llm
from famma_runner.utils import QuestionPrompt


logger = get_logger('generation_runner', 'generation_runner.log')


@Runner.register("generation")
class GenerationRunner(Runner):
    def __init__(self, config):
        self.model_config = config["model"]
        self.generation_config = config["generation"]
        self.data_config = config["data"]
        self.config = config

        self.llm = self.setup_model()
        self.llm_name = self.llm.model_name

        self.dataset_df = self.setup_dataset()

        # Initialize the DDB
        self.target_db_name = f'{self.llm_name}_DDB'
        self.target_db = initialize_database(output_db=self.target_db_name)

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.config.get('model', None),
                      'generation_config': self.config.get('generation', None), }
        llm = LLM.build_from_config(llm_config)

        return llm

    def setup_dataset(self):
        # convert dataset to DataFrame for easy processing
        data = read_json(self.data_config.data_dir)
        dataset_df = pd.DataFrame(data)
        return dataset_df
    
    def generate_answer_for_one_main_question(self, sub_question_set_df):
        """
        Generates model answer and explanation for a subset of questions, including both multiple-choice 
        and open-ended types.
        """
        # Get the context from the first sub_question
        context = sub_question_set_df.iloc[0].get("context", "")

        # Collect images from the first sub-question
        # parent_dir is the parent directory of the dataset - self.data_config.data_dir
        parent_dir = os.path.dirname(self.data_config.data_dir)
        images = collect_images_from_first_subquestion(sub_question_set_df, parent_dir=parent_dir)

        sub_questions = []

        for idx, row in sub_question_set_df.iterrows():
            question = f"subquestion {idx} - {row['question_type']} - {row['question']}"
            
            if row['question_type'] == 'multiple-choice':
                options = row['options']
                question += f" {options}"
            
            sub_questions.append(question)

        prompt = QuestionPrompt.init().format(
            context=context,
            sub_questions=sub_questions
        )
        
        model_response = generate_response_from_llm(self.llm, prompt, images)

        return model_response

    def run(self):
        dataset_df = self.dataset_df.copy()
        for (main_question_id, language), group in dataset_df.groupby(['main_question_id', 'language']):
            key = f'{main_question_id}_{language}'
            if key in self.target_db:
                continue
            try:
                logger.info(f'start generating answers for {key}')
                model_response = self.generate_answer_for_one_main_question(group)
                
                # Add model response to the output
                # iterate over group DataFrame
                for idx in range(len(group)):
                    row = group.iloc[idx]
                    row['model_answer'] = model_response['sub-question-{}'.format(idx)]['answer']
                    row['model_explanation'] = model_response['sub-question-{}'.format(idx)]['explanation']
                
                write_to_database(self.target_db_name, key, model_response)
            except Exception as e:
                logger.error(
                    "Error processing main_question_id %s: %s", key, str(e))
                continue
        
        # Save the DataFrame to a file or database as needed
        dataset_df.to_csv('output_samples.csv', index=False)
        
        logger.info('Generation complete') 
        logger.info('Result saved to %s in json format', self.target_db_name)
        logger.info('Result saved to %s in csv format', 'output_samples.csv')

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[prompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        if remaining_prompts:
            vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                        remaining_indices, remaining_prompts, vllm_outputs
                ):
                    self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs

    def prompts_to_outputs(
            self, prompts: list[str | list[dict[str, str]]]
    ) -> list[list[str]]:
        if self.args.use_cache:
            outputs = []
            batch_size = self.args.cache_batch_size
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i: i + batch_size]
                batch_outputs = self.run_batch(batch)
                outputs.extend(batch_outputs)
                self.save_cache()
        else:
            outputs = self.run_batch(prompts)
        return outputs

    def generate_outputs(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        prompts = [
            format_prompt(problem, self.model.model_style) for problem in benchmark
        ]
        outputs = self.prompts_to_outputs(prompts)
        return outputs