from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments
import pandas as pd
import os
from famma_runner.runners.base_runner import Runner
from famma_runner.utils import collect_images_from_first_subquestion, generate_response_from_llm, safe_parse_response
from famma_runner.utils import QuestionPrompt, LANGUAGE_ORDER, DC, order_by_language

logger = get_logger('generation_runner', 'generation_runner.log')


@Runner.register("generation")
class GenerationRunner(Runner):
    def __init__(self, config):
        self.model_config = config["model"]
        self.generation_config = GenerationArguments(**config.get('generation', {}))
        self.data_config = config["data"]
        self.config = config

        self.llm = self.setup_model()
        self.llm_name = self.llm.model_config.model_full_name

        self.dataset_df = self.setup_dataset()

        # filter the dataset by main_question_id
        if self.data_config.main_question_id is not None:
            self.dataset_df = self.dataset_df[self.dataset_df['main_question_id'] == self.data_config.main_question_id]

        # Initialize the DDB
        release_version = self.data_config.data_dir.split('/')[-1].split('.')[0]
        self.target_db_name = f'{self.llm_name}_ans_{release_version}'
        self.target_db = initialize_database(output_db=self.target_db_name)

        self.ocr_model = None
        self.use_ocr = self.model_config.get('use_ocr', False)
        if self.use_ocr:
            # ref: https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html#11-install-paddlepaddle
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(use_angle_cls=True)

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.model_config,
                      'generation_config': self.generation_config}

        # If using custom model, load it from custom_llm.py
        if self.model_config.model_name == "custom_llm":
            from custom_llm import MyCustomModel
        llm = LLM.build_from_config(llm_config)

        return llm

    def setup_dataset(self):
        # convert dataset to DataFrame for easy processing
        data = read_json(self.data_config.data_dir)
        dataset_df = pd.DataFrame(data)
        # Create a new column for sorting languages
        order_by_language(dataset_df, LANGUAGE_ORDER, DC.MAIN_QUESTION_ID, DC.SUB_QUESTION_ID, DC.LANGUAGE)

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
        question_id_list = []
        for _, row in sub_question_set_df.iterrows():
            question_dict = {
                "id": row['question_id'],
                "type": row['question_type'],
                "question": row['question']
            }

            # Add options if it's a multiple-choice question
            if row['question_type'] == 'multiple-choice':
                question_dict["options"] = row['options']

            sub_questions.append(question_dict)
            question_id_list.append(row['question_id'])

        # Format the prompt with the structured questions
        prompt = QuestionPrompt.init().format(
            context=context,
            sub_questions=sub_questions
        )

        model_output = generate_response_from_llm(self.llm, prompt, images, use_ocr=self.use_ocr, ocr_model=self.ocr_model)
        model_response = safe_parse_response(model_output, question_id_list)

        return model_response

    def run(self):
        # Create a copy of the DataFrame at the start
        dataset_df = self.dataset_df.copy()

        for (_, language, main_question_id), group in dataset_df.groupby(
                ['language_order', DC.LANGUAGE, DC.MAIN_QUESTION_ID]):
            key = f'{language}_{main_question_id}'
            if key in self.target_db:
                continue
            try:
                logger.info(f'start generating answers for {language} -- main_question_id: {main_question_id}')
                model_response = self.generate_answer_for_one_main_question(group)

                # Aggregate all subquestions with their answers into a single dictionary
                subquestion_responses = {}
                for idx in range(len(group)):
                    output_key = group.iloc[idx]['question_id']

                    # Create a JSON object with the original input data and the model response
                    input_data_with_response = group.iloc[idx].to_dict()
                    input_data_with_response.update({
                        'model_answer': model_response[output_key]['answer'],
                        'model_explanation': model_response[output_key]['explanation']
                    })

                    # Store the response in the subquestion_responses dictionary
                    subquestion_responses[output_key] = input_data_with_response

                # Write the aggregated subquestion responses to the database
                write_to_database(self.target_db_name, key, subquestion_responses)
            except Exception as e:
                logger.error(
                    "Error processing main_question_id %s: %s", main_question_id, str(e))
                continue

        # Save the DataFrame to a file
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
