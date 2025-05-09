from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments
import pandas as pd
import os
import omegaconf
from famma_runner.runners.base_runner import Runner
from famma_runner.utils import collect_images_from_first_subquestion, generate_response_from_llm, safe_parse_response
from famma_runner.utils import QuestionPrompt, LANGUAGE_ORDER, DC, order_by_language, ProgramOfThoughtsQuestionPrompt

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
        # for each question_id, we need to find out its main_question_id and language
        # then filter the dataset by main_question_id and language  
        self.dataset_df, self.filtered_main_question_ids = self.filter_dataset_by_question_id(self.dataset_df, self.data_config.question_id)

        # Initialize the DDB
        release_version = self.data_config.data_dir.split('/')[-1].split('.')[0]
        self.target_db_name = f'{self.llm_name}_ans_{release_version}'
        self.target_db = initialize_database(output_db=self.target_db_name)

        self.use_pot = self.model_config.get('use_pot', False)
        self.is_reasoning_model = self.model_config.get('is_reasoning_model', False)
        self.ocr_model = None
        self.use_ocr = self.model_config.get('use_ocr', False)
        if self.use_ocr:
            # ref: https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html#11-install-paddlepaddle
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(use_angle_cls=True)

    def filter_dataset_by_question_id(self, dataset_df, question_ids):
        """
        Filter dataset by specific question_ids.
        
        Args:
            dataset_df: The dataset DataFrame to filter
            question_ids: A single question_id or list of question_ids in format 
                         {language}_{main_question_id}_{sub_question_id}_{version}
            
        Returns:
            Filtered DataFrame containing only rows matching the main_question_ids and languages
        """
        filtered_main_question_ids = None
        if not question_ids:
            return dataset_df, filtered_main_question_ids
        
        # Convert single question_id to list for consistent processing
        if not isinstance(question_ids, omegaconf.ListConfig):
            question_ids = [question_ids]
        
        if len(question_ids) == 0:
            return dataset_df, filtered_main_question_ids
        
        # Create empty DataFrame to collect all filtered results
        filtered_results = pd.DataFrame()
        
        # Track unique language-main_question_id pairs to avoid duplicate processing
        processed_pairs = set()
        
        for question_id in question_ids:
            try:
                # Parse the question_id to extract components
                parts = question_id.split('_')
                if len(parts) >= 3:  # Ensure we have at least language, main_question_id, and sub_question_id
                    language = parts[0]
                    main_question_id = int(parts[1])
                    
                    # Create a unique identifier for this language-main_question_id pair
                    pair_key = f"{language}_{main_question_id}"
                    
                    # Skip if we've already processed this pair
                    if pair_key in processed_pairs:
                        # logger.info(f"Skipping duplicate filter for {pair_key}")
                        continue
                    
                    processed_pairs.add(pair_key)
                    
                    logger.info(f"Filtering dataset for language: {language}, main_question_id: {main_question_id}")
                    
                    # Filter the dataset by both main_question_id and language
                    current_filtered = dataset_df[
                        (dataset_df[DC.MAIN_QUESTION_ID] == main_question_id) & 
                        (dataset_df[DC.LANGUAGE] == language)
                    ]
                    
                    if current_filtered.empty:
                        logger.warning(f"No matching questions found for {question_id}")
                    else:
                        logger.info(f"Found {len(current_filtered)} questions matching {question_id}")
                        # Append to our results
                        filtered_results = pd.concat([filtered_results, current_filtered])
                else:
                    logger.warning(f"Invalid question_id format: {question_id}")
            except Exception as e:
                logger.error(f"Error parsing question_id {question_id}: {str(e)}")
        
        # If we didn't find any matches, return the original dataset
        if filtered_results.empty:
            logger.warning("No matching questions found for any of the provided question_ids")
            return dataset_df

        logger.info(f"Total of {len(filtered_results)} questions matched across all filters")
        
        # Extract unique main_question_ids from filtered results
        filtered_main_question_ids = filtered_results[DC.MAIN_QUESTION_ID].unique().tolist()
        return filtered_results, filtered_main_question_ids

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.model_config,
                      'generation_config': self.generation_config}

        # If using custom model, load it from custom_llm.py
        if self.model_config.model_name == "custom_llm":
            pass
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
        if self.use_pot:
            prompt = ProgramOfThoughtsQuestionPrompt.init().format(
                context=context,
                sub_questions=sub_questions
            )
        else:
            prompt = QuestionPrompt.init().format(
                context=context,
                sub_questions=sub_questions
            )

        model_output = generate_response_from_llm(self.llm, prompt, images, use_ocr=self.use_ocr, ocr_model=self.ocr_model)
        model_response = safe_parse_response(model_output, question_id_list,is_reasoning_model=self.is_reasoning_model)

        return model_response

    def run(self):
        # Create a copy of the DataFrame at the start
        dataset_df = self.dataset_df.copy()

        for (_, language, main_question_id), group in dataset_df.groupby(
                ['language_order', DC.LANGUAGE, DC.MAIN_QUESTION_ID]):
            key = f'{language}_{main_question_id}'
            # Skip if already in database AND not specifically requested in filtered_main_question_ids
            if key in self.target_db and (self.filtered_main_question_ids is None or main_question_id not in self.filtered_main_question_ids):
            # Skip this question since it's already in the database and not specifically requested
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
                    
                    # Always include model_answer and model_explanation
                    response_update = {
                        'model_answer': model_response[output_key]['answer'],
                        'model_explanation': model_response[output_key]['explanation']
                    }

                    # Only include model_reasoning if self.is_reasoning_model is True
                    if self.is_reasoning_model and 'reasoning' in model_response:
                        response_update['model_reasoning'] = model_response['reasoning']
                
                    input_data_with_response.update(response_update)

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
