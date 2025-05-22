from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json, convert_to_dict
import pandas as pd
from famma_runner.runners.base_runner import Runner
from famma_runner.utils import DC, LANGUAGE_ORDER, calculate_accuracy

logger = get_logger('analyzer', 'analyzer.log')


@Runner.register("analyzer")
class Analyzer(Runner):
    def __init__(self, config):
        self.data_config = config["data"]
        self.config = config

        to_analyze_db_name = self.data_config.model_name_to_eval
        # Form the target database name using the extracted names
        self.target_db_name = f'{to_analyze_db_name}_result'
        self.target_db = initialize_database(output_db=self.target_db_name)

        self.dataset_df = self.setup_dataset()
        self.metrics = {}
        self.correct_question_ids = {}

    def setup_dataset(self):
        # Load the dataset
        dataset_json = read_json(self.data_config.data_dir)
        # Convert the JSON data into a list of dictionaries
        records = []
        for _, details in dataset_json.items():
            records.append(details)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(records)
        df['is_correct_by_model'] = df['is_correct_by_model'].map(lambda x: 1 if x == 'correct' or x is True else 0)
        return df

    def run(self):
        # Define analysis types and their corresponding filters
        analysis_types = {
            'consolidated': lambda df: df,  # No filter, use all data
            'arithmetic': lambda df: df[df['is_arithmetic'] == '1'],
            'no_arithmetic': lambda df: df[df['is_arithmetic'] == '0']
        }

        for analysis_type, filter_func in analysis_types.items():
            # Filter the dataset based on the current analysis type
            current_df = filter_func(self.dataset_df)

            # Create metrics dictionary for current analysis type
            current_metrics = {}

            # Basic counts
            current_metrics['total_questions'] = len(current_df)
            current_metrics['total_english_questions'] = len(current_df[current_df[DC.LANGUAGE] == 'english'])
            current_metrics['total_chinese_questions'] = len(current_df[current_df[DC.LANGUAGE] == 'chinese'])
            current_metrics['total_french_questions'] = len(current_df[current_df[DC.LANGUAGE] == 'french'])

            # Calculate accuracies
            print("Total questions:", len(current_df))
            print("Sum correct:", current_df['is_correct_by_model'].sum())
            print("Mean (accuracy):", current_df['is_correct_by_model'].mean())
            print("Manual acc:", current_df['is_correct_by_model'].sum() / len(current_df))

            overall_acc = calculate_accuracy(current_df)
            current_metrics["overall_acc"] = overall_acc

            overall_acc_by_subfield = calculate_accuracy(current_df, group_by=[DC.SUBFIELD])
            current_metrics["overall_acc_by_subfield"] = overall_acc_by_subfield

            overall_acc_by_difficulty = calculate_accuracy(current_df, group_by=[DC.TOPIC_DIFFICULTY])
            current_metrics["overall_acc_by_difficulty"] = overall_acc_by_difficulty

            overall_acc_by_language = calculate_accuracy(current_df, group_by=[DC.LANGUAGE])
            current_metrics["overall_acc_by_language"] = overall_acc_by_language

            # Language-specific analysis
            current_correct_question_ids = {}

            for language in LANGUAGE_ORDER:
                language_df = current_df[current_df[DC.LANGUAGE] == language]

                language_acc = calculate_accuracy(language_df, group_by=[DC.TOPIC_DIFFICULTY])
                current_metrics[f"overall_acc_by_difficulty_{language}"] = language_acc

                for difficulty in language_df[DC.TOPIC_DIFFICULTY].unique():
                    difficulty_df = language_df[language_df[DC.TOPIC_DIFFICULTY] == difficulty]
                    correct_ids = difficulty_df[difficulty_df['is_correct_by_model'] == 1][DC.QUESTION_ID].tolist()
                    current_correct_question_ids[f"{language}_{difficulty}"] = correct_ids

            # Save correct_question_ids into metrics
            if self.data_config.save_question_ids:
                current_metrics["correct_question_ids"] = current_correct_question_ids

            # Store the metrics for this analysis type
            self.metrics[analysis_type] = current_metrics

            # Log summary for current analysis type
            logger.info(f"\nAnalysis results for {analysis_type}:")
            logger.info(f"Total questions: {current_metrics['total_questions']}")
            logger.info(f"Overall accuracy: {current_metrics['overall_acc']:.2%}")
            logger.info(f"Accuracy by language: {current_metrics['overall_acc_by_language']}")

        # Write all metrics to database
        write_to_database(self.target_db_name, 'metrics', convert_to_dict(self.metrics))
