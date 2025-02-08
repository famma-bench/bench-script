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

        to_analyze_db_name = self.data_config.data_dir.split("/")[-1].split(".")[0]
        # Form the target database name using the extracted names
        self.target_db_name = f'{to_analyze_db_name}_result'
        self.target_db = initialize_database(output_db=self.target_db_name)

        self.dataset_df = self.setup_dataset()
        self.metrics = {}

    def setup_dataset(self):
        # Load the dataset
        dataset_json = read_json(self.data_config.data_dir)
        # Convert the JSON data into a list of dictionaries
        records = []
        for _, details in dataset_json.items():
            records.append(details)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(records)
        df.to_csv('temp.csv', index=False)
        df['is_correct_by_model'] = df['is_correct_by_model'].map(lambda x: 1 if x == 'correct' else 0)
        return df

    def run(self):
        self.metrics['total_questions'] = len(self.dataset_df)
        # add the total number of english questions
        self.metrics['total_english_questions'] = len(self.dataset_df[self.dataset_df[DC.LANGUAGE] == 'english'])
        # add the total number of chinese questions
        self.metrics['total_chinese_questions'] = len(self.dataset_df[self.dataset_df[DC.LANGUAGE] == 'chinese'])
        # add the total number of french questions
        self.metrics['total_french_questions'] = len(self.dataset_df[self.dataset_df[DC.LANGUAGE] == 'french'])
        overall_acc = calculate_accuracy(self.dataset_df)
        self.metrics["overall_acc"] = overall_acc

        overall_acc_by_subfield = calculate_accuracy(self.dataset_df, group_by=[DC.SUBFIELD])
        self.metrics["overall_acc_by_subfield"] = overall_acc_by_subfield

        # for the overall accuracy, we compute the accuracy by difficulty
        overall_acc_by_difficulty = calculate_accuracy(self.dataset_df, group_by=[DC.TOPIC_DIFFICULTY])
        self.metrics["overall_acc_by_difficulty"] = overall_acc_by_difficulty

        # Calculate overall accuracy by language
        overall_acc_by_language = calculate_accuracy(self.dataset_df, group_by=[DC.LANGUAGE])
        self.metrics["overall_acc_by_language"] = overall_acc_by_language

        # then we repeat this process for each language subset
        for language in LANGUAGE_ORDER:
            # Filter the dataset for the current language
            language_df = self.dataset_df[self.dataset_df[DC.LANGUAGE] == language]
            
            # Calculate accuracy for the filtered dataset
            language_acc = calculate_accuracy(language_df, group_by=[DC.TOPIC_DIFFICULTY])
            
            # Convert the accuracy to a dictionary
            self.metrics[f"overall_acc_by_difficulty_{language}"] = language_acc

        write_to_database(self.target_db_name, 'metrics', convert_to_dict(self.metrics))
