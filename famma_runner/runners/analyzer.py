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
        overall_acc = calculate_accuracy(self.dataset_df)
        self.metrics["overall_acc"] = overall_acc

        overall_acc_by_subfield = calculate_accuracy(self.dataset_df, group_by=[DC.SUBFIELD])
        self.metrics["overall_acc_by_subfield"] = overall_acc_by_subfield.to_dict()

        # for the overall accuracy, we compute the accuracy by difficulty
        overall_acc_by_difficulty = calculate_accuracy(self.dataset_df, group_by=[DC.TOPIC_DIFFICULTY])
        self.metrics["overall_acc_by_difficulty"] = overall_acc_by_difficulty.to_dict()

        # then we repeat this process for each language subset
        for language in LANGUAGE_ORDER:
            language_acc = calculate_accuracy(self.dataset_df, group_by=[DC.LANGUAGE, DC.TOPIC_DIFFICULTY])
            # convert the accuracy to a dictionary
            # merge the language and difficulty into a single key
            language_acc = {f"{language}_{difficulty}": acc for difficulty, acc in language_acc.items()}
            self.metrics[f"overall_acc_by_difficulty_{language}"] = language_acc
            # we compute the accuracy by difficulty for each language
            language_acc_by_difficulty = calculate_accuracy(self.dataset_df,
                                                            group_by=[DC.LANGUAGE, DC.TOPIC_DIFFICULTY])
            # merge the language and difficulty into a single key   
            language_acc_by_difficulty = {f"{language}_{difficulty}": acc for difficulty, acc in language_acc_by_difficulty.items()}
            self.metrics[f"overall_acc_by_difficulty_{language}"] = language_acc_by_difficulty

        write_to_database(self.target_db_name, 'metrics', self.metrics)
