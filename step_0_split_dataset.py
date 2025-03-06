import pandas as pd
from famma_runner.utils.data_utils import sample_questions

if __name__ == "__main__":
    """
    Split dataset into train and test for v2406 version
    """

    df = pd.read_csv("source_data/FAMMA v1203 - merge.csv", header=0)
    sampled_df, res_df = sample_questions(df)

    sampled_df.to_csv("source_data/FAMMA-test.csv", index=False)
    res_df.to_csv("source_data/FAMMA-train.csv", index=False)
