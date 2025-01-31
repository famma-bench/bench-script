import pandas as pd

def calculate_accuracy(df, target_col='is_correct_by_model', group_by=None):
        """Calculate accuracy of the model's answers."""
        if group_by:
            grouped = df.groupby(group_by)
            accuracy = grouped[target_col].mean()
        else:
            accuracy = df[target_col].mean()
        if isinstance(accuracy, pd.Series):
            accuracy = accuracy.to_dict()
            # Convert tuple keys to string keys if grouping by multiple columns
            if any(isinstance(k, tuple) for k in accuracy.keys()):
                accuracy = {str(k[0]) + "_" + str(k[1]): v for k, v in accuracy.items()}
        # Convert numpy/pandas numerical types to native Python float
        if isinstance(accuracy, dict):
            accuracy = {k: float(v) for k, v in accuracy.items()}
        else:
            accuracy = float(accuracy)
        return accuracy