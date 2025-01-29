

def calculate_accuracy(df, target_col='is_correct_by_model', group_by=None):
        """Calculate accuracy of the model's answers."""
        if group_by:
            grouped = df.groupby(group_by)
            accuracy = grouped[target_col].mean()
        else:
            accuracy = df[target_col].mean()
        
        return accuracy