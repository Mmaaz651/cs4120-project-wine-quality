# src/utils.py

import pandas as pd

def summarize_metrics(task_name, results_dict):
    """
    Convert a metrics dictionary into a formatted DataFrame
    and print it for summary.
    """
    df = pd.DataFrame(results_dict).T
    print(f"\n===== {task_name} Metrics =====")
    print(df.round(4))
    print()
    return df
