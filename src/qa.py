"""
QA and logger

- Check if there's missing date
- Check if there's null value
- Check if there's outlier
- etc.
"""

import pandas as pd
from loguru import logger


def log_data_summary(df, df_name, df_desc, cols=None, show_col=False, transpose=True):
    df = df.copy()
    if show_col:
        logger.info(list(df))
    if cols:
        df = df[cols]
    df_detail = df.describe(percentiles=[0.25, 0.75, 0.99])
    if transpose:
        df_detail = df_detail.T
    logger.info(
        f"{df_desc} overview\n{df_name} shape: {df.shape}\n{df_name} summary:\n{df_detail.to_string()}"
    )


def check_negative_values(df, col):
    df = df.copy()
    df = df[df[col] < 0]
    if len(df) > 0:
        logger.warning(f"For {col}, there are {len(df)} records with negative value!")


if __name__ == "__main__":
    df_sample = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    x = 1
    log_data_summary(df_sample, "df_sample", "Sample data")
