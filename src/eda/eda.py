"""
EDA helper class for generating non-plot exploratory summaries.

All methods return structured Python objects (DataFrames, dicts).
"""

import pandas as pd
import numpy as np


class EDAHelper:
    """
    Perform structured exploratory data analysis on a dataframe
    without printing or plotting.
    """

    def __init__(self, df):
        """
        Initialize the helper and detect numeric and categorical columns.
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    def overview(self):
        """
        Return basic dataset information:
        number of rows, columns, data types, and memory usage.
        """
        return {
            "n_rows": self.df.shape[0],
            "n_cols": self.df.shape[1],
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str),
        }

    def missing_values(self):
        """
        Return a table with missing value counts and percentages.
        """
        missing_count = self.df.isna().sum()
        missing_values_percentage = (missing_count / len(self.df)) * 100

        return (
            pd.DataFrame(
                {
                    "column": self.df.columns,
                    "missing_count": missing_count.values,
                    "missing_values_percentage": missing_values_percentage.values,
                }
            )
            .sort_values("missing_values_percentage", ascending=False)
            .reset_index(drop=True)
        )

    def numeric_summary(self):
        """
        Return summary statistics for numeric columns.
        """
        if not self.numeric_cols:
            return pd.DataFrame()
        return self.df[self.numeric_cols].describe().T

    def categorical_summary(self, top_n=10):
        """
        Return value counts (top N) for each categorical column.
        """
        results = {}
        for col in self.categorical_cols:
            results[col] = self.df[col].value_counts().head(top_n)
        return results

    def correlation_matrix(self):
        """
        Return a correlation matrix for numeric columns.
        """
        if not self.numeric_cols:
            return pd.DataFrame()
        return self.df[self.numeric_cols].corr()

    def outlier_summary(self):
        """
        Return outlier statistics for each numeric column using the IQR method.
        """
        q1 = self.df[self.numeric_cols].quantile(0.25)
        q3 = self.df[self.numeric_cols].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers_count = (
            (self.df[self.numeric_cols] < lower) | (self.df[self.numeric_cols] > upper)
        ).sum()
        outlier_percentage = outliers_count / len(self.df) * 100

        summary_df = pd.DataFrame(
            {
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outliers Count": outliers_count,
                "Outliers %": outlier_percentage,
            }
        )

        return summary_df

    def run_all(self, top_n=10):
        """
        Run all EDA components and return results in a dictionary.
        """
        return {
            "overview": self.overview(),
            "missing_values": self.missing_values(),
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(top_n),
            "correlation_matrix": self.correlation_matrix(),
            "outlier_summary": self.outlier_summary(),
        }