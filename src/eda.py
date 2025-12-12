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
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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
        n_missing = self.df.isna().sum()
        pct_missing = (n_missing / len(self.df)) * 100

        return (
            pd.DataFrame({
                "column": self.df.columns,
                "n_missing": n_missing.values,
                "pct_missing": pct_missing.values,
            })
            .sort_values("pct_missing", ascending=False)
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
        records = []

        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (self.df[col] < lower) | (self.df[col] > upper)
            n_outliers = mask.sum()

            records.append({
                "column": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": int(n_outliers),
                "pct_outliers": float(n_outliers / len(self.df) * 100),
            })

        return pd.DataFrame(records)

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
