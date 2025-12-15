import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import List, Optional
from xverse.transformer import WOE
import logging

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ----------------------------
# Custom Transformer: Time Features
# ----------------------------
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features (hour, day, month, year) from a datetime column.

    Parameters
    ----------
    datetime_col : str
        Name of the datetime column to extract features from.
    """

    def __init__(self, datetime_col: str):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TimeFeaturesExtractor":
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in DataFrame")
        logging.info("TimeFeaturesExtractor fit completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in DataFrame")

        logging.info("Extracting time features...")
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors='coerce', utc=True)
        X['txn_datetime'] = dt
        X['txn_hour'] = dt.dt.hour
        X['txn_day'] = dt.dt.day
        X['txn_month'] = dt.dt.month
        X['txn_year'] = dt.dt.year
        logging.info("Time features extracted.")
        return X


# ----------------------------
# Custom Transformer: Numeric Aggregation
# ----------------------------
class NumericAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates numeric columns per customer.

    Parameters
    ----------
    group_col : str
        Column name to group by (e.g., 'CustomerId').
    numeric_cols : list of str
        List of numeric columns to aggregate.
    """

    def __init__(self, group_col: str = 'CustomerId', numeric_cols: Optional[List[str]] = None):
        self.group_col = group_col
        self.numeric_cols = numeric_cols or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NumericAggregator":
        missing_cols = [col for col in self.numeric_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Numeric columns missing in DataFrame: {missing_cols}")
        if self.group_col not in X.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame")
        logging.info("NumericAggregator fit completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols:
            return pd.DataFrame({self.group_col: X[self.group_col].unique()})

        logging.info("Aggregating numeric columns...")
        agg_dict = {col: ['sum', 'mean', 'count', 'std'] for col in self.numeric_cols}
        agg = X.groupby(self.group_col).agg(agg_dict)
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg[self.group_col] = agg.index
        logging.info("Numeric aggregation completed.")
        return agg.reset_index(drop=True)


# ----------------------------
# Custom Transformer: Categorical Aggregation (Top-K proportions)
# ----------------------------
class CategoricalTopKAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates categorical columns per customer using top-K category proportions.

    Parameters
    ----------
    group_col : str
        Column name to group by.
    categorical_cols : list of str
        List of categorical columns to aggregate.
    top_k : int
        Number of top categories to consider per column.
    """

    def __init__(self, group_col: str = 'CustomerId', categorical_cols: Optional[List[str]] = None, top_k: int = 3):
        self.group_col = group_col
        self.categorical_cols = categorical_cols or []
        self.top_k = top_k
        self.top_categories_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalTopKAggregator":
        if self.group_col not in X.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame")
        for col in self.categorical_cols:
            if col not in X.columns:
                raise ValueError(f"Categorical column '{col}' not found in DataFrame")
            self.top_categories_[col] = X[col].value_counts().head(self.top_k).index.tolist()
        logging.info("CategoricalTopKAggregator fit completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_cols:
            return pd.DataFrame({self.group_col: X[self.group_col].unique()})

        logging.info("Aggregating top-K categorical features...")
        result_dfs = []
        for col in self.categorical_cols:
            top_cats = self.top_categories_[col]
            for cat in top_cats:
                col_name = f"{col}_{cat}_ratio".replace(" ", "_").lower()
                cat_flag = (X[col] == cat).astype(int)
                cat_ratio = X.assign(**{col_name: cat_flag}).groupby(self.group_col)[col_name].mean()
                result_dfs.append(cat_ratio.rename(col_name))
        df_cat = pd.concat(result_dfs, axis=1).reset_index()
        logging.info("Top-K categorical aggregation completed.")
        return df_cat


# ----------------------------
# Full Feature Engineering Pipeline
# ----------------------------
class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline combining:
    - Time feature extraction
    - Numeric aggregation
    - Categorical top-K aggregation
    - Scaling and imputation

    Parameters
    ----------
    datetime_col : str
        Column containing transaction datetime.
    group_col : str
        Customer identifier column.
    numeric_cols : list of str
        Numeric columns to aggregate.
    categorical_cols : list of str
        Categorical columns to aggregate.
    top_k : int
        Top-K categories to consider for categorical aggregation.
    """

    def __init__(self,
                 datetime_col: str = 'TransactionStartTime',
                 group_col: str = 'CustomerId',
                 numeric_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None,
                 top_k: int = 3):
        self.datetime_col = datetime_col
        self.group_col = group_col
        self.numeric_cols = numeric_cols or ['Amount']
        self.categorical_cols = categorical_cols or []
        self.top_k = top_k

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringPipeline":
        logging.info("Fitting FeatureEngineeringPipeline...")

        # Time features
        self.time_extractor = TimeFeaturesExtractor(self.datetime_col)
        X_time = self.time_extractor.fit_transform(X)

        # Numeric imputer
        self.numeric_imputer = SimpleImputer(strategy='median')
        X_time[self.numeric_cols] = self.numeric_imputer.fit_transform(X_time[self.numeric_cols])

        # Numeric aggregation
        self.numeric_agg = NumericAggregator(group_col=self.group_col, numeric_cols=self.numeric_cols)
        X_num_agg = self.numeric_agg.fit_transform(X_time)

        # Categorical top-K aggregation
        self.cat_agg = CategoricalTopKAggregator(group_col=self.group_col,
                                                 categorical_cols=self.categorical_cols,
                                                 top_k=self.top_k)
        X_cat_agg = self.cat_agg.fit_transform(X_time)

        # Columns for preprocessor
        self.all_numeric_cols = [c for c in X_num_agg.columns if c != self.group_col]
        self.all_categorical_cols = [c for c in X_cat_agg.columns if c != self.group_col]

        # Preprocessor
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), self.all_numeric_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0))]), self.all_categorical_cols)
        ])

        # Fit preprocessor
        X_agg = pd.concat([X_num_agg, X_cat_agg.drop(columns=[self.group_col], errors='ignore')], axis=1)
        self.preprocessor.fit(X_agg)
        logging.info("FeatureEngineeringPipeline fit completed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Transforming data using FeatureEngineeringPipeline...")
        X_time = self.time_extractor.transform(X)
        X_time[self.numeric_cols] = self.numeric_imputer.transform(X_time[self.numeric_cols])

        X_num_agg = self.numeric_agg.transform(X_time)
        X_cat_agg = self.cat_agg.transform(X_time)

        X_agg = pd.concat([X_num_agg, X_cat_agg.drop(columns=[self.group_col], errors='ignore')], axis=1)

        X_processed = self.preprocessor.transform(X_agg)
        all_cols = self.all_numeric_cols + self.all_categorical_cols
        X_processed_df = pd.DataFrame(X_processed, columns=all_cols)
        X_processed_df[self.group_col] = X_num_agg[self.group_col].values

        logging.info("FeatureEngineeringPipeline transformation completed.")
        return X_processed_df


# ----------------------------
# WoE Transformer Wrapper
# ----------------------------
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for xverse WOE transformer.

    Parameters
    ----------
    features : list of str
        List of categorical features to apply WoE transformation.
    
    Attributes
    ----------
    valid_features_ : list
        Features successfully binned by WOE.
    iv_ : pd.DataFrame
        Information value table.
    """

    def __init__(self, features: List[str]):
        self.features = features
        self.valid_features_: List[str] = []
        self.iv_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        if not all(f in X.columns for f in self.features):
            missing = [f for f in self.features if f not in X.columns]
            raise ValueError(f"Features not found in DataFrame: {missing}")

        logging.info("Fitting WoETransformer...")
        self.woe_ = WOE(monotonic_binning=False)
        self.woe_.fit(X[self.features], y)

        # Keep only features that have valid WoE bins
        self.valid_features_ = list(self.woe_.woe_bins.keys())
        self.iv_ = self.woe_.iv_df
        logging.info(f"WoETransformer fit completed. Valid features: {self.valid_features_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.valid_features_:
            raise ValueError("No valid WoE features found. Fit the transformer first.")
        logging.info("Transforming data using WoETransformer...")
        woe_df = self.woe_.transform(X[self.valid_features_])
        woe_df.columns = [f"{c}_WOE" for c in self.valid_features_]
        logging.info("WoE transformation completed.")
        return woe_df
