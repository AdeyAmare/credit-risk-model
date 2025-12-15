import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Optional, Union
import logging

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProxyTargetGenerator:
    """
    Generates a proxy credit risk target variable (`is_high_risk`) 
    based on RFM segmentation using KMeans clustering.

    Parameters
    ----------
    customer_col : str
        Name of the column identifying customers. Default is 'CustomerId'.
    date_col : str
        Name of the transaction date column. Default is 'TransactionStartTime'.
    value_col : str
        Name of the transaction value column. Default is 'Value'.
    snapshot_date : str or pd.Timestamp, optional
        Reference date for recency calculation. Defaults to one day after the latest transaction.
    n_clusters : int
        Number of clusters to form using KMeans. Default is 3.
    random_state : int
        Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        customer_col: str = 'CustomerId',
        date_col: str = 'TransactionStartTime',
        value_col: str = 'Value',
        snapshot_date: Optional[Union[str, pd.Timestamp]] = None,
        n_clusters: int = 3,
        random_state: int = 42
    ):
        self.customer_col = customer_col
        self.date_col = date_col
        self.value_col = value_col
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        logging.info("ProxyTargetGenerator initialized.")

    # -----------------------------
    # 1️⃣ Calculate RFM metrics
    # -----------------------------
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics per customer.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least customer, date, and value columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [customer_col, recency, frequency, monetary].

        Raises
        ------
        ValueError
            If required columns are missing or date conversion fails.
        """
        logging.info("Calculating RFM metrics...")
        required_cols = {self.customer_col, self.date_col, self.value_col}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df_copy = df.copy()
        try:
            df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error converting {self.date_col} to datetime: {e}")

        snapshot = self.snapshot_date
        if snapshot is None:
            snapshot = df_copy[self.date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot = pd.to_datetime(snapshot)

        rfm = df_copy.groupby(self.customer_col).agg(
            recency=(self.date_col, lambda x: (snapshot - x.max()).days),
            frequency=(self.date_col, 'count'),
            monetary=(self.value_col, 'sum')
        ).reset_index()
        logging.info("RFM metrics calculated.")
        return rfm

    # -----------------------------
    # 2️⃣ Scale RFM features
    # -----------------------------
    def scale_rfm(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the RFM features (recency, frequency, monetary).

        Parameters
        ----------
        rfm : pd.DataFrame
            DataFrame containing columns ['recency', 'frequency', 'monetary'].

        Returns
        -------
        pd.DataFrame
            Scaled RFM features with same column names.

        Raises
        ------
        ValueError
            If required RFM columns are missing.
        """
        logging.info("Scaling RFM features...")
        required_cols = {'recency', 'frequency', 'monetary'}
        if not required_cols.issubset(rfm.columns):
            missing = required_cols - set(rfm.columns)
            raise ValueError(f"Missing RFM columns: {missing}")

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
        self.scaler = scaler
        logging.info("RFM features scaled.")
        return pd.DataFrame(rfm_scaled, columns=['recency', 'frequency', 'monetary'])

    # -----------------------------
    # 3️⃣ Cluster customers
    # -----------------------------
    def cluster_customers(self, rfm_scaled: pd.DataFrame) -> pd.Series:
        """
        Cluster customers using KMeans on scaled RFM features.

        Parameters
        ----------
        rfm_scaled : pd.DataFrame
            Scaled RFM features DataFrame.

        Returns
        -------
        pd.Series
            Cluster labels for each customer.

        Raises
        ------
        ValueError
            If input DataFrame is empty or has incorrect shape.
        """
        logging.info("Clustering customers with KMeans...")
        if rfm_scaled.empty:
            raise ValueError("Input rfm_scaled DataFrame is empty")
        if rfm_scaled.shape[1] != 3:
            raise ValueError("rfm_scaled must have exactly 3 columns: ['recency', 'frequency', 'monetary']")

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(rfm_scaled)
        self.kmeans = kmeans
        logging.info("Customers clustered into %d clusters.", self.n_clusters)
        return pd.Series(clusters, name='cluster')

    # -----------------------------
    # 4️⃣ Assign high-risk label
    # -----------------------------
    def assign_high_risk(self, rfm: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
        """
        Assign high-risk label based on cluster RFM statistics.

        Parameters
        ----------
        rfm : pd.DataFrame
            Original RFM DataFrame.
        clusters : pd.Series
            Cluster labels corresponding to each row in RFM.

        Returns
        -------
        pd.DataFrame
            DataFrame with [customer_col, is_high_risk] columns.
        """
        logging.info("Assigning high-risk labels...")
        rfm = rfm.copy()
        rfm['cluster'] = clusters

        cluster_stats = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()

        cluster_stats['recency_rank'] = cluster_stats['recency'].rank(ascending=True)
        cluster_stats['frequency_rank'] = cluster_stats['frequency'].rank(ascending=False)
        cluster_stats['monetary_rank'] = cluster_stats['monetary'].rank(ascending=False)

        cluster_stats['score'] = (
            cluster_stats['recency_rank'] +
            cluster_stats['frequency_rank'] +
            cluster_stats['monetary_rank']
        )

        high_risk_cluster = cluster_stats['score'].idxmax()
        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
        logging.info("High-risk labels assigned.")
        return rfm[[self.customer_col, 'is_high_risk']]

    # -----------------------------
    # 5️⃣ Generate final high-risk DataFrame
    # -----------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline to generate high-risk proxy variable.

        Parameters
        ----------
        df : pd.DataFrame
            Input transaction DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with [customer_col, is_high_risk].
        """
        logging.info("Generating high-risk proxy variable...")
        rfm = self.calculate_rfm(df)
        rfm_scaled = self.scale_rfm(rfm)
        clusters = self.cluster_customers(rfm_scaled)
        high_risk_df = self.assign_high_risk(rfm, clusters)
        logging.info("High-risk proxy variable generation complete.")
        return high_risk_df
