import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Optional, Union

class ProxyTargetGenerator:
    """
    Generates a proxy credit risk target variable (`is_high_risk`) 
    based on RFM segmentation using KMeans clustering.
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
        self.snapshot_date = snapshot_date
        self.value_col = value_col
        self.n_clusters = n_clusters
        self.random_state = random_state

    # -----------------------------
    # 1️⃣ Calculate RFM metrics
    # -----------------------------
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics per customer,
        ensuring monetary matches the single .agg() approach.
        """
        df_copy = df.copy()
        df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col], errors='coerce')

        # Determine snapshot date
        snapshot = self.snapshot_date
        if snapshot is None:
            snapshot = df_copy[self.date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot = pd.to_datetime(snapshot)

        # Compute RFM metrics using single groupby and agg
        rfm = df_copy.groupby(self.customer_col).agg(
            recency=(self.date_col, lambda x: (snapshot - x.max()).days),
            frequency=(self.date_col, 'count'),
            monetary=(self.value_col, 'sum')  # <-- matches .agg() monetary exactly
        ).reset_index()

        return rfm



    # -----------------------------
    # 2️⃣ Scale RFM features
    # -----------------------------
    def scale_rfm(self, rfm: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
        self.scaler = scaler
        return pd.DataFrame(rfm_scaled, columns=['recency', 'frequency', 'monetary'])

    # -----------------------------
    # 3️⃣ Cluster customers
    # -----------------------------
    def cluster_customers(self, rfm_scaled: pd.DataFrame) -> pd.Series:
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        clusters = kmeans.fit_predict(rfm_scaled)
        self.kmeans = kmeans
        return pd.Series(clusters, name='cluster')

    # -----------------------------
    # 4️⃣ Assign high-risk label
    # -----------------------------
    def assign_high_risk(self, rfm: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
        rfm = rfm.copy()
        rfm['cluster'] = clusters

        # Compute cluster-level mean RFM metrics
        cluster_stats = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()

        # Rank clusters properly:
        # - High recency (long time since last transaction) = worse
        # - Low frequency = worse
        # - Low monetary = worse
        cluster_stats['recency_rank'] = cluster_stats['recency'].rank(ascending=True)
        cluster_stats['frequency_rank'] = cluster_stats['frequency'].rank(ascending=False)
        cluster_stats['monetary_rank'] = cluster_stats['monetary'].rank(ascending=False)

        # Risk score = sum of ranks
        cluster_stats['score'] = (
            cluster_stats['recency_rank'] +
            cluster_stats['frequency_rank'] +
            cluster_stats['monetary_rank']
        )

        # Identify high-risk cluster
        high_risk_cluster = cluster_stats['score'].idxmax()

        # Assign binary high-risk label
        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

        return rfm[[self.customer_col, 'is_high_risk']]

    # -----------------------------
    # 5️⃣ Generate final high-risk DataFrame
    # -----------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.calculate_rfm(df)
        rfm_scaled = self.scale_rfm(rfm)
        clusters = self.cluster_customers(rfm_scaled)
        high_risk_df = self.assign_high_risk(rfm, clusters)
        return high_risk_df
