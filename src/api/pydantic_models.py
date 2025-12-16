from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_count: int
    Amount_std: float
    transactionid_transactionid_76871_ratio: float
    transactionid_transactionid_73770_ratio: float
    transactionid_transactionid_26203_ratio: float
    batchid_batchid_67019_ratio: float
    batchid_batchid_51870_ratio: float
    batchid_batchid_113893_ratio: float
    accountid_accountid_4841_ratio: float
    accountid_accountid_4249_ratio: float
    accountid_accountid_4840_ratio: float
    subscriptionid_subscriptionid_3829_ratio: float
    subscriptionid_subscriptionid_4429_ratio: float
    subscriptionid_subscriptionid_1372_ratio: float
    customerid_customerid_7343_ratio: float
    customerid_customerid_3634_ratio: float
    customerid_customerid_647_ratio: float
    currencycode_ugx_ratio: float
    providerid_providerid_4_ratio: float
    providerid_providerid_6_ratio: float
    providerid_providerid_5_ratio: float
    productid_productid_6_ratio: float
    productid_productid_3_ratio: float
    productid_productid_10_ratio: float
    productcategory_financial_services_ratio: float
    productcategory_airtime_ratio: float
    productcategory_utility_bill_ratio: float
    channelid_channelid_3_ratio: float
    channelid_channelid_2_ratio: float
    channelid_channelid_5_ratio: float
    recency: float
    frequency: float
    monetary: float


class PredictionResponse(BaseModel):
    predicted_risk: int
    predicted_risk_prob: Optional[float] = None