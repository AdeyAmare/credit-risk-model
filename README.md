# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding
## 1) Why Basel II makes interpretability and documentation essential

Basel II says banks must measure how risky their loans are and keep enough capital to cover losses. Because capital levels (and regulator approval) depend on model results, models must be transparent and well-documented so regulators can audit them and the bank can explain decisions to customers. Clear rules and documentation reduce regulatory risk and make it easier to prove the model is fair and stable. 

## 2) Why we need a proxy for “default” and the business risks of using it

We cannot wait years to see which borrowers truly never repay, so we label training data with a practical proxy — commonly something like 90 days past due  — to stand in for true default. This gives us timely labels but introduces risk: if the proxy is too strict we wrongly reject good customers (lost revenue); if it’s too lenient we approve risky ones (future losses). Proxies can also introduce sample bias and hide long-term recovery patterns, so ongoing monitoring and periodic re-calibration are necessary. 

## 3) Key trade-offs: simple interpretable models vs. complex high-performance models

Explainability & compliance: Simple scorecards (e.g., logistic regression + WoE) map features to points and are easy to explain, audit, and defend to regulators and customers — a strong advantage under Basel II. 

Predictive power: Complex models (e.g., gradient boosting) often give better accuracy, especially with many features or alternative data sources, but they behave like “black boxes” and need extra explainability and governance. 

Operational cost & risk management: Interpretable models are cheaper to validate and monitor. Complex models require more governance, monitoring for model drift, and explainability tooling — increasing cost and operational risk. 

Practical guideline: Use a transparent scorecard as the default (meets regulatory and customer-facing needs). Consider complex models only when they deliver clear, measurable business value and you have the governance and explainability measures in place.