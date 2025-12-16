# Notebooks

This folder contains Jupyter notebooks for **exploration and analysis** of the project data. Notebooks use the reusable code in `src/` for consistent workflows.

## Contents

- **`eda.ipynb`**:
Performs a full exploratory data analysis (EDA) on the raw dataset using the EDAHelper class. Tasks include:

   - Loading raw data.
   - Generating summaries for numeric and categorical columns.
   - Analyzing missing values, correlations, and outliers.
   - Producing visualizations to understand distributions and relationships.

- **`feature_engineering.ipynb`**
Implements feature engineering to generate customer-level features.Tasks include:

   - Extracting time features from transaction timestamps.
   - Aggregating numeric features per customer.
   - Aggregating top-K categorical features per customer.
   - Running a feature engineering pipeline combining all transformations.
   - Visualizing numeric, time, and categorical features.
   - Applying Weight of Evidence (WoE) encoding for categorical variables.
   - Saving the resulting features for downstream modeling.

- **`target_engineering.ipynb`**
Generates a high-risk customer target using RFM (Recency, Frequency, Monetary) analysis and clustering. Tasks include:

   - Calculating RFM metrics per customer.
   - Scaling RFM features and clustering customers using KMeans.
   - Assigning high-risk labels based on cluster profiles.
   - Visualizing distributions of RFM metrics, clusters, and high-risk customers.
   - Merging RFM metrics and high-risk labels into the customer feature set.
   - Saving the final dataset for modeling.

- **`model_training.ipynb`**
   - Handles model training, evaluation, and prediction. Tasks include:
   - Loading processed features with high-risk targets.
   - Training multiple models (e.g., RandomForest, Logistic Regression).
   - Evaluating model performance using accuracy, precision, recall, F1, and ROC-AUC.
   - Visualizing model comparison metrics, confusion matrices, and ROC curves.
   - Saving trained models and generating predictions for unseen data.

## Usage

1. Ensure your virtual environment is activated and dependencies are installed.

2. Make sure raw and processed data are available in the directories defined by Config.DATA_DIR.

3. Open any notebook in Jupyter or VSCode and execute cells sequentially to reproduce analyses and features.

4. Outputs (features, targets, models, predictions) are saved in Config.DATA_DIR / "processed" and Config.models_dir.