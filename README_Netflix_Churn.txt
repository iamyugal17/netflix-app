NETFLIX CUSTOMER CHURN ANALYTICS — QUICK START

1. Open Netflix_Churn_Analytics_Colab.ipynb in Google Colab.
2. Run all cells from top to bottom.
3. Upload netflix_customer_churn1(1).csv when prompted.
4. Download the generated Netflix_Churn_Analytics_Artifacts.zip.
5. Extract the ZIP into a folder on your computer.
6. Copy app.py and requirements.txt into the extracted folder so the folder structure is:

Netflix_Churn_Analytics/
  app.py
  requirements.txt
  data/engineered_customers.csv
  models/best_churn_pipeline.joblib
  models/model_metadata.json
  models/kmeans_model.joblib
  models/kmeans_scaler.joblib
  outputs/model_comparison.csv
  outputs/permutation_feature_importance.csv
  outputs/retained_customer_risk_scores.csv
  outputs/retention_strategy_matrix.csv
  ...other outputs...

7. Open a terminal in that folder and run:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py

Important notes:
- The Colab notebook uses K-Means without the churned target. Churn rates are calculated after clusters are assigned.
- watch_hours and avg_watch_time_per_day are separate variables and are retained independently.
- The model predicts the historical churn label in this synthetic dataset. No future time horizon is supplied, so risk probabilities are not validated future churn forecasts.
