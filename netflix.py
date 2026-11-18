import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Netflix Customer Intelligence", page_icon="🎬", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "engineered_customers.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

st.markdown("""
<style>
.stApp {background: #101010; color: #f5f5f5;}
section[data-testid="stSidebar"] {background: #171717;}
div[data-testid="stMetric"] {background: #1e1e1e; border: 1px solid #333; padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data

def load_data(path_string):
    return pd.read_csv(path_string)

@st.cache_resource

def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data

def load_json(path_string):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)

st.sidebar.title("🎬 NETFLIX")
st.sidebar.caption("CUSTOMER INTELLIGENCE")
page = st.sidebar.radio(
    "Navigation",
    ["Executive Dashboard", "Customer Segmentation", "Model Laboratory",
     "Individual Prediction", "Retention Strategy", "Batch Prediction", "Data Audit"]
)

if not DATA_PATH.exists():
    st.error(f"Could not find {DATA_PATH}. Run the Colab notebook and place its extracted `data`, `models`, and `outputs` folders beside app.py.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

required = {"customer_id", "churned", "watch_hours", "avg_watch_time_per_day", "last_login_days"}
missing = required - set(df.columns)
if missing:
    st.error(f"Dataset is missing required columns: {sorted(missing)}")
    st.stop()

# Shared sidebar filters for dashboard and segmentation.
st.sidebar.divider()
st.sidebar.subheader("Dashboard filters")
filter_columns = [c for c in ["age_group", "gender", "region", "device", "monthly_fee_band", "payment_method", "favorite_genre"] if c in df.columns]
filtered = df.copy()
for col in filter_columns:
    vals = sorted(filtered[col].dropna().astype(str).unique()).tolist()
    chosen = st.sidebar.multiselect(col.replace("_", " ").title(), vals, default=vals)
    filtered = filtered[filtered[col].astype(str).isin(chosen)]

st.title("Netflix Customer Intelligence")
st.caption("Churn prediction · K-Means customer segmentation · Retention analytics")

if page == "Executive Dashboard":
    st.subheader("Executive overview")
    if filtered.empty:
        st.warning("No customers match the selected filters.")
        st.stop()
    total = len(filtered)
    churned = int(filtered["churned"].sum())
    retained = total - churned
    churn_rate = filtered["churned"].mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}")
    c3.metric("Retained", f"{retained:,}")
    c4.metric("Churn rate", f"{churn_rate:.1f}%")
    c5, c6, c7 = st.columns(3)
    c5.metric("Average monthly fee", f"{filtered['monthly_fee'].mean():.2f}" if "monthly_fee" in filtered else "N/A")
    c6.metric("Average watch hours", f"{filtered['watch_hours'].mean():.2f}")
    c7.metric("Average inactivity days", f"{filtered['last_login_days'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        status = pd.DataFrame({"Status": ["Churned", "Retained"], "Customers": [churned, retained]})
        st.plotly_chart(px.pie(status, names="Status", values="Customers", hole=.55, title="Churn vs retention"), use_container_width=True)
    with col2:
        selected_segment = st.selectbox("Segment analysis", [c for c in filter_columns if c in filtered.columns], index=0 if filter_columns else None)
        if selected_segment:
            seg = filtered.groupby(selected_segment, dropna=False)["churned"].agg(customers="size", churn_rate="mean").reset_index()
            seg["churn_rate_pct"] = seg["churn_rate"] * 100
            st.plotly_chart(px.bar(seg, x=selected_segment, y="churn_rate_pct", text_auto=".1f",
                                   hover_data=["customers"], title=f"Churn rate by {selected_segment}"), use_container_width=True)

    st.subheader("Engagement behaviour")
    metric = st.selectbox("Choose a viewing or inactivity measure", [c for c in ["watch_hours", "avg_watch_time_per_day", "last_login_days", "engagement_score"] if c in filtered.columns])
    st.plotly_chart(px.box(filtered, x="churned", y=metric, color="churned", points="outliers",
                           title=f"{metric} by churn status"), use_container_width=True)
    st.dataframe(filtered.head(100), use_container_width=True)

elif page == "Customer Segmentation":
    st.subheader("K-Means customer segmentation")
    if "cluster" not in df.columns:
        st.warning("Cluster labels are missing. Run the K-Means section in the Colab notebook first.")
        st.stop()
    seg = filtered.copy()
    summary_cols = [c for c in ["age", "watch_hours", "avg_watch_time_per_day", "last_login_days", "monthly_fee", "number_of_profiles", "engagement_score"] if c in seg.columns]
    summary = seg.groupby("cluster").agg(
        customers=("customer_id", "size"),
        churn_rate=("churned", "mean"),
        **{f"avg_{c}": (c, "mean") for c in summary_cols}
    ).reset_index()
    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.bar(summary, x="cluster", y="customers", text_auto=True, title="Customers per cluster"), use_container_width=True)
    with b:
        st.plotly_chart(px.bar(summary, x="cluster", y="churn_rate_pct", text_auto=".1f", title="Churn rate per cluster"), use_container_width=True)
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    if {"watch_hours", "last_login_days"}.issubset(seg.columns):
        st.plotly_chart(px.scatter(seg, x="watch_hours", y="last_login_days", color="cluster",
                                   hover_data=["churned", "customer_id"], title="Cluster profile: watch hours vs inactivity"), use_container_width=True)
    st.download_button("Download filtered cluster assignments", seg.to_csv(index=False).encode(),
                       "customer_cluster_assignments.csv", "text/csv")

elif page == "Model Laboratory":
    st.subheader("Model performance comparison")
    metrics_path = OUTPUT_DIR / "model_comparison.csv"
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not metrics_path.exists() or not metadata_path.exists():
        st.warning("Model evaluation outputs are missing. Run the supervised modelling section in Colab first.")
        st.stop()
    metrics = pd.read_csv(metrics_path)
    metadata = load_json(str(metadata_path))
    st.metric("Selected model", metadata.get("best_model_name", "Not recorded"))
    display_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "cv_roc_auc_mean"] if c in metrics.columns]
    st.dataframe(metrics[display_cols].round(4), use_container_width=True, hide_index=True)
    if "roc_auc" in metrics:
        st.plotly_chart(px.bar(metrics, x="model", y="roc_auc", text_auto=".3f", title="Held-out ROC-AUC comparison"), use_container_width=True)
    st.info("Metrics are measured on a held-out test set from this synthetic dataset. They should not be presented as real-world Netflix model performance.")

elif page == "Individual Prediction":
    st.subheader("Predict churn probability for one customer")
    model_path = MODEL_DIR / "best_churn_pipeline.joblib"
    meta_path = MODEL_DIR / "model_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        st.warning("The trained model artifacts are missing. Run the Colab modelling section first.")
        st.stop()
    model = load_model(str(model_path))
    meta = load_json(str(meta_path))
    feature_cols = meta["feature_columns"]
    with st.form("single_customer_form"):
        entered = {}
        form_cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with form_cols[i % 3]:
                if col not in df.columns:
                    st.error(f"Feature {col} not found in the prepared dataset.")
                    continue
                 # Subscription fee is a fixed set of observed bands in this dataset,
                # so present it as a dropdown instead of a numeric +/- spinner.
                if col == "monthly_fee":
                    fee_options = sorted(
                        pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()
                    )
                    if not fee_options:
                        st.error("No valid monthly fee options are available in the dataset.")
                        continue
                    entered[col] = st.selectbox(
                        "Monthly Fee Band",
                        fee_options,
                        format_func=lambda value: f"Fee {float(value):g} per month"
                    )
                elif col == "monthly_fee_band":
                    fee_band_options = sorted(df[col].dropna().astype(str).unique().tolist())
                    if not fee_band_options:
                        st.error("No monthly fee bands are available in the dataset.")
                        continue
                    entered[col] = st.selectbox("Monthly Fee Band", fee_band_options)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    med = float(df[col].median())
                    minimum = float(df[col].min())
                    maximum = float(df[col].max())
                    entered[col] = st.number_input(col.replace("_", " ").title(), min_value=minimum, max_value=maximum, value=med)
                else:
                    options = sorted(df[col].dropna().astype(str).unique().tolist())
                    entered[col] = st.selectbox(col.replace("_", " ").title(), options)
        submitted = st.form_submit_button("Estimate churn probability", type="primary")
    if submitted:
        customer = pd.DataFrame([entered], columns=feature_cols)
        probability = float(model.predict_proba(customer)[0, 1])
        low = float(meta.get("risk_thresholds", {}).get("low_to_medium", .35))
        high = float(meta.get("risk_thresholds", {}).get("medium_to_high", .65))
        risk = "Low" if probability < low else ("Medium" if probability < high else "High")
        c1, c2 = st.columns(2)
        c1.metric("Predicted churn probability", f"{probability:.1%}")
        c2.metric("Risk category", risk)
        st.progress(min(max(probability, 0.0), 1.0))
        if risk == "High":
            st.error("Prioritise a retention review and targeted engagement intervention.")
        elif risk == "Medium":
            st.warning("Monitor activity and consider personalised recommendations.")
        else:
            st.success("Lower predicted risk. Continue monitoring the customer experience.")
        st.caption("The probability is a model estimate learned from historical labels; it is not a guaranteed future outcome.")

elif page == "Retention Strategy":
    st.subheader("Customer retention action centre")
    risk_path = OUTPUT_DIR / "retained_customer_risk_scores.csv"
    strategy_path = OUTPUT_DIR / "retention_strategy_matrix.csv"
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        risk_filter = st.selectbox("Risk category", ["All", "High", "Medium", "Low"])
        view = risk_df if risk_filter == "All" else risk_df[risk_df["risk_category"] == risk_filter]
        st.metric("Customers in selected risk group", f"{len(view):,}")
        st.dataframe(view.sort_values("predicted_churn_probability", ascending=False).head(500), use_container_width=True, hide_index=True)
        st.download_button("Download risk list", view.to_csv(index=False).encode(), "retention_risk_list.csv", "text/csv")
    else:
        st.warning("Risk scores are missing. Run the customer risk profiling section in Colab.")
    if strategy_path.exists():
        st.subheader("Segment-level retention opportunities")
        strategy = pd.read_csv(strategy_path)
        st.dataframe(strategy.head(30), use_container_width=True, hide_index=True)
    st.caption("Retention actions are hypotheses to test. The dataset does not measure campaign effectiveness or causal lift.")

elif page == "Batch Prediction":
    st.subheader("Upload customers for batch scoring")
    model_path = MODEL_DIR / "best_churn_pipeline.joblib"
    meta_path = MODEL_DIR / "model_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        st.warning("Model files are missing. Run the Colab modelling section first.")
        st.stop()
    model = load_model(str(model_path))
    meta = load_json(str(meta_path))
    required_features = meta["feature_columns"]
    upload = st.file_uploader("Upload CSV with the model input columns", type=["csv"])
    if upload is not None:
        batch = pd.read_csv(upload)
        missing_features = [c for c in required_features if c not in batch.columns]
        if missing_features:
            st.error(f"Missing required model columns: {missing_features}")
        else:
            scored = batch.copy()
            scored["predicted_churn_probability"] = model.predict_proba(batch[required_features])[:, 1]
            low = float(meta.get("risk_thresholds", {}).get("low_to_medium", .35))
            high = float(meta.get("risk_thresholds", {}).get("medium_to_high", .65))
            scored["risk_category"] = np.select(
                [scored["predicted_churn_probability"] < low, scored["predicted_churn_probability"] < high],
                ["Low", "Medium"], default="High"
            )
            st.metric("Customers scored", f"{len(scored):,}")
            st.dataframe(scored.sort_values("predicted_churn_probability", ascending=False).head(500), use_container_width=True)
            st.download_button("Download scored customers", scored.to_csv(index=False).encode(), "batch_churn_predictions.csv", "text/csv")

elif page == "Data Audit":
    st.subheader("Data quality audit")
    audit_path = OUTPUT_DIR / "data_quality_audit.csv"
    onehot_path = OUTPUT_DIR / "onehot_consistency_audit.csv"
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    if audit_path.exists():
        st.dataframe(pd.read_csv(audit_path), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame({"column": df.columns, "missing": df.isna().sum().values, "dtype": df.dtypes.astype(str).values}), use_container_width=True)
    if onehot_path.exists():
        st.subheader("One-hot encoding consistency checks")
        st.dataframe(pd.read_csv(onehot_path), use_container_width=True)
    st.info("Zero values in avg_watch_time_per_day are retained. That variable is treated independently from watch_hours.")

st.divider()
st.caption("Academic analytics prototype. Predictions and segment associations depend on the synthetic dataset and should not be interpreted as causal or verified Netflix business outcomes.")
