#Jo-Anna Martinez (2305804)
import streamlit as st
import pandas as pd
import joblib 
import matplotlib.pyplot as plt 

st.set_page_config(page_title="Tourism Forecasting Dashboard", layout = "wide")
st.title("Jamaica Tourism Arrival Forecasting System")
st.markdown("This dashboard allows you to forecast the number of tourists visiting Jamaica based on the selected parameters. Choose values in the sidebar and click 'Generate Forecast' to see the forecasted number of visitors.")

# Load the pipeline 
model = joblib.load("rf_pipeline.pkl")

pre = model.named_steps["preprocessor"]
ohe = pre.named_transformers_["cat"]
cat_cols = pre.transformers_[0][2]
cats = dict(zip(cat_cols, ohe.categories_))

port_options = sorted([str(x) for x in cats["Port of Entry"]])
visit_type_options = sorted([str(x) for x in cats["Type of Visit"]])
origin_options = sorted([str(x) for x in cats["Origin"]])
month_options = sorted([str(x) for x in cats["Month"]])

# Sidebar Inputs 
st.sidebar.header("Input Parameters")


port = st.sidebar.selectbox("Port of Entry", port_options)
visit_type = st.sidebar.selectbox("Type of Visit", visit_type_options)
origin = st.sidebar.selectbox("Origin", origin_options)
month = st.sidebar.selectbox("Month", month_options)

input_data = pd.DataFrame([{
    "Port of Entry": str(port),
    "Type of Visit": str(visit_type),
    "Origin": str(origin),
    "Month": str(month)
}])

#Prediction 
for c in ["Port of Entry", "Type of Visit", "Origin", "Month"]:
    input_data[c] = input_data[c].astype(str)

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Forecast Results")
    if st.sidebar.button("Generate Forecast"):
        try:
            pred = model.predict(input_data)[0]
            st.metric("Predicted Number of Visitor Arrivals", f"{pred:,.0f}")
            st.success("Forecast generated successfully!")
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
with col2:
    st.divider()

# Visualization
st.subheader ("Tourism Trends Visualization")
try: 
    #extract trained RF model 
    rf_model = model.named_steps["model"]
    #Encoded features for visualization
    feature_names = pre.get_feature_names_out()
    importances = rf_model.feature_importances_
    importances_df =(
        pd.DataFrame({"Feature": feature_names,"Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False).head(10).sort_values("Importance")
    )

    fig, ax = plt.subplots()
    ax.barh(importances_df["Feature"], importances_df["Importance"], color="skyblue")
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
except Exception as e:
    st.info(f"Feature importance visualization is unavailable: {e}")
