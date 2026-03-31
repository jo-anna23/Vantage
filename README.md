# Vantage: Touria=sm Arrival Prediction in Jamaica 
## Overview 
This project builds a machine learning forecasting tool to predict monthly visitor arrivals to Jamaica by country of origin and port of entry. It modernizes tourism planning by replacing retrospective reports with predictive analytics.

## Problem
Tourism is Jamaica’s largest economic driver, yet agencies and businesses rely on historical summaries instead of predictive tools. This limits resource planning, marketing, and infrastructure decisions. Our solution provides data-driven forecasts to support smarter strategies.

## Dataset 
- **Source:** Jamaica Open Data Portal (Jamaica Tourist Board).
- **Records:** ~19,000 entries (2014 onwards).
- **Features:** Country of origin, port of entry, year, month, purpose of visit.
- **Target:** Monthly visitor counts.

## Approach
- **Preprocessing:** Cleaning, encoding, scaling, pipeline development.
- **Models:**
    - Linear Regression (baseline).
	- Random Forest Regressor (primary model).
- **Evaluation Metrics:** MAE, RMSE, R² Score.
- **Interpretability:** SHAP plots for feature importance and transparency.
  
## Results
- **Random Forest outperformed Linear Regression::** 
- **Models:**
    - MAE: 74.9 vs 323.8
    - RMSE: 1243.3 vs 2348.4
	- R² Score: 0.92 vs 0.73
- **Key Drivers:** Visitor origin, type of visit (VAC), and major airports (e.g., Sangster International).
- **Deployment:** FastAPI + Streamlit interface for real-world usability.
  
## Impact 
- **Tourism Agencies:** Target campaigns and anticipate demand.
- **Businesses:** Optimize staffing and inventory.
- **Government:** Allocate resources more effectively.
  
## Team 
- Jo-Anna Martinez
- Amoya Jordan
- Dontae Tracey
- Jadian Tulloch
Faculty of Engineering & Computing, University of Technology, Jamaica
Course: **AAI4001 – Supervised Machine Learning**
