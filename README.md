# Olist Customer Analytics & Return Probability Predictor

End-to-end customer analytics system built on 93,000+ real Brazilian e-commerce 
orders. Combines SQL-driven business analysis with an ML-based return probability 
predictor, deployed as a live Streamlit dashboard and a Dockerized Flask REST API.

**Live Demo:** https://olist-customer-analytics-xjawxrsct7ra3dwspu4x5r.streamlit.app/

---

## The Business Problem

How should an e-commerce platform allocate its marketing budget across 93,000 
customers when 97% of them never buy again? Which customers are worth targeting 
with win-back campaigns? Is poor service causing the churn, or is it something 
structural?

---

## What This Project Does

### Layer 1 — SQL Analytics (PostgreSQL)

**Cohort Retention Analysis:** Tracks what % of customers return each month after 
their first purchase. Finding: retention drops below 1% immediately across all 
cohorts — consistent with Olist's product mix being dominated by durable goods 
(furniture, electronics) with naturally long repurchase cycles.

**RFM Segmentation:** Scores each customer on Recency, Frequency, and Monetary 
value. Standard NTILE-based frequency scoring broke down because 97% of customers 
bought exactly once — adapted to binary frequency scoring instead. Key finding: 
At Risk customers (1.1% of base) have the highest average spend at R$295 and 
represent the highest-value win-back opportunity.

**Funnel Analysis:** Tracks order drop-off from placement to delivery using 
timestamp columns (not status labels — cross-validated for consistency). Finding: 
97% end-to-end delivery rate confirms poor service is not the retention problem.

### Layer 2 — Machine Learning (XGBoost)

Predicts which first-time buyers will make a second purchase, targeting the 73,000 
customers in the "Needs Attention" and "Promising" RFM segments.

Key design decisions:
- 90-day observation window applied to avoid right-censoring bias (customers who 
  bought recently haven't had time to return yet)
- Category-adjusted recency feature engineered from scratch: days since purchase 
  divided by that category's average repurchase window, calculated from repeat 
  customer behaviour
- SMOTE applied to training set only to handle 97/3 class imbalance
- Evaluated on PR-AUC (not accuracy) given severe class imbalance
- SHAP analysis revealed review score and category-adjusted recency as strongest 
  predictors

Model performance: XGBoost PR-AUC 0.060 vs baseline 0.032. Modest but meaningful 
— SHAP confirms the model learned real signal. Weak absolute performance reflects 
that return behaviour on durable goods platforms is driven primarily by external 
need, not first-order experience.

### Layer 3 — Deployment

- **Streamlit dashboard:** Live analytics + interactive prediction tool
- **Flask REST API:** POST /predict endpoint returning JSON probability scores
- **Docker:** Containerized Flask API for portable deployment

---

## Tech Stack

SQL: PostgreSQL, pgAdmin4, SQLAlchemy  
ML: XGBoost, Scikit-learn, SMOTE (imbalanced-learn), SHAP  
Visualization: Matplotlib, Seaborn, Streamlit  
Deployment: Streamlit Community Cloud, Flask, Docker  

---

## Key Findings

| Finding | Value |
|---|---|
| Overall retention rate | <1% after first purchase |
| Champion customers | 986 (1.1%) — avg spend R$372 |
| At Risk customers | 997 (1.1%) — avg spend R$295, gone 430 days |
| End-to-end delivery rate | 97% |
| Model PR-AUC (XGBoost) | 0.060 vs 0.032 baseline |
| Strongest predictor | Review score (SHAP) |

---

## How to Run Locally

**Streamlit dashboard:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Flask API:**
```bash
python api.py
# POST to http://localhost:5000/predict
```

**Docker:**
```bash
docker build -t olist-api .
docker run -p 5000:5000 olist-api
```
