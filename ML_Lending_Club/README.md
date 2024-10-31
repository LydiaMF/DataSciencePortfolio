# Lending Club Default Risk

- main goals: predict loan acceptance, default status, and interest rate; deploy model

- skills learned:
   - handling tables > 5GB, 30 Mio. rows, 160 features.
   - correlation strength and feature importances
   - model selection for classification and linear regression
   - exploring recursive feature elimination
   - exercise model deployment (Docker, Google Cloud Platform)

- most important modules used:
   - dask, fastparquet (memory saving processing and storing)
   - pandas, numpy (standard processing)
   - plotly, seaborn, matplotlib (visualization)
   - pingouin (statistical analysis)
   - scikit-learn, imblearn, lightgbm, xgboost, yellowbrick, pickle (modelling and performance)
   - eli5, shap (explainability)
   - fastapi, pydantic, uvicorn (deployment)