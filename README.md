# DataSciencePortfolio
Collection of Projects accomplished during the Data Science Programm at Turing College.

All projects include an intense data cleaning step, and EDA and statistical inference step.

- packages used:
   - pandas, numpy (standard processing)
   - sqlite3, duckdb (database processing)
   - dask, fastparquet (memory saving processing and storing)
   - plotly, seaborn, matplotlib (visualization)
   - pingouin (statistical analysis)
   - scikit-learn, statsmodels, imblearn, lightgbm, xgboost, yellowbrick, pickle (modelling and performance)
   - eli5, shap (explainability)
   - fastapi, pydantic, uvicorn (deployment)
   - torch, lightning
   - albumentations (transforms)

- skills demonstrated:
   - handling and aggregating multiple tables with SQL, duckdb and pandas
   - handling tables > 5GB, 30 Mio. rows, 160 features with dask and pandas
   - handling multiple tables, a multitude of (cross-table) aggregations, and 
       feature engineering with domain knowledge with dask and pandas --> > 300 features
   - Looker dashboards and python plots for data visualization
   - statistical inference
   - correlation strength and feature importances
   - linear and logistic regression in statsmodels
   - model selection for classification and linear regression
   - recursive feature elimination
   - model deployment (Docker, Google Cloud Platform)
   - image data analysis and classification (computer vision)



## Classical Machine Learning:

- [Home Credit Default Risk](./ML_Home_Credit/README.md): 
  - aggregating and combining many auxiliary tables into main table on customer's features and performance
  - predict loan status (default)
  - deploy model

- [Lending Club](./ML_Lending_Club/README.md): 
  - handling and aggregation tables > 5GB, 30 Mio. rows, 150 features
  - predict loan acceptance, loan status (default), and interest rate
  - deploy models


## SQL, Looker, and Logistic and Linear Regression

- [European Football Leagues Data](./DA_Football/README.md): 
   - aggregate/merge multiple tables with SQL
   - predict goal difference and win/loss

## Deep Learning:

- Computer Vision (WORK IN PROGRESS): [Classification of Mushrooms with Pytorch Lightning](./DL_Mushrooms/README.md)