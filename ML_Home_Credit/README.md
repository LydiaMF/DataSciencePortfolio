# Home Credit Default Risk

- main goal: predict default risk, deploy model

- skills learned:
   - handling multiple tables, a multitude of (cross-table) aggregations, and feature engineering with domain knowledge --> > 300 features
   - correlation strength and feature importances
   - model selection for classification
   - exploring recursive feature elimination
   - exercise model deployment (Docker, Google Cloud Platform)

- most important modules used:
   - dask (memory saving processing)
   - pandas, numpy (standard processing)
   - plotly, seaborn, matplotlib (visualization)
   - pingouin (statistical analysis)
   - scikit-learn, imblearn, lightgbm, xgboost, yellowbrick, pickle (modelling and performance)
   - eli5, shap (explainability)
   - fastapi, pydantic, uvicorn (deployment)



   ## Feedback from Senior Team Leads (STL, data science experts currently active in the industry)

- STL-1: 91/100 points
  - Well done! You put really much effort into this work. 
  
  - There are several things that could improve it even more:
    - Presentation took too long - you should aim to explain your work in 20-30 minutes. For EDA focus on features that are useful from business perspective and show/try to talk more about anomalies/interesting patterns rather than outlook everything;
    - Try to narrow the problem that you want to solve - when you have this much data it is easy to get lost - one of the possible solutions is to focus on only 1-2 types of loans rather than all of them - they oppose different risks associated with customers so different attributes should be taken in order to estimate risk. Currently it is not clear if model behaves similarly on different types of loans.
    - Using dummy estimator (predicting all cases as a most populated class) is not a robust technique to use - accuracy will be high, but precision and recall will be not informative. You can choose either toss a coin (generate random sequence of ones and zeros with given probability) or you can use AUC as a metric to find if model behaves better than a random estimation;
    - You have successfully trained models but haven’t explained how models make a decision. If you do this you will show your maturity as a specialist;


- STL-2 (after incorporating several of STL-1's suggestions): 95/100 points

  - Thank you for the great project! Hard work and impressive results.
  
  - What was good:
    1. introduction - short, concise, I would understand what is all about even If I wasn't familiar with the dataset.
    2. Overview of the problem, set of datasets, what was your task.
    3. understanding of the entire ML process and proper order of action.
    4. thorough EDA, especially for the application dataset and spotting some interesting patterns already at this stage of your analysis.
    5. reasoning behind your choices and explanations why something is important.
    6. introducing pipeline into your workflows.
    7. understanding what a specific type of error means for the business and trying to optimize it.
    8. adding 'probability score' to the result from the prediction so the decisive person can make an informed decision.
    9. in-depth explainability section.

  - What could be improved:
    1. visualisations, specifically invisible pie plots, and overlapped annotation.
    2. transformation on features should be performed only train set and saved inside the pipelines, so thanks to that you could keep the test set intact just from the start. Transformation were generally performed in a correct way, but on entire dataset, but in a real life scenario you wouldn't have access to the test set at all, so use some nice libraries for feature engineering such as https://feature-engine.trainindata.com/en/latest/index.html to avoid any possible data leakage, that is present when those transformation are done before the split.
    3. for such complex projects, create a report/final summary, that helps with understanding of your final findings.