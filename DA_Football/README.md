# Football Wins and Goals Prediction

- main goals: predict wins and number of goals 

- skills learned:
   - handling and aggregating multiple tables with SQL, duckdb and pandas
   - Looker dashboards for data visualization
   - correlation strength and feature importances
   - statistical inference
   - exploring linear and logistic regression in statsmodels
   - exploring recursive feature elimination (iterative approach to save time)

- most important modules used:
   - sqlite3, duckdb (database processing)
   - pandas, numpy (standard processing)
   - plotly, seaborn, matplotlib (visualization)
   - scikit-learn, statsmodels (modelling and performance)


## Feedback from Peer Reviewer and from Senior Team Lead (STL, data science expert currently active in the industry)

- peer: 93/100 points
  - What was good:
    - The project was well structured
    - Very good utilization of SQL, it shows your deeper understanding of it
    - Very Good EDA, you have explored almost everything
    - I liked the map, the plots are very well done
    - Modelling is done correctly
    - Good feature engineering prior to modelling
    - The dashboard was really well done 
    - statistical inference is done well
  - What could be improved:
    - Separate functions from the notebook in to a different python file
    - Remove unnecessary cells, and commented code blocks
    - *I for got to mention this earlier but also try to add explanation to some of the correlation matrices, since some of them were missing it and, the plots are too large to understand what is going on
    - Remove unnecessary outputs that overwhelm the reader of your notebook
    - Look in to other model performance evaluation methods





- STL (after incorporating several of peer's suggestions): 100/100 points - 'Very well done!'
  - What was good:
    * Good quality checks of the data
    * Well formatted data visualisations
    * Interesting players analysis per ranking
    * Good use of SQL with duckdb
    * Good integration of looker into the presentation
    * Very well setup dashboard
    * Nice in-depth analysis of different aspects of the data
    * Good idea to use past performance 
    * Good composition of SQL queries
    * Very good ideas for feature engineering 
    * Good use of recursive feature selection
    * Good variety of approaches is used to find the best model
  - What can be improved:
    * Be less strict with feature selection when it comes to predictive modelling
    * Don’t use player attributes from the future.

