# functions for
# M2 S4 Football data 

import numpy as np
import pandas as pd
import sqlite3
import duckdb
import re

import sys
import os
import warnings
#from importlib import reload


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from xml.etree.ElementTree import fromstring, ElementTree
import xml.etree.ElementTree as ET

import scipy.stats as stats
import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import statsmodels.stats.api as sms
import pingouin as pg


from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.sandbox.predict_functional import predict_functional
#from statsmodels.stats.weightstats import DescrStatsW

#from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedKFold, StratifiedKFold

#from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error
#from sklearn.metrics import confusion_matrix


from sklearn.base import BaseEstimator, RegressorMixin

from datetime import datetime



def corrplot(corr, abs_cut_off = 0.2, fontsize = 10, title = ''):
    '''
    simple correlation plot:
    corr = df.corr()
    abs_cut_off = blend all values below this absolute value
    fontsize = .. of the annotations on top of the color tiles
    title ---> plot will show 'Correlations ' + your title, e.g. 'of Features'
    '''
    fig = plt.figure(figsize=(16,10)) # define plot area
    ax = fig.gca() # define axis

    corr_cut = corr

    corr_cut[np.abs(corr_cut) < abs_cut_off] = 0

    sns.heatmap(corr_cut,
            vmin=-1,
            vmax=1,
            cmap = 'coolwarm',
            annot=True,
            annot_kws={"fontsize":fontsize}
            )
    plt.title('Correlations ' +title+ ' \n')

    fig.show()
    
    
    
    
def nan_type_overview(data):
    '''
    summarize the existence of Nans, how many
    they are and the data types per feature.
    '''
    return pd.concat([data.isnull().any(),
                      data.isnull().sum(),
                      data.dtypes],
                     axis=1
                     ).rename(columns={0: 'Nan?',
                                       1: '#Nan',
                                       2: 'type'})    



def get_histogram_ranges(df, features):
    '''
    helper function to tune the plot ranges of histograms
    slightly (10%) beyond the data limits

    features contains a list of features of the dataframe df
    '''
    ranges_x = []
    for feature in features:
        mini = df[feature].min()
        maxi = df[feature].max()
        margin = (maxi-mini) * 0.1
        ranges_x.append([mini - margin, maxi + margin])
    return ranges_x
    
    
    
    
def quick_histogram(df, features, names, bins=[20,20], ranges_x=[[0, 6], [0, 6]], hue = None):

    '''
    quick histogramm and boxplot function

    samples_df = list of samples dataframes
    features = feature to pick for each sample dataframes in samples_df (--> list)
    sample_names = display names per feature  (--> list)
    bins = binning per feature  (--> list)
    ranges_x = x-values ranges per feature  (--> list)
    hue = plotting/coloring data per category of this feature (--> str)
    '''

    title_text = 'Distribution for '

    #for feature in data.columns:
    if ranges_x == None:
        ranges_x = get_histogram_ranges(df, features)

    for feature, name, bin, range_x in zip(features, names, bins, ranges_x):

        size = ' (n = ' + str(len(df[feature])) + ')'
        fig = px.histogram(df, x=feature, range_x = range_x,
                   nbins = bin,
                   color = hue,
                   text_auto='.1f', #color = 'sign',
                   labels={
                     feature: feature.replace('_',' ')},
                   histnorm='percent',
                   opacity = 0.5,
                   marginal="box",
                   title= title_text + name.replace('_',' ') + size)
        fig.update_layout(
            barmode="overlay",
            bargap=0.05)

        fig.show()
        
        
        
        
        
def roundToNearestZero(number, digit):
    '''
    helper-function to
    round a number to the first digit (number of digits) significant decimals

    '''
    if number < 0:
        sign = -1
    else:
        sign = 1
    n = abs(number)
    if n < 1:
        # Find the first non-zero digit.
        s = f'{n:.99f}'
        index = re.search('[1-9]', s).start()

        shortened = float(s[:index + digit+4])
        return round(shortened, index+digit-2)*sign
    elif n>10:
        return round(n/100., digit)*100.*sign
    else:
        return round(n, digit)*sign
        
        
        
def multicol_histograms(df, features,
                        xlabels = None,
                        n_cols = 2,
                        hue = None,
                        binwidth = None,
                        title = '',
                        yticklabels = None):

    '''
    function to plot seaborn histograms and boxplots in multiple columns
    plots can be colored by a categorical (hue) or by two columns sharing the same axes

    binwidth: unit size of a bin (int)
    title: plot title (str)

    case 1:
    features = [feature1, feature2, feature3, feature4]  ---> plots for each feature
    hue = feature 5 ---> plot in categories of feature 5 and color them
    yticklabels: ignored

    case 2:
    features = [[feature1, feature2], [feature3, feature4]]  ---> plots for groups of features/each list
    hue: ignored
    yticklabels = [[renamed1, renamed2], [renamed3, renamed4]]
                    ---> rename y-axis ticks if feature names are too large
    xlabels: [common_name_1_2, common_name_3_4] ---> rename x-axis labels per feature group

    '''

    n_plots = len(features)

    if n_cols > n_plots:
          n_cols = n_plots
    if n_cols < 1:
          n_cols = 1

    n_rows = int(np.ceil(n_plots/n_cols))

    if binwidth == None:
        binwidth = n_plots*[5]

    ### define image:

    fig_height = int(4*n_rows)

    fig = plt.figure(layout='constrained', figsize=(10, fig_height))
    subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07, hspace=0.1)

    fig.suptitle('Distribution of ' + title +'\n', fontsize='xx-large')

    for index,feature in enumerate(features):

        ### two plots per feature: boxplot and histogram:

        col = index % n_cols
        row = int(index / n_cols)
        if n_cols >= n_plots or n_cols == 1:
            subfig = subfigs[index]
        else:
            subfig = subfigs[row][col]

        plt.subplots_adjust(hspace=0.001)

        [ax_t, ax_b] = subfig.subplots(2, 1, height_ratios=[1,5], sharex=True)

        # multiple features in one subplot now possible with this here:
        if isinstance(feature, list):
            ranges_x_help = get_histogram_ranges(df,feature)
            ranges_x = sum(ranges_x_help, [])
            ranges_x = [min(ranges_x), max(ranges_x)]
            ranges_x = [roundToNearestZero(ranges_x[0],2), roundToNearestZero(ranges_x[1],2)]
            df_final = df[feature]
            x = None
            if xlabels == None:
                xlabel = feature[0].replace('_',' ')
            else:
                xlabel = xlabels[index]
            legend = yticklabels[index]
        else:
            x = feature
            df_final = df
            ranges_x = get_histogram_ranges(df, [feature])
            ranges_x = [roundToNearestZero(ranges_x[0][0],2), roundToNearestZero(ranges_x[0][1],2)]
            xlabel = feature.replace('_',' ')
            legend=''

        # boxplot
        sns.boxplot(data = df_final, x=x, y = hue,  ax=ax_t, orient = 'h')
        if yticklabels is not None:
            ax_t.set_yticklabels(yticklabels[index])
        if hue is not None:
            ax_t.set_ylabel(hue.replace('_',' '))

        ax_t.set_xlabel('')

        # histogram
        sns.histplot(data = df_final, x=x, hue = hue, binwidth = binwidth[index],
                     binrange = ranges_x,
                     stat = 'percent',
                     ax=ax_b)
        size = ' (n = ' + str(len(df[feature])) + ')'

        ax_b.set_xlabel(xlabel + size, fontsize=14)
        
        
        
def group_other(df, feature, rank_lvl=0):
    '''
    group_other - replace all feature values with a count lower than rank_lvl by 'Other'

    df = Pandas dataframe
    feature = feature/column name string
    rank_lvl: show feature values with counts higher than this value
    '''
    df_counts=df[feature].value_counts()
    names = list(df_counts.loc[df_counts>rank_lvl].index)
    names.append(not_answered)
    df.loc[(~(df[feature].isin(names)) & ~(df[feature].isna())), feature] = 'Other'
    
    
                    
def quick_pie(df, feature, rank_lvl=0, show_other=False, title=''):
    '''
    quick_pie - make a simple pie plot as feature overview

    df = Pandas dataframe
    feature = feature/column name string
    rank_lvl: show feature values with counts higher than this value
    show_other: replace all feature values with a count lower than rank_lvl by 'Other'
    '''

    if show_other:
       group_other(df, feature, rank_lvl=rank_lvl)
       df_counts=df[feature].value_counts()
       names = list(df_counts.index)
       values = list(df_counts.values)
    else:
       df_counts=df[feature].value_counts()
       names = list(df_counts.loc[df_counts>rank_lvl].index)
       values = list(df_counts[df_counts>rank_lvl].values)

    fig = px.pie(df_counts, values = values, names = names)

    fig.update_layout(title_text=title + ' (Total: ' + str(df_counts.values.sum()) +')',
                      height = 500, width = 600)

    return fig.show()
    
    
    
# Football data specific

def missing_values(df, feature, no_count_col = 'missing_values'):
    '''
    function to missing values in a feature column - total and per season - and the total number of matches
    '''
    df_no = pd.DataFrame({'country': df.country.unique()})

    for league in df.country.unique():
        df_no.loc[df_no['country'] == league, 'match_count'] = len(df.loc[df.country == league])
        df_no.loc[df_no['country'] == league, no_count_col] = df.loc[df.country == league][feature].isnull().sum()
        for year in df.season.unique():
            df_no.loc[df_no['country'] == league, year] = df.loc[(df.country == league) & (df.season == year)][feature].isnull().sum()
    return df_no            
        
        
        
        
# XML translation code from: https://www.kaggle.com/code/bernardomartinez/tfm-an-lisis-exploratorio#Team        
        
def extract_xml(row,col_name,xml_key,away_home):
    count = 0

    element = row[col_name]
    team_id = row[away_home + "_team_api_id"]

    if type(element) == int:
        return element

    elif element != None:
       # print(row,element)
        tree = ElementTree(fromstring(element))
        root = tree.getroot()

        for child in root.iter(xml_key):

            if str(team_id) == child.text:
                count +=1
        return count
    else:
        return np.nan        
        
        
        
        
def extract_possession_xml(row,col_name,xml_key):
    count = 0
    sum_pos = 0

    element = row[col_name]

    if type(element) == int:
        return element

    elif element != None:
       # print(row,element)
        tree = ElementTree(fromstring(element))
        root = tree.getroot()

        for child in root.iter(xml_key):
            count+=1
            sum_pos += int(child.text)

        if count == 0:
            return np.nan
        else:
            return sum_pos/count
    else:
        return np.nan
        
        
        
        
def q_q_plot(samples, sample_names):
    '''
    quick quantile-quantile-plot function

    samples = list of samples (Series)
    sample_names = display names per sample  (--> list)
    '''
    if len(samples)==2:
        fig = plt.figure(figsize=(12,8))
        fig.suptitle('Q-Q-Plots (scipy) for ', fontsize='16')

        ax1 = plt.subplot(221)
        res = stats.probplot(samples[0], plot=ax1)
        ax1.set_title(sample_names[0], fontsize='11')
        ax2 = plt.subplot(222)
        res = stats.probplot(samples[1], plot=ax2)
        ax2.set_title(sample_names[1], fontsize='11')

    else:
        fig = plt.figure(figsize=(10,8))
        fig.suptitle('Q-Q-Plot (scipy) for ', fontsize='16')

        ax = fig.add_subplot(111)
        res = stats.probplot(samples[0], plot=ax)
        ax.set_title(sample_names[0])

        plt.title(sample_names[0], fontsize='11')
        
        
        
        
        
        
        
        
        
def collect_wins_goals(df, df_source):

    performance = ['matches', 'win', 'draw', 'loss', 'goals',
                   'goal_diff', 'shoton', 'shotoff', 'foulcommit',
                   'card', 'cross', 'corner', 'possession']

    performance_xml = performance[6:]   # create XML based quantities
    performance_wins = performance[1:]  # create 'total_' quantities for all

    # dummy filling to define order of features in data
    for feature in performance:
        for location in ['home', 'away']:
            df[location+ '_' + feature] = np.nan

    for team in df.team_api_id.values:
        for season_id in df.loc[df.team_api_id == team].season.unique():
            for location in ['home', 'away']:
                df_location = df_source.loc[(df_source[location + '_team_api_id'] == team) &
                                            (df_source['season'] == season_id)]

                # df-team and season selector:
                df_select = (df.team_api_id == team) & (df.season == season_id)

                # collect matches
                matches_count = len(df_location)
                df.loc[df_select, location + '_matches'] = matches_count

                # collect wins, draws, losses per total number of matches
                df_location_winner_counts = df_location[location + '_winner'].value_counts()

                for outcome in ['Win', 'Draw', 'Loss']:
                    location_count = [df_location_winner_counts[outcome]
                                      if outcome in df_location_winner_counts else 0][0]
                    df.loc[df_select, location + '_' + outcome.lower()] = (location_count / matches_count)

                # collect goals and differences per total number of matches
                df.loc[df_select, location + '_goals'] = df_location[location + '_team_goal'].mean()
                df.loc[df_select, location + '_goal_diff'] = df_location[location + '_goal_diff'].mean()

                for feature in performance_xml:
                    df.loc[df_select, location + '_' + feature] = df_location[feature + '_' + location].mean()


    # average home and away performance for each perfomance type to a 'total'
    for suffix in performance_wins:
        df['total_' + suffix] = (df['home_' + suffix] + df['away_' + suffix]) / 2.

    # define home advantage and away disadvantage
    advantage_ratio = df['home_goals'] /  df['away_goals']
    df['home_advantage'] = advantage_ratio
    df['away_disadvantage'] = 1 / advantage_ratio

    return df




def quick_scatter(df, xvalues, yvalues, names = [''], xtitle = '', ytitle ='', title = ''):

    fig = go.Figure()

    for x, y, name in zip(xvalues, yvalues, names):

        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name=name))

    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle)

    fig.show()







def collect_previous_seasons(all_seasons = [], season_interest = ''):

    # expects correct sorting

    all_previous_seasons = []
    for season in all_seasons:
        if season == season_interest:
            break
        else:
            all_previous_seasons.append(season)

    return all_previous_seasons
    
    
    
    
    
 
 
def add_seasons_feature_means_to_match(df_match, df_wins,
                                     country_list=[],
                                     number_of_last_seasons=2,
                                     limit_all_seasons = 5):


    df = df_match

    performance = ['win', 'draw', 'loss', 'goals', 'goal_diff', 'shoton', 'shotoff', 'foulcommit',
     'card', 'cross', 'corner', 'possession']

    home_advantage = 'home_advantage'

    for feature in performance:
        for location in ['home', 'away']:
            df[location + '_team_' + feature + '_season'] = np.nan

    df['home_team_goals_paring'] = np.nan
    df['away_team_goals_paring'] = np.nan


    for country in country_list:
        print('Deriving table for', country,)
        df_country = df.loc[(df.country == country)]
        seasons = df_country.season.unique()

        for season_id in seasons[1:]: # skip first season for a team, bc no earlier data present
            df_season = df_country.loc[df_country.season == season_id]
            print('...', season_id, end=' ')


            # define timeranges for averages:

            all_previous_seasons = collect_previous_seasons(all_seasons = seasons, season_interest = season_id)

            if len(all_previous_seasons) > limit_all_seasons:
                all_previous_seasons = all_previous_seasons[ -1 * limit_all_seasons : ]
            if number_of_last_seasons > len(all_previous_seasons): #limit_all_seasons:
                number_of_last_seasons = len(all_previous_seasons) #limit_all_seasons
            last_few_seasons = all_previous_seasons[ -1 * number_of_last_seasons : ]


            for home_team in df_season.home_team_api_id.unique():
                for away_team in df_season.away_team_api_id.unique():

                    match_row = ((df.home_team_api_id == home_team) &
                                (df.away_team_api_id == away_team) &
                                (df.season == season_id) &
                                (df.country == country))

                    # copy the goals and wins, etc
                    for location, team in zip(['home', 'away'], [home_team, away_team]):
                        df_wins_last_seasons = df_wins.loc[
                                                    (df_wins.team_api_id == team) &
                                                    (df_wins.season.isin(last_few_seasons))
                                                    ]
                        for feature in performance:
                            df.loc[match_row,
                                   location + '_team_' + feature + '_season'] = df_wins_last_seasons['total_' + feature].mean()

                        df.loc[match_row,
                                   location + '_home_advantage_goals_season'] = df_wins_last_seasons[home_advantage].mean()


                    # average goals for given team pairing over all last seasons
                    df_pairing = df.loc[((df.home_team_api_id == home_team) &
                                      (df.away_team_api_id == away_team) &
                                      (df.season.isin(all_previous_seasons)) &
                                      (df.country == country))]

                    df_pairing_reverse = df.loc[((df.home_team_api_id == away_team) &
                                              (df.away_team_api_id == home_team) &
                                              (df.season.isin(all_previous_seasons)) &
                                              (df.country == country))]

                    df_pairing_home_goals = df_pairing['home_team_goal'].mean()
                    df_pairing_away_goals = df_pairing['away_team_goal'].mean()

                    df_pairing_reverse_home_goals = df_pairing_reverse['away_team_goal'].mean()
                    df_pairing_reverse_away_goals = df_pairing_reverse['home_team_goal'].mean()

                    write_pairing = ((df.home_team_api_id == home_team) &
                                     (df.away_team_api_id == away_team) &
                                     (df.season == season_id) &
                                     (df.country == country))

                    df.loc[write_pairing,
                            'home_team_goals_paring'] = (df_pairing_home_goals +
                                                         df_pairing_reverse_home_goals
                                                         ) / 2.

                    df.loc[write_pairing,
                            'away_team_goals_paring'] = (df_pairing_away_goals +
                                                         df_pairing_reverse_away_goals
                                                         ) / 2.
        print('.')

    return df.loc[df.country.isin(country_list)] ##_match
    
    
    
    
        
    
    
    
def scattermatrix(df, features, colors = [], titles = []):

    for feature_group, color, title in zip(features, colors, titles):
        dfc = df[feature_group]#.dropna()
        fig = px.scatter_matrix(dfc, dimensions=feature_group, color=color,
            title="Scatter Matrix" + title,
            labels={col:col.replace('total_', '').replace('_', '<br>') for col in df[feature_group]} ) # remove underscore

        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height = 1400, width = 1000)

        fig.show()    
    
    



def mean_player_attr_team_match(dfm, dfp):
    '''
    '''

    list_feature_name = ['age_season',
                         'overall_rating',
                         'ball_control',
                         'gk_handling']

    keeper_quality = 'gk_handling'

    for match_id in dfm.match_api_id.unique():
        df_match = dfm.loc[dfm.match_api_id == match_id]

        for location in ['home', 'away']:

            player_ids = []
            for n in range(1,12):
                player_id = df_match[location + '_player_' + str(n)].values[0]
                if pd.isna(player_id):
                    break
                else:
                    player_ids.append(player_id)


            player_rows = dfp.loc[(dfp.player_api_id.isin(player_ids)) & (dfp.season == df_match.season.values[0])]

            for feature_name in list_feature_name[:-1]:
                dfm.loc[dfm.match_api_id == match_id, location + '_' +feature_name] = player_rows[feature_name].mean()
            dfm.loc[dfm.match_api_id == match_id, location + '_' + keeper_quality] = player_rows[keeper_quality].max()

    return dfm
    
    
    
    
    
        
        
def quick_RFECV(X, y, model_estimator=None, cv=None, min_features_to_select = 1, title = '', scoring=None):
    '''
    recursive feature elimination with cross-validation
    results plotting from sklearn example

    X, y = predictors and response variable
    model_estimator = model type e.g. LinearRegression
    cv = cross-validation method e.g. RepeatedKFold
    min_features_to_select = final selection has at least these many features
    title = suffix for "Recursive Feature Elimination \nwith correlated features " to customize
    scoring = scoring method, e.g. 'accuracy', 'r2'

    '''
    if model_estimator == None:
        model_estimator = LinearRegression()
    if cv == None:
        cv = RepeatedKFold(n_splits=5, n_repeats= 100, random_state=1)

    rfecv = RFECV(
        estimator=model_estimator,
        step=1,
        cv=cv,
        scoring= scoring, #"accuracy",  # error: 'continuous is not supported'
        min_features_to_select=min_features_to_select,
        n_jobs=2,
        )
    fitted = rfecv.fit(X, y)
    #fitted.get_feature_names_out()

    n_scores = len(fitted.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        fitted.cv_results_["mean_test_score"],
        yerr=fitted.cv_results_["std_test_score"],
            )
    plt.title("Recursive Feature Elimination \nwith correlated features " + title)
    plt.show()

    return fitted
    
    
    
    
    
    
def multi_block_RFECV(df, response, feature_list, titles = '',
                      model_est=None, cv=None,
                      min_features_to_select = 1,
                      scoring = None):
    '''
    dropna() executed only for the columns used - not the entire dataset ---> saves more rows

    '''
    selected_features = []
    mean_test_score = []
    std_test_score = []
    data_size = []


    for model_features, title in zip(feature_list, titles):
        all_x_y = np.concatenate([model_features,[response]])
        df_cut = df[all_x_y].dropna()
        print('Using', len(df_cut), 'rows for model ' + title)
        data_size.append(len(df_cut))
        X = df_cut[model_features]
        y = df_cut[response]
        fitted = quick_RFECV(X, y, model_estimator=model_est, cv=cv,
                         min_features_to_select = min_features_to_select,
                         title = title, scoring = None)

        mean_test_score.append(fitted.cv_results_['mean_test_score'].tolist()[fitted.n_features_ -1])
        std_test_score.append(fitted.cv_results_['std_test_score'].tolist()[fitted.n_features_ -1])
        selected_features.append(fitted.get_feature_names_out().tolist())

    titles_pd = [title.replace('for ', '') for title in titles]

    scores = pd.DataFrame(list(zip(titles_pd,
                           data_size,
                           mean_test_score,
                           std_test_score,
                           selected_features)),
               columns =['subset', 'sample size', 'mean_test_score', 'std_test_score', 'RFECVselected'])

    sum_selected_features = np.concatenate(selected_features)

    return scores, sum_selected_features





# translate statsmodels into scikit learn style

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True, groups=None):
        self.model_class = model_class
        self.groups = groups
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)
        
        
        
        
        
        
        
        
def quick_RFE(X, y, model_est=None, max_features_to_select= 1):
              #, title = ''):
    #, scoring=None):
    '''
    recursive feature elimination with cross-validation
    results plotting from sklearn example

    X, y = predictors and response variable
    model_estimator = model type e.g. LinearRegression
    cv = cross-validation method e.g. RepeatedKFold
    min_features_to_select = final selection has at least these many features
    title = suffix for "Recursive Feature Elimination \nwith correlated features " to customize
    scoring = scoring method, e.g. 'accuracy', 'r2'

    '''

    selected_features = []



    if model_est==None or model_est=='lin_reg':
        model_estimator = LinearRegression()
        pipe_lin = Pipeline([('scale', StandardScaler()),
                             ("regressor", SMWrapper(sm.OLS))
                            ])
        model_est ='lin_reg'
    if model_est=='log_reg':
        model_estimator = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        pipe_log = Pipeline([('scale', StandardScaler()),
                             ("regressor", SMWrapper(sm.MNLogit))
                            ])


    if model_est =='lin_reg':

        Cond_Nr = []
        AIC = []
        F_stat = []
        R2 = []
        R2_adj = []
        p_val_coeff = []
        p_val_signif = []
        n_signif = []
        n_insignif = []

        for n in range(1, max_features_to_select+1):

            rfe = RFE(estimator=model_estimator,
                  n_features_to_select = n,
                  step=1)

            fitted = rfe.fit(X, y)

            n_features_selected = fitted.get_feature_names_out().tolist()
            selected_features.append(n_features_selected)

            X_selected = X[n_features_selected]#.columns
            #Model = sm.OLS(y, sm.add_constant(X[n_features_selected]))
            #result = Model.fit()
            pipe_lin.fit(X_selected, y)
            regressor = pipe_lin._final_estimator
            result = regressor.results_

            Cond_Nr.append(result.condition_number.round(1))
            AIC.append(result.aic.round(1))
            F_stat.append(result.fvalue.round(1))
            R2.append(result.rsquared.round(3))
            R2_adj.append(result.rsquared_adj.round(3))
            p_val_coeff.append(result.pvalues.values.round(3))
            singnif_list = [0 if x >= 0.05 else 1 for x in result.pvalues.values]
            p_val_signif.append(singnif_list)
            n_signif.append(singnif_list.count(1))
            n_insignif.append(singnif_list.count(0))

        overview = pd.DataFrame({
                            'F_stat': F_stat,
                            'R2': R2,
                            'R2_adj': R2_adj,
                            'p_v_signif': p_val_signif,
                            'n_signif': n_signif,
                            'n_insignif': n_insignif,
                            'AIC': AIC,
                            'CN': Cond_Nr,
                            'p_val_coeff': p_val_coeff,
                            'selected_features': selected_features,
                            })


    if model_est =='log_reg':

        AIC = []
        P_R2 = []
        LLF = []
        LLNull = []
        LLR = []
        LLR_p_val = []
        p_val_coeff = []
        p_val_signif = []
        n_signif = []
        n_insignif = []

        for n in range(1, max_features_to_select+1):

            rfe = RFE(estimator=model_estimator,
                  n_features_to_select = n,
                  step=1)

            fitted = rfe.fit(X, y)

            n_features_selected = fitted.get_feature_names_out().tolist()
            selected_features.append(n_features_selected)

            X_selected = X[n_features_selected]
            pipe_log.fit(X_selected, y)
            regressor = pipe_log._final_estimator
            result = regressor.results_

            LLF.append(result.llf.round(1))
            LLNull.append(result.llnull.round(1))
            LLR.append(result.llr.round(1))
            LLR_p_val.append(result.llr_pvalue)
            P_R2.append(result.prsquared.round(3))
            AIC.append(result.aic.round(1))
            pvals = [pval[0].round(3) for pval in regressor.results_.pvalues.values]
            p_val_coeff.append(pvals)
            singnif_list = [0 if x >= 0.05 else 1 for x in pvals]
            p_val_signif.append(singnif_list)
            n_signif.append(singnif_list.count(1))
            n_insignif.append(singnif_list.count(0))

        overview = pd.DataFrame({
                            'LLF': LLF,
                            'LLNull': LLNull,
                            'LLR': LLR,

                            'LLR_p_val': LLR_p_val,
                            'P_R2': P_R2,

                            'p_v_signif': p_val_signif,
                            'n_signif': n_signif,
                            'n_insignif': n_insignif,
                            'AIC': AIC,
                            'p_val_coeff': p_val_coeff,
                            'selected_features': selected_features,
                            })

    return overview.round(3)
    
    
    
    
    
    
    
    
                
            
                
        
        
        
        
        
        
        
        
                        
        
        
        
        
        
            
    
    
    
    
    










