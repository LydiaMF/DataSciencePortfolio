# functions for
# Home Credit Prediction data

import sys

import numpy as np
import pandas as pd
import re

# import dask as dd
from dask.typing import Graph  # , Key, NoDefault, no_default

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import pearsonr, spearmanr
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
)

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
)  # training and testing data split
from sklearn.model_selection import cross_validate  # ,cross_val_predict,

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier

# from optuna.integration import OptunaSearchCV

from imblearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    # make_scorer,
    # accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from skopt import BayesSearchCV

# from skopt.space import Real, Categorical, Integer


from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor


import time


from typing import Union, Literal, Callable


def nan_type_overview(data: pd.DataFrame) -> pd.DataFrame:
    """
    summarize the existence of Nans, how many
    they are and the data types per feature.
    """

    overview = pd.concat(
        [data.isnull().any(), data.isnull().sum(), data.dtypes], axis=1
    ).rename(columns={0: "Nan?", 1: "#Nan", 2: "type"})

    return overview


def get_histogram_ranges(
    df: pd.DataFrame, features: list[str]
) -> list[list[float]]:
    """
    helper function to tune the plot ranges of histograms
    slightly (10%) beyond the data limits

    features contains a list of features of the dataframe df
    """
    ranges_x = []
    for feature in features:
        mini = df[feature].min()
        maxi = df[feature].max()
        margin = (maxi - mini) * 0.1
        ranges_x.append([mini - margin, maxi + margin])
    return ranges_x


def facet_annot_mod(keyword: Union[str, int, float], series: pd.Series) -> str:
    # small helper functions to add total counts to legend of a facet plot
    keywordstr = str(keyword)

    # need more attention - weird behaviour:
    # avoid conflict in Pandas understanding if keyword or index
    # for e.g. for keys 0 and 1:
    # index = series.index[series.index==keyword]

    return keywordstr + "    (" + str(series[keyword]) + ")"


def quick_histogram(
    df: pd.DataFrame,
    features: list[str],
    names: list[str],
    bins: list[int] = [20, 20],
    ranges_x: list[list[float]] = [[0, 6], [0, 6]],
    hue: Union[str, None] = None,
    name_replace: Union[str, None] = None,
) -> go.Figure:
    """
    quick histogramm and boxplot function

    samples_df = list of samples dataframes
    features = feature to pick for each sample dataframes in
               samples_df (--> list)
    sample_names = display names per feature  (--> list)
    bins = binning per feature  (--> list)
    ranges_x = x-values ranges per feature  (--> list)
    hue = plotting/coloring data per category of this feature (--> str)
    name_replace = replace '_' in feature names
                   (name_replace = 'underscore') or separate feature
                   names at capital letters (name_replace = 'capsep')
    """

    title_text = "Distribution for "

    # for feature in data.columns:
    if ranges_x is None:
        ranges_x = get_histogram_ranges(df, features)

    if hue is None:
        hue_name = (
            ""  # applying re.sub to hue = None crashes ---> string needed
        )
    else:
        hue_name = hue

    for feature, name, bin, range_x in zip(features, names, bins, ranges_x):
        if name_replace == "underscore":
            labels = {
                feature: feature.replace("_", " "),
                hue: hue_name.replace("_", " "),
            }
            name = name.replace("_", " ")

        elif name_replace == "capsep":
            labels = {
                feature: re.sub(r"\B([A-Z])", r" \1", feature),
                hue: re.sub(r"\B([A-Z])", r" \1", hue_name),
            }
            name = re.sub(r"\B([A-Z])", r" \1", name)

        else:
            labels = {}

        size = " (n = " + str(len(df[feature])) + ")"
        fig = px.histogram(
            df,
            x=feature,
            range_x=range_x,
            nbins=bin,
            color=hue,
            text_auto=".1f",  # color = 'sign',
            labels=labels,
            histnorm="percent",
            opacity=0.5,
            marginal="box",
            title=title_text + name + size,
        )
        fig.update_layout(barmode="overlay", bargap=0.05)

        if hue is not None:
            fig.for_each_trace(
                lambda t: t.update(
                    name=facet_annot_mod(t.name, df[hue].value_counts())
                )
            )

        fig.show()


def roundToNearestZero(number: float, digit: int) -> Union[float, None]:
    """
    helper-function to
    round a number to the first digit (number of digits) significant decimals

    """
    if number < 0:
        sign = -1
    else:
        sign = 1
    n = abs(number)
    if n < 1:
        # Find the first non-zero digit.
        s = f"{n:.99f}"
        index_test = re.search("[1-9]", s)
        if index_test is not None:
            index = index_test.start()
        else:
            return print("No index determined.")

        shortened = float(s[: index + digit + 4])
        result = round(shortened, index + digit - 2) * sign
    elif n > 10:
        result = round(n / 100.0, digit) * 100.0 * sign
    else:
        result = round(n, digit) * sign
    return result


def multicol_histograms(
    df: pd.DataFrame,
    features: Union[list[str], list[list[str]]],
    xlabels: Union[list[str], None] = None,
    n_cols: int = 2,
    hue: Union[str, None] = None,
    binwidth: Union[list[int], None] = None,
    title: str = "",
    yticklabels: Union[list[str], list[list[str]], None] = None,
) -> plt.Figure:
    """
    function to plot seaborn histograms and boxplots in multiple columns
    plots can be colored by a categorical (hue) or by two columns
    sharing the same axes

    binwidth: unit size of a bin (int)
    title: plot title (str)

    case 1:
    features = [feature1, feature2, feature3, feature4]  ---> plots for
               each feature
    hue = feature 5 ---> plot in categories of feature 5 and color them
    yticklabels: ignored

    case 2:
    features = [[feature1, feature2], [feature3, feature4]]
               ---> plots for groups of features/each list
    hue: ignored
    yticklabels = [[renamed1, renamed2], [renamed3, renamed4]]
                    ---> rename y-axis ticks, if feature names are
                    too large
    xlabels: [common_name_1_2, common_name_3_4] ---> rename x-axis
             labels per feature group

    """

    n_plots = len(features)

    if n_cols > n_plots:
        n_cols = n_plots
    if n_cols < 1:
        n_cols = 1

    n_rows = int(np.ceil(n_plots / n_cols))

    if binwidth is None:
        binwidth = n_plots * [5]

    # define image:
    fig_height = int(4 * n_rows)

    fig = plt.figure(layout="constrained", figsize=(10, fig_height))
    subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07, hspace=0.1)

    fig.suptitle("\nDistribution of " + title + "\n", fontsize="xx-large")

    for index, feature in enumerate(features):
        # two plots per feature: boxplot and histogram:

        col = index % n_cols
        row = int(index / n_cols)
        if n_cols >= n_plots or n_cols == 1:
            subfig = subfigs[index]
        else:
            subfig = subfigs[row][col]

        plt.subplots_adjust(hspace=0.001)

        [ax_t, ax_b] = subfig.subplots(2, 1, height_ratios=[1, 5], sharex=True)

        # multiple features in one subplot now possible with this here:
        if isinstance(feature, list):
            ranges_x_help = get_histogram_ranges(df, feature)
            ranges_x_concat = sum(ranges_x_help, [])
            ranges_x_minmax = [min(ranges_x_concat), max(ranges_x_concat)]
            ranges_x = [
                roundToNearestZero(ranges_x_minmax[0], 2),
                roundToNearestZero(ranges_x_minmax[1], 2),
            ]
            df_final = df[feature]
            x = None
            if xlabels is None:
                xlabel = feature[0].replace("_", " ")
            else:
                xlabel = xlabels[index]
                # legend = yticklabels[index]
        else:
            x = feature
            df_final = df
            ranges_x = get_histogram_ranges(df, [feature])
            ranges_x = [
                roundToNearestZero(ranges_x[0][0], 2),
                roundToNearestZero(ranges_x[0][1], 2),
            ]
            xlabel = feature.replace("_", " ")
            # legend = ""

        # boxplot

        sns.boxplot(data=df_final, x=x, y=hue, ax=ax_t, orient="h")
        if yticklabels is not None:
            ax_t.set_yticklabels(yticklabels[index])
        if hue is not None:
            ax_t.set_ylabel(hue.replace("_", " "))

        ax_t.set_xlabel("")

        # histogram
        sns.histplot(
            data=df_final,
            x=x,
            hue=hue,
            binwidth=binwidth[index],
            binrange=ranges_x,
            stat="percent",
            ax=ax_b,
            kde=True,
        )
        size = " (n = " + str(len(df[feature])) + ")"

        ax_b.set_xlabel(xlabel + size, fontsize=14)

    return fig


def group_other(df: pd.DataFrame, feature: str, rank_lvl: int = 0) -> None:
    """
    group_other - replace all feature values with a count lower than
                  rank_lvl by 'Other'

    df = Pandas dataframe
    feature = feature/column name string
    rank_lvl: show feature values with counts higher than this value
    """
    df_counts = df[feature].value_counts()
    names = list(df_counts.loc[df_counts > rank_lvl].index)
    # names.append(not_answered)
    df.loc[
        (~(df[feature].isin(names)) & ~(df[feature].isna())), feature
    ] = "Other"

    return None


def quick_pie(
    df: pd.Series,
    title: str = "",
    color_discrete_map: Union[dict, None] = None,
) -> go.Figure:
    """
    quick_pie - make a simple pie plot as feature overview

    df = Pandas series

    """

    fig = px.pie(
        df,
        values=list(df.values),
        names=list(df.index),
        color=list(df.index),
        # color=df.index,#.replace('_', ' '),
        color_discrete_map=color_discrete_map,
    )

    fig.update_layout(
        title_text=title,  # + " (Total: " + str(df.values.sum()) + ")",
        height=500,
        width=600,
    )

    return fig


def quick_pie_value_count(
    df: pd.DataFrame,
    feature: str,
    rank_lvl: int = 0,
    show_other: bool = False,
    title: str = "",
    color_discrete_map: Union[dict, None] = None,
) -> go.Figure:
    """
    quick_pie - make a simple pie plot as feature overview

    df = Pandas dataframe
    feature = feature/column name string
    rank_lvl: show feature values with counts higher than this value
    show_other: replace all feature values with a count lower than
                rank_lvl by 'Other'
    """

    if show_other:
        group_other(df, feature, rank_lvl=rank_lvl)
        df_counts = df[feature].value_counts()
        names = list(df_counts.index)
        values = list(df_counts.values)
    else:
        df_counts = df[feature].value_counts()
        names = list(df_counts.loc[df_counts > rank_lvl].index)
        values = list(df_counts[df_counts > rank_lvl].values)

    fig = px.pie(
        df_counts,
        values=values,
        names=names,
        color=names,
        color_discrete_map=color_discrete_map,
    )

    title = title.replace("_", " ").title()
    title_single = "Proportions of the Categories of " + title
    # print('df_counts:', df_counts)
    # print('df_counts values:', df_counts.values)
    # print('df_counts values sum:',
    #        np.sum(df_counts.values))

    fig.update_layout(
        # title_text=title_single + "
        #  (Total: " + str(df_counts.values.sum()) + ")",
        # <--- # old pandas (end of 2023)
        title_text=title_single
        + " (Total: "
        + str(np.sum(df_counts.values))
        + ")",  # for pyarrow pandas!
        height=500,
        width=600,
    )

    return fig


def multicol_pie(
    df: pd.DataFrame,
    feat: Union[str, None] = None,
    hue: Union[str, None] = None,
    counts: Union[str, None] = None,
    title: str = "",
) -> go.Figure:
    if (feat is None) or (hue is None):
        print("Please enter a feature name for feat/hue.")
        return

    # counts = df[feat].value_counts()
    labels = df[feat].unique()
    n_plots = len(labels)

    n_rows = int(np.ceil(n_plots / 2))

    spec_entries = [
        [{"type": "domain"}, {"type": "domain"}] for i in range(n_rows)
    ]

    # sum_label = df.loc[(df[feat] == label)]#.counts#.sum()
    # print(sum_label)

    fig = make_subplots(
        n_rows,
        2,
        specs=spec_entries,
        subplot_titles=[
            # str(label).replace("_", " ") + " (" + str(counts[label]) + ")"
            str(label).replace("_", " ")
            + " ("
            + str(df.loc[(df[feat] == label)][counts].sum())
            + ")"
            for label in labels
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.0001,
    )

    for label, i in zip(labels, range(1, n_plots + 1)):
        if i % 2 == 1:
            col_n = 1
        if i % 2 == 0:
            col_n = 2

        grouping = df.loc[df[feat] == label]  # .groupby(hue).size()
        fig.add_trace(
            go.Pie(
                labels=grouping[hue],
                values=grouping[counts],
                scalegroup="one",
                name=label,
                rotation=90,
            ),
            int(np.ceil(i / 2)),
            col_n,
        )

    fig.update_layout(title_text=title, height=n_rows * 300, width=800)
    return fig


def multicol_pie_value_count(
    df: pd.DataFrame,
    rank: Union[str, None] = None,
    hue: Union[str, None] = None,
    title: str = "",
) -> go.Figure:
    if (rank is None) or (hue is None):
        print("Please enter a feature name for rank/hue.")
        return

    counts = df[rank].value_counts()
    labels = counts.index
    n_plots = len(labels)

    n_rows = int(np.ceil(n_plots / 2))

    spec_entries = [
        [{"type": "domain"}, {"type": "domain"}] for i in range(n_rows)
    ]

    fig = make_subplots(
        n_rows,
        2,
        specs=spec_entries,
        subplot_titles=[
            str(label).replace("_", " ") + " (" + str(counts[label]) + ")"
            for label in labels
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.0001,
    )

    for label, i in zip(labels, range(1, n_plots + 1)):
        if i % 2 == 1:
            col_n = 1
        if i % 2 == 0:
            col_n = 2

        grouping = df.loc[df[rank] == label].groupby(hue).size()
        fig.add_trace(
            go.Pie(
                labels=grouping.index,
                values=grouping,
                scalegroup="one",
                name=label,
                rotation=90,
            ),
            int(np.ceil(i / 2)),
            col_n,
        )

    fig.update_layout(title_text=title, height=n_rows * 300, width=800)
    return fig


def corrplot(
    corr: pd.DataFrame,
    abs_cut_off: float = 0.2,
    fontsize: int = 10,
    title: str = "",
    annot: Union[bool, pd.DataFrame] = True,
) -> plt.Figure:
    """
    simple correlation plot:
    corr = df.corr()
    abs_cut_off = blend all values below this absolute value
    fontsize = .. of the annotations on top of the color tiles
    title ---> plot will show 'Correlations ' + your title,
               e.g. 'of Features'
    annot = switch on labels (T/F) or define labels as dataframe
            of same shape as corr
    """
    fig = plt.figure(figsize=(16, 10))  # define plot area
    ax = fig.gca()  # define axis

    corr_cut = corr

    corr_cut[np.abs(corr_cut) < abs_cut_off] = 0

    sns.heatmap(
        corr_cut,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=annot,
        fmt="",
        ax=ax,
        # annot_kws={"fontsize": fontsize},
    )
    plt.title("Correlations " + title + " \n")

    return fig


def corr_plot_pval(
    df: pd.DataFrame,
    features: list[str],
    corr_method: Literal["pearson", "spearman"] = "spearman",
    abs_cut_off: float = 0.1,
    title: str = "of features with |corr| >",
) -> plt.Figure:
    # p-value hack from:
    # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance

    corr = df[features].corr(method=corr_method)
    if corr_method == "spearman":
        corrfunc = spearmanr
    elif corr_method == "pearson":
        corrfunc = pearsonr
    pval = df[features].corr(method=lambda x, y: corrfunc(x, y)[1]) - np.eye(
        *corr.shape
    )
    p = pval.applymap(
        lambda x: "".join(["*" for t in [0.05, 0.01, 0.001] if x <= t]) + "\n"
    )
    labels = p + corr.round(2).astype(str)

    # fig =
    corrplot(
        corr,
        abs_cut_off=abs_cut_off,
        fontsize=10,
        title=title + str(abs_cut_off),
        annot=labels,
    )

    return None  #


def quick_scatter(
    df: pd.DataFrame,
    xvalues: list[str],
    yvalues: list[str],
    names: list[str] = [""],
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
) -> go.Figure:
    fig = go.Figure()

    for x, y, name in zip(xvalues, yvalues, names):
        print(x, y)
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers", name=name))

    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle)

    return fig


def scattermatrix(
    df: pd.DataFrame,
    features: list[str],
    colors: list[Union[str, None]] = [None],
    titles: list[str] = [""],
) -> go.Figure:
    for feature_group, color, title in zip(features, colors, titles):
        dfc = df[feature_group]  # .dropna()
        if color in feature_group:
            feature_group.remove(color)
        fig = px.scatter_matrix(
            dfc,
            dimensions=feature_group,
            color=color,
            opacity=0.5,
            title="Scatter Matrix" + title,
            labels={
                col: col.replace("_", "<br>") for col in df[feature_group]
            },
        )  # remove underscore

        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=1400, width=900)

        return fig


def q_q_plot(
    samples: list[list[float]], sample_names: list[str] = [""]
) -> plt.Figure:
    """
    quick quantile-quantile-plot function

    samples = list of samples (Series)
    sample_names = display names per sample  (--> list)
    """
    if len(samples) == 2:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Q-Q-Plots for ", fontsize=16)

        ax1 = plt.subplot(221)
        stats.probplot(samples[0], plot=ax1)
        ax1.set_title(sample_names[0], fontsize=11)
        ax2 = plt.subplot(222)
        stats.probplot(samples[1], plot=ax2)
        ax2.set_title(sample_names[1], fontsize=11)

    else:
        fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(111)
        stats.probplot(samples[0], plot=ax)

        plt.title("Q-Q-Plots for " + sample_names[0], fontsize=12)

    return None  # fig


# incorporate a special early_stopping_rounds- and evaluation-procedure
# for XGBoost into sklearn BaseEstimator's fit - function from
# https://stackoverflow.com/questions/50824326/xgboost-with-gridsearchcv-scaling-pca-and-early-stopping-in-sklearn-pipeline


class EstWithEarlyStop(BaseEstimator):
    def __init__(
        self,
        early_stopping_rounds: int = 5,
        test_size: float = 0.1,
        eval_metric: str = "auc",
        **estimator_params,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.eval_metric = eval_metric
        if self.estimator is not None:
            self.set_params(**estimator_params)

    def set_params(self, **params) -> BaseEstimator:
        self.estimator = self.estimator.set_params(**params)
        return self

    def get_params(self, **params) -> BaseEstimator:
        return self.estimator.get_params()

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> BaseEstimator:
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size
        )

        # !!! special early_stopping_rounds not working anymore (2024)
        #  --> use XGB 2.0.2
        if "early_stopping_rounds" not in fit_params:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds
        if "eval_metric" not in fit_params:
            fit_params["eval_metric"] = self.eval_metric
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = [(x_val, y_val)]
        self.estimator.fit(
            x_train,
            y_train,
            **fit_params,
        )
        self.classes_ = self.estimator.classes_

        return self

    def predict(self, X: np.ndarray) -> BaseEstimator:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> BaseEstimator:
        return self.estimator.predict_proba(X)


class XGBoostRegressorWithEarlyStop(EstWithEarlyStop, RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBRegressor()
        super(XGBoostRegressorWithEarlyStop, self).__init__(*args, **kwargs)


class XGBoostClassifierWithEarlyStop(EstWithEarlyStop, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBClassifier()
        super(XGBoostClassifierWithEarlyStop, self).__init__(*args, **kwargs)


class LGBMWithEarlyStop(BaseEstimator):
    def __init__(
        self,
        test_size: float = 0.1,
        **estimator_params,
    ):
        self.test_size = test_size
        if self.estimator is not None:
            self.set_params(**estimator_params)
            # self.classes_ = self.estimator.classes_

    def set_params(self, **params) -> BaseEstimator:
        self.estimator = self.estimator.set_params(**params)
        return self

    def get_params(self, **params) -> BaseEstimator:
        return self.estimator.get_params()

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> BaseEstimator:
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size
        )
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = [(x_val, y_val)]
        self.estimator.fit(x_train, y_train, **fit_params)
        self.classes_ = self.estimator.classes_

        return self

    def predict(self, X: np.ndarray) -> BaseEstimator:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> BaseEstimator:
        return self.estimator.predict_proba(X)


class LGBMRegressorWithEarlyStop(LGBMWithEarlyStop, RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.estimator = LGBMRegressor()
        super(LGBMRegressorWithEarlyStop, self).__init__(*args, **kwargs)


class LGBMClassifierWithEarlyStop(LGBMWithEarlyStop, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        self.estimator = LGBMClassifier()
        super(LGBMClassifierWithEarlyStop, self).__init__(*args, **kwargs)


def add_dict_per_list_dict_item(
    list_of_dicts: list[dict], dict_to_add_to_each: dict
) -> list[dict]:
    extended_list_of_dicts = []
    for dict_item in list_of_dicts:
        new = dict_item.copy()
        new.update(dict_to_add_to_each)
        extended_list_of_dicts.append(new)
    return extended_list_of_dicts


def model_scores(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "binary",
    models: Union[
        list[Union[BaseEstimator, RegressorMixin, ClassifierMixin]], None
    ] = None,
    model_names: Union[list[str], None] = None,
    cv: Union[int, Callable] = 5,
    fit_params: Union[list[Union[dict, None]], None] = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Function for evaluating the performance of machine learning models
    using cross-validation, collecting various CV performance metrics,
    and predicting.
    It calculates metrics such as accuracy, precision, recall,
    F1 score (micro and macro), and AUC-ROC (OvR) for each model using
    K-fold cross-validation.

    X (array-like): Predictors (features).
    y (array-like): Response variable
    X_test (array-like): Predictors for prediction
    model_type (str): 'binary', 'multiclass', 'regression'
    models (list): List of estimator objects.
    model_names (list): List of names for the estimators.
    cv (int/cv_object): Number of cross-validation folds.
    fit_params: list of parameter-dictionaries that need
                to be handed to the fit-function

    returns:
    model_performance: dataframe with the various metrics
    collect_cv_results: summary dict for each model
    probabilities: probabilities from each model for X_test
    """

    # Default model and model names if not provided
    if models is None:
        models = [SVC(kernel="rbf")]

    if model_names is None:
        model_names = ["rbf-SVC"]

    if fit_params is None:
        fit_params = [None] * len(model_names)

    # Define scoring metrics for cross-validation

    if model_type == "binary":
        scoring = [
            "accuracy",
            "balanced_accuracy",
            "f1",  # (for binary targets),
            "f1_micro",
            "f1_macro",
            "f1_weighted",  # 'f1_samples',  #(multilabel sample)
            "precision",
            "precision_micro",
            "precision_macro",
            "precision_weighted",  # 'precision_samples',
            "average_precision",
            "recall",
            "recall_micro",
            "recall_macro",
            "recall_weighted",  # 'recall_samples',
            "roc_auc",
            "roc_auc_ovr",
            "roc_auc_ovo",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_weighted",
        ]

    if model_type == "multiclass":
        scoring = [
            "accuracy",
            "balanced_accuracy",
            # 'f1', #(for binary targets),
            "f1_micro",
            "f1_macro",
            "f1_weighted",  # 'f1_samples',  #(multilabel sample)
            # 'precision',
            "precision_micro",
            "precision_macro",
            "precision_weighted",  # 'precision_samples', 'average_precision',
            # 'recall',
            "recall_micro",
            "recall_macro",
            "recall_weighted",  # 'recall_samples',
            # 'roc_auc',
            "roc_auc_ovr",
            "roc_auc_ovo",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_weighted",
        ]

    if model_type == "regression":
        scoring = [
            "r2",
            "explained_variance",
            "neg_median_absolute_error",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
        ]

    # Dict to collect performance metrics
    collect_scores = {}
    for item in scoring:
        collect_scores[item + "_mean"] = []
        collect_scores[item + "_std"] = []

    # List to collect cross-validation results and probalilities
    collect_cv_results = []
    probabilities = []
    reg_predict = []
    fitted_estimators = []

    # Loop through each model and perform cross-validation
    for model, model_name, fit_param in zip(models, model_names, fit_params):
        print(f"Executing {model_name}")
        # Perform cross-validation and collect results
        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            verbose=False,
            fit_params=fit_param,
            return_estimator=True,
        )

        # Collect mean and standard deviation of performance metrics
        for item in scoring:
            collect_scores[item + "_mean"].append(
                np.abs(cv_result["test_" + item].mean())
            )
            collect_scores[item + "_std"].append(
                np.abs(cv_result["test_" + item].std())
            )

        # Collect cross-validation results and probabilities_model
        collect_cv_results.append(cv_result)

        if model_type != "regression":
            # Get probabilities for X_test
            probabilities_model = cv_result["estimator"][0].predict_proba(
                X_test
            )
            probabilities.append(probabilities_model)
        if model_type == "regression":
            # Collect estimators:
            fitted_estimators.append(cv_result["estimator"][0])
            # Get predictions for X_test
            reg_predict_model = cv_result["estimator"][0].predict(X_test)
            reg_predict.append(reg_predict_model)

    print("Done!")

    # Create a dataframe to store model performance metrics
    for item in scoring:
        collect_scores[item.replace("_", " ") + " mean"] = collect_scores.pop(
            item + "_mean"
        )
        collect_scores[item.replace("_", " ") + " std"] = collect_scores.pop(
            item + "_std"
        )

    model_performance = pd.DataFrame(
        collect_scores,
        index=model_names,
    )

    if model_type == "regression":
        return (
            model_performance,
            collect_cv_results,
            reg_predict,
            fitted_estimators,
        )
    else:
        return model_performance, collect_cv_results, probabilities


def plot_confusion_matrices(
    # X_train,
    # y_train,
    probabilities: list[np.ndarray],
    y_test: np.ndarray,
    # estimators=[SVC(kernel="rbf")],
    names: list[str] = ["my_model"],
    labels: Union[list[str], None] = None,
    # fit_params=None,
    normalize: Union[str, None] = None,
    threshold: Union[float, None] = None,
    # cv=5,
    n_cols: int = 2,
    title: str = "",
) -> plt.Figure:
    """
    Plot Seaborn heatmap of confusion matrices for given estimators and
    data.

    Args:
    X (array-like): Predictors (features).
    y (array-like): Response variable.
    probabilities: list of predict_proba(X_test) per estimator
    #estimators (list): List of estimator objects.
    names (list): List of names for the estimators.
    fit_params (list(dict)): list of parameter-dictionaries that need
                to be handed to the fit-function
    cv (int): Number of cross-validation folds.
    n_cols (int): Number of columns in the plot grid.
    title (str): Plot title.
    """

    if threshold is None:
        threshold = 0.5

    n_plots = len(names)
    n_cols = max(1, min(n_cols, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))

    # variable for scaling subplot size
    n_cols_width = 3

    fig_height = 4 * n_rows
    fig_width = min(12, 4 * n_cols)
    if labels is not None:
        length = len(labels)
        if length > 5:
            if n_cols > 2:
                print("Too many classes to display. Reducing to n_cols = 2")
                n_cols = 2
                n_cols_width = n_cols
            if n_cols == 1:
                n_cols_width = 2
        if length > 15:  # and n_col > 2:
            print("Too many classes to display. Reducing to n_cols = 1")
            n_cols = 1
            n_cols_width = n_cols

        width = max(n_cols, n_cols_width)
        fig_height = 4 * n_rows * 3 / width
        fig_width = min(12, 4 * n_cols * 3 / width)

    n_rows = int(np.ceil(n_plots / n_cols))

    # Create a grid of subplots to accommodate confusion matrices

    fig = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
    subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07, hspace=0.1)

    # Set the main title of the entire plot
    fig.suptitle("\nConfusion Matrices " + title + "\n", fontsize="xx-large")

    # Loop through each estimator and its corresponding name
    # for index, (prob, name, fit_param)
    # in enumerate(zip(probabilities, names, fit_params)):
    for index, (prob, name) in enumerate(zip(probabilities, names)):
        row = index // n_cols
        col = index % n_cols

        # subfig = subfigs[row][col]
        if n_plots == 1:
            subfigs = [subfigs]  # Wrap single subplot in a 2D list

        if n_cols >= n_plots or n_cols == 1:
            subfig = subfigs[index]
        else:
            subfig = subfigs[row][col]

        # This unelegant approach prevents empty plot grids,
        # when n_row x n_col > n_plots!
        ax = subfig.subplots(1, 1)

        # Get predictions from probabilities
        y_pred = (prob[:, 1] > threshold).astype("float")

        # Compute the confusion matrix for the true labels and predicted labels
        cm = confusion_matrix(y_test, y_pred, normalize=normalize)

        # cmd = ConfusionMatrixDisplay(cm, display_labels=label)
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
        )

        # Set the subplot title to the name of the current estimator
        ax.set_title(name)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return None  # fig


def hyper_search(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_names: Union[list[str], None] = None,
    param_grid: Union[list[dict], None] = None,
    param_distributions: Union[list[dict], None] = None,
    param_bayes: Union[list[dict], None] = None,
    # param_distro_optuna: Union[list[dict], None] = None,
    cv: Union[int, Callable] = 5,
    niter: int = 10,
    searches: Union[list[str], None] = None,
    scoring: Union[list[str], str] = "f1",  # f1 is f1_binary
    refit: Union[str, bool] = False,
    verbose: bool = False,
    # optuna_n_trials=5,
    fit_params: Union[list[dict], None] = None,
    random_state: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Performs hyperparameter search for multiple classifiers using
    different search algorithms.

    Args:
    pipe: Pipeline for the classifier.
    X_train: Training data.
    y_train: Training labels.
    classifier_names: Names of the classifiers.
    param_grid: Grid of hyperparameters for grid search (must contain
               classifier)
    param_distributions: Distributions of hyperparameters for random
                         search (must contain classifier)
    param_bayes: Distributions of hyperparameters for Bayes
                         search (must contain classifier)
    cv: Number of cross-validation folds.
    niter: Number of iterations for random search.
    searches: List of search algorithms (options: 'Grid', 'Random',
              'HalvingGrid', 'HalvingRandom').
    scoring: str or list of scoring metrics to use
    refit: if scoring is a list, which of them to use for 'best_score'?
           Set str of this scorer here
    verbose: Whether to print verbose output.
    fit_params: list of parameter-dictionaries that need
                to be handed to the fit-function

    return: DataFrame containing hyperparameter search results.
    """

    best_score = []
    best_params = []
    Searchmethod = []
    classifier_used = []
    timing = []

    if searches is None:
        searches = ["Grid"]

    if classifier_names is None:
        classifier_names = ["my_model"]

    if fit_params is None:
        fit_params = [{}] * len(classifier_names)

    allowed_search = [
        "Grid",
        "Random",
        "HalvingGrid",
        "HalvingRandom",
        "Tune",
        "Bayes"
        # "Optuna",
    ]
    check_search = [True if x in allowed_search else False for x in searches]

    if sum(check_search) < len(check_search):
        print("Unknown search algorithm!")
        # break
    else:
        pass

    if ("Grid" in searches) or ("HalvingGrid" in searches):
        if param_grid is not None:
            classifiers = range(len(param_grid))
        else:
            searches.remove("Grid")
            searches.remove("HalvingGrid")
    elif ("Random" in searches) or ("HalvingRandom" in searches):
        if param_distributions is not None:
            classifiers = range(len(param_distributions))
        else:
            searches.remove("Random")
            searches.remove("HalvingRandom")

    elif "Bayes" in searches:
        if param_bayes is not None:
            classifiers = range(len(param_bayes))
        else:
            searches.remove("Bayes")

    for index, classifier_spec in enumerate(classifiers):
        # !!! ALERT !!! loop works only reliably, if all searches use
        # same list of classifiers or if just one search used!
        for search in searches:
            start = time.time()

            if search == "Grid":
                clf = GridSearchCV(
                    pipe,
                    param_grid=param_grid[index],
                    cv=cv,
                    verbose=verbose,
                    n_jobs=-1,
                    scoring=scoring,
                    refit=refit,
                )
            if search == "HalvingGrid":  # does not allow list of scorers
                if isinstance(scoring, list) and isinstance(refit, str):
                    scoring = refit
                clf = HalvingGridSearchCV(
                    pipe,
                    param_grid=param_grid[index],
                    cv=cv,
                    verbose=verbose,
                    n_jobs=-1,
                    factor=2,
                    scoring=scoring,
                )
            if search == "Random":
                clf = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_distributions[index],
                    cv=cv,
                    verbose=verbose,
                    n_jobs=-1,
                    n_iter=niter,
                    scoring=scoring,
                    refit=refit,
                    random_state=random_state,
                )
            if (
                search == "HalvingRandom"
            ):  # refit is Boolean here (i.e. takes best scorer to refit)
                if isinstance(scoring, list) and isinstance(refit, str):
                    scoring = refit
                clf = HalvingRandomSearchCV(
                    pipe,
                    param_distributions=param_distributions[index],
                    cv=cv,
                    verbose=verbose,
                    n_jobs=-1,
                    factor=2,
                    scoring=scoring,
                    random_state=random_state,
                )

            if search == "Bayes":
                # refit is Boolean here (i.e. takes best scorer to refit)
                if isinstance(scoring, list) and isinstance(refit, str):
                    scoring = refit
                clf = BayesSearchCV(
                    pipe,
                    search_spaces=param_bayes[index],
                    cv=cv,
                    verbose=verbose,
                    n_jobs=-1,
                    n_iter=niter,
                    scoring=scoring,
                    refit=refit,
                    random_state=random_state,
                )

            print("Executing", classifier_names[index])
            print("for search method:", search)

            best_clf = clf.fit(X_train, y_train, **fit_params[index])

            dt = time.time() - start

            timing.append(round(dt, 1))
            best_clf_score = best_clf.best_score_.round(3)
            best_score.append(best_clf_score)
            best_params.append(best_clf.best_params_)
            Searchmethod.append(search)
            classifier_used.append(classifier_names[index])

            print("best score:", best_clf_score)
            print("best parameters:", best_clf.best_params_)
            print(" ")

    hyper_params = pd.DataFrame(
        {
            "classifier": classifier_used,
            "search method": Searchmethod,
            "best score": best_score,
            "time[s]": timing,
            "best params": best_params,
        }
    )

    return hyper_params


def quick_preprocess_hyper_tester(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pipe_steps_pre: Union[list[tuple[str]], None] = None,
    classifier: Union[
        BaseEstimator, RegressorMixin, ClassifierMixin, None
    ] = None,
    classifier_name: list[str] = ["Test"],
    param_grid: Union[dict, None] = None,
    param_distro: Union[dict, None] = None,
    cv: Union[int, Callable] = 5,
    niter: int = 20,
    searches: list[str] = ["Random"],
    scoring: str = "balanced_accuracy",
    verbose: float = 0,
) -> pd.DataFrame:
    if classifier is None:
        classifier = RandomForestClassifier()

    pipe_steps = pipe_steps_pre.copy()
    pipe_steps.append(("classifier", classifier))
    pipeline = Pipeline(steps=pipe_steps)

    print(pipeline[-1])

    if param_grid is not None:
        n_sets = len(param_grid)
    elif param_distro is not None:
        n_sets = len(param_distro)
    else:
        print("No parameters given.")
        return

    hyper_params = hyper_search(
        pipeline,
        X_train,
        y_train,
        param_grid=param_grid,
        param_distributions=param_distro,
        classifier_names=classifier_name * n_sets,
        cv=cv,
        niter=niter,
        searches=searches,
        scoring=scoring,
        verbose=verbose,
    )

    return hyper_params


def count_dtypes(df: pd.DataFrame, name: str = "") -> None:
    if name is None:
        name = ""
    else:
        name = name + " "
    dtype_counts = df.dtypes.value_counts()
    print("\nThe dataset " + str(name) + "has:")
    for item in dtype_counts.index:
        print(dtype_counts[item], "features of type " + str(item) + ".")

    return None


def nan_type_overview_dd(
    data: Graph, size: Union[str, None] = None
) -> pd.DataFrame:
    """
    for dask-dataframes!

    returns: Pandas df of summary the number of Nans, their percentage and
    the data types per feature.

    data = dask dataframe
    size = int; - something correponding to len(data)

    """
    nan_overview = pd.concat(
        [data.dtypes, data.isnull().sum().compute()], axis=1
    ).rename(columns={0: "type", 1: "NaN[abs]"})
    if size is None:
        size = data.shape[0].compute()
    nan_overview["NaN[%]"] = (nan_overview["NaN[abs]"] / size * 100.0).astype(
        "float"
    )

    return nan_overview.round(1)


def get_dup(df: pd.DataFrame, printout: bool = False) -> pd.DataFrame:
    duplicates = df[df[df.columns].duplicated() == True]
    if printout:
        print("Data(sub)set has:", len(duplicates), "duplicates.")
    return duplicates


def get_dup_dd(
    df: Graph,
    name: str = "my data",
    size: Union[str, None] = None,
    return_pd: bool = True,
) -> tuple[str, pd.DataFrame]:
    df_dup = df.map_partitions(lambda x: get_dup(x), meta=df)

    if return_pd:
        df_dup = df_dup.compute()
    df_dup_len = len(df_dup)
    if size is None:
        size = len(df)
    print(
        'Total number of duplicates in "',
        name,
        '" :',
        df_dup_len,
        "(",
        round((df_dup_len / size), 1),
        "%).",
    )
    return df_dup_len, df_dup


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """by Fred Cirera,
    https://stackoverflow.com/a/1094933/1870254,
    modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def get_var_size(locals):
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list(locals.items())),
        key=lambda x: -x[1],
    )[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def replace_with_nan(full_str: str, to_replace: str, replace_by: str) -> str:
    if isinstance(full_str, str):
        return full_str.replace(to_replace, replace_by)
    else:
        return full_str


def bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Union[str, None] = None,
    huecolor: Union[str, None] = None,
    huesequence: Union[str, None] = None,
    percent: bool = True,
    title: str = "Plot",
    annotate_total: bool = True,
    category_order: dict = {},
    legend_pos: str = "Right",
) -> go.Figure:
    labels = {x: x.replace("_", " "), y: y.replace("_", " ")}
    if hue is not None:
        labels[hue] = hue.replace("_", " ")

    fig = px.bar(
        df,
        x=x,
        y=y,
        labels=labels,
        color=hue,
        title=title,
        width=800,
        height=600,
        color_discrete_map=huecolor,
        color_discrete_sequence=huesequence,
        category_orders=category_order,
    )

    fig.update_layout(barmode="group")

    if legend_pos == "Left":
        fig.update_layout(
            legend=dict(xanchor="left", x=0.05, yanchor="top", y=0.97)
        )

    if legend_pos == "Left_b":
        fig.update_layout(
            legend=dict(xanchor="left", x=0.05, yanchor="top", y=0.23)
        )

    if legend_pos == "Right":
        fig.update_layout(
            legend=dict(xanchor="left", x=0.8, yanchor="top", y=0.97)
        )

    if legend_pos == "Right_b":
        fig.update_layout(
            legend=dict(xanchor="left", x=0.8, yanchor="top", y=0.23)
        )

    return fig.show()


def get_VIF(dfs: list[pd.DataFrame], selection: list[str]) -> pd.DataFrame:
    """ """

    df_0 = dfs[0]
    df_0.columns = [col.replace(" ", "_") for col in df_0.columns]

    X_0 = df_0[selection]

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_0.columns

    n = 1
    for df in dfs:
        df.columns = [col.replace(" ", "_") for col in df.columns]
        X = df[selection]
        X = X.dropna()
        # calculating VIF for each feature
        colname = "VIF_df" + str(n)
        vif_data[colname] = [
            variance_inflation_factor(X.values, i)
            for i in range(len(X.columns))
        ]
        n += 1

    return vif_data


def drop_high_VIF(
    df: pd.DataFrame,
    num_feat_vif_inp: list[str],
    exclude: list[str] = [],
    cut_offs: list[float] = [10],
) -> tuple[pd.DataFrame, list[str]]:
    to_remove_from_VIF = []
    num_feat_vif = num_feat_vif_inp
    vifs = get_VIF([df], num_feat_vif)

    for cut in cut_offs:
        to_remove_from_VIF = list(
            vifs.loc[vifs["VIF_df1"] > cut]["feature"].values
        )
        for keep in exclude:
            if keep in to_remove_from_VIF:
                to_remove_from_VIF.remove(keep)
        for item in to_remove_from_VIF:
            # if item in to_remove_from_VIF:
            num_feat_vif.remove(item)

        vifs = get_VIF([df], num_feat_vif)

    return vifs, num_feat_vif


def standardize_df(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    dfs_standard = []
    for df in dfs:
        mapper = DataFrameMapper([(df.columns[:], StandardScaler())])
        scaled_features = mapper.fit_transform(df.copy())
        df_standard = pd.DataFrame(
            scaled_features, index=df.index, columns=df.columns[:]
        )
        dfs_standard.append(df_standard)

    return dfs_standard


def get_corr_with_cut(
    df: pd.DataFrame,
    target: str,
    cutoff: float = 0.1,
    greater: bool = True,
    corr_method: Literal["pearson", "spearman"] = "spearman",
) -> pd.DataFrame:
    target_corr = df.corrwith(df[target], method=corr_method)
    if greater:
        cond = target_corr > cutoff
        ascending = False
    else:
        cond = target_corr < cutoff
        ascending = True

    cut_corr = target_corr.loc[cond]
    cut_corr = cut_corr.round(2).sort_values(ascending=ascending)

    return cut_corr


def find_multicorr(
    df: pd.DataFrame,
    target: str,
    list_compare_features: list[str],
    cutoff: float = 0.1,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    inner_corr_strength: float = 0.01,
    positive: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if positive:
        greater = True
    else:
        greater = False

    corr = get_corr_with_cut(
        df, target, cutoff=cutoff, greater=greater, corr_method=corr_method
    )

    print("Features correlated to target", target, " and their strenghts:\n")
    print(corr)

    kick_out = []
    for compare_feat in list_compare_features:
        if positive:
            cond = corr[compare_feat].idxmax()
        else:
            cond = corr[compare_feat].idxmin()
        print(cond)
        compare_corr = df[compare_feat].corrwith(df[cond], method=corr_method)
        compare_corr.drop(labels=cond, inplace=True)
        compare_corr_select = compare_corr.loc[
            compare_corr > inner_corr_strength
        ]

        kick_out.extend(list(compare_corr_select.index))

    return corr, kick_out


def pie_and_multicol_pie(
    df: pd.DataFrame,
    feat: Union[str, None] = None,
    hue: Union[str, None] = None,
    counts: Union[str, None] = None,
    title: str = "Plot",
) -> None:
    df = df.astype("object")
    df_feat = df.groupby(by=[feat])["counts"].sum()

    title = title.replace("_", " ").title()
    title_single = "Proportions of the Categories of " + title
    quick_pie(df_feat, title=title_single).show()

    hue_title = hue.replace("_", " ").title()
    title_multi = (
        "Proportions of " + hue_title + " for the Categories of " + title
    )
    multicol_pie(
        df, feat=feat, hue=hue, counts="counts", title=title_multi
    ).show()

    return None


def remove_outlier(
    df: pd.DataFrame, high: float = 0.95, low: float = 0.05
) -> pd.DataFrame:
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[
                (df[name] > quant_df.loc[low, name])
                & (df[name] < quant_df.loc[high, name])
            ]
    return df


def dependency_test(
    df_in: pd.DataFrame,
    features: list[str],
    target: str,
    type: Literal[
        "f_classif", "f_regress", "chi2", "mutual_info_classif"
    ] = "f_classif",
) -> pd.DataFrame:
    # allowed types: 'f_classif', 'f_regress', 'chi2', 'mutual_info_classif'

    if target in features:
        features.remove(target)

    # filtered_df = df[df[['name', 'country', 'region']].notnull().all(1)]
    df = df_in[
        df_in[features].notnull().all(1)
    ]  # else LabelEncoder crashes because of NaN
    # - had not had the issue before pyarrow pandas
    # - no idea, if there ever where NaNs in the Turing exam data

    X = df[features].copy()
    X = X.apply(LabelEncoder().fit_transform)
    y = df[target].copy().to_frame()

    if (type == "f_classif") or (type == "f_regress"):
        if is_numeric_dtype(y[target]):
            pass
        else:
            y = y.apply(LabelEncoder().fit_transform)
    else:
        y = y.apply(LabelEncoder().fit_transform)

    prefix = type

    if type == "f_classif":
        res = f_classif(X, y)
    elif type == "f_regress":
        res = f_regression(X, y)
    elif type == "chi2":
        res = chi2(X, y)
    elif type == "mutual_info_classif":
        res = mutual_info_classif(X, y)
    else:
        print("Given dependency test type not available.")
        print("Please, choose one of the following as type:")
        print('"f_classif", "f_regress", "chi2", "mutual_info_classif"')
        return None

    if type == "mutual_info_classif":
        feature_res = pd.DataFrame(
            {
                prefix + "_scores": res,
            },
            index=features,
        )
    else:
        feature_res = pd.DataFrame(
            {
                prefix + "_scores": res[0],
                prefix + "_pvals": res[1],
            },
            index=features,
        )

    return feature_res


def plot_model_score(
    dfs: list[pd.DataFrame],
    score: str = "accuracy",
    classifiers_names: Union[list[str], None] = None,
    colors: Union[str, None] = None,
    tilt_x: bool = False,
    title: str = "",
) -> plt.Figure:
    """
    dfs: list of model_scores - results

    cretes boxplot for each df in dfs for a selected score in one plot
    """

    if classifiers_names is None:
        classifiers_names = []
        for i in range(0, len(dfs[0])):
            classifiers_names.append("classifier_" + str(i))

    if colors is None:
        colors = [None] * len(dfs)

    fig, ax = plt.subplots(figsize=(10, 6))

    for df, color in zip(dfs, colors):
        score_list = [df[i]["test_" + score] for i in range(0, len(df))]
        df_scores = pd.DataFrame(score_list, index=classifiers_names)
        fig = sns.boxplot(
            df_scores.T, showmeans=True, color=color, boxprops=dict(alpha=0.3)
        )

    ax.set_ylabel(score.replace("_", " ").title(), size=12)

    if tilt_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_title(title)

    return fig


def plot_y_true_pred(
    y_pred: np.ndarray, y_true: np.ndarray, name: str = ""
) -> plt.Figure:
    plt.figure(figsize=(8, 6))
    plt.scatter(y=y_pred, x=y_true)
    plt.ylabel("Model Predictions ($\^y$)")
    plt.xlabel("y_true")
    plt.title("\nTrue values scatter plot vs predictions from model " + name)

    return plt


def plot_residual_pred(
    residuals: np.ndarray, y_pred: np.ndarray, name: str = ""
) -> plt.Figure:
    plt.figure(figsize=(8, 6))
    print("Red dashed line: mean residual:", round(np.mean(residuals), 2))
    plt.axhline(y=np.mean(residuals), color="r", ls="--", linewidth=3)
    plt.scatter(y_pred, residuals)
    plt.xlabel("Model Predictions ($\^y$)")
    plt.ylabel("Residual Values")
    plt.title("\nResidual scatter plot vs predictions from " + name)

    return plt


def regression_diagnostic_plots(
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> plt.Figure:
    print("\n\n\nDiagnostic Plots for Model: " + name + "\n")

    plot_y_true_pred(y_pred, y_true, name=name)
    plot_residual_pred(residuals, y_pred, name=name)
    q_q_plot([residuals], ["Residuals for model " + name])

    return None


def drop_from_list(list_to_mod: list[str], drop_list: list[str]) -> list[str]:
    """remove items from a list"""
    for item in drop_list:
        if item in list_to_mod:
            list_to_mod.remove(item)
    return list_to_mod


def get_test_scores_binary(
    y_test: np.ndarray,
    probalilities: list[np.ndarray],
    model_names: list[str] = ["model"],
) -> pd.DataFrame:
    """
    Derive the 'balanced accuracy', 'f1', 'precision', 'recall',
    'roc auc' score for test data and the data predicted based on
    their probabilities per model.
    """

    scores = {}
    # collect_scores = {}
    for item in ["balanced accuracy", "f1", "precision", "recall", "roc auc"]:
        scores[item] = []

    for prob in probalilities:
        threshold = 0.5
        y_pred = (prob[:, 1] > threshold).astype("float")

        scores["balanced accuracy"].append(
            balanced_accuracy_score(y_test, y_pred)
        )
        scores["f1"].append(f1_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred))
        scores["recall"].append(recall_score(y_test, y_pred))
        scores["roc auc"].append(roc_auc_score(y_test, prob[:, 1]))

    model_test_performance = pd.DataFrame(scores, index=model_names)

    return model_test_performance
