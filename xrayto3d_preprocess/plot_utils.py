"""matplotlib utils
- for consistent styling
"""
from math import floor, log10
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def set_style():
    """create consistend matplotlib style"""
    matplotlib.style.use("ggplot")
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["lines.linewidth"] = 3
    matplotlib.rcParams["lines.markersize"] = 10


def reset_style():
    """reset matplotlib style"""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def get_barplot(data: Dict, fig=None, ax=None):
    """barplot and return updated figure"""
    if ax is None:
        fig, ax = plt.subplots()
    labels = list(data.keys())
    count = list(data.values())

    # creating the bar plot
    plt.bar(labels, count, width=0.4)
    plt.xticks(fontsize=10)

    return fig, ax


def scatterplot_1d(data: List, label, color="#E24A33", fig=None, ax=None):
    """scatterplot and return updated figure"""
    if ax is None:
        fig, ax = plt.subplots()
    plt.scatter(x=range(len(data)), y=data, label=label, s=6, c=color)
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))
    return fig, ax


def horizontal_line(y_intercept, label, color, fig=None, ax=None):
    """plot a horizontal line and return the updated plot"""
    if ax is None:
        fig, ax = plt.subplots()
    plt.axhline(y=y_intercept, linestyle="--", label=label, color=color)
    return fig, ax


def human_readable_formatter(value):
    """
    make matplotlib tickmarks with large number more readable
    e.g. 100000 -> 1M
    https://stackoverflow.com/questions/61330427/set-y-axis-in-millions
    """
    num_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value)) / 3)
    value = round(value / 1000**num_thousands, 2)
    return f"{value:g}" + " KMGTPEZY"[num_thousands]


def rand_jitter(arr, stdev=0.01):
    """jitter coordinates"""
    stdev = stdev * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev
