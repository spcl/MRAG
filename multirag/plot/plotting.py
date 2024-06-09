# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Ales Kubicek
# Lucas Weitzendorf
# Roman Niggli

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter


class PlotExporter:
    """
    Class to store the plots in a file.
    """
    def __init__(self, export_dir: str, file_format: str = "pdf") -> None:
        """
        Initialize PlotExporter instance with output directory and file format.

        :param export_dir: Output directory.
        :type export_dir: str
        :param file_format: File format. Defaults to pdf.
        :type file_format: str
        """
        self.export_dir = export_dir
        self.file_format = file_format

    def save_figure(self, file_name: str) -> None:
        """
        Store plot in a file.

        :param file_name: File name for storing the plot.
        :type file_name: str
        """
        if not file_name.endswith(f".{self.file_format}"):
            file_name += f".{self.file_format}"

        file_path = os.path.join(self.export_dir, file_name)
        if file_dir := os.path.dirname(file_path):
            os.makedirs(file_dir, exist_ok=True)

        print(f"Saving figure to {file_path}...")
        plt.savefig(file_path, format=self.file_format, bbox_inches="tight")
        plt.close()


def plot_absolute_retrieval_improvement_box(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
) -> None:
    """
    Box Plots - Absolute Retrieval Improvement

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    """
    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": True,
        "legend_location": "upper left"
    }

    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step
    measures = ["success_ratio", "category_success_ratio"]
    backgrounds = ["#eaeaf2ff", "#d0d0d0ad"]

    for measure, background in zip(measures, backgrounds):
        sns.set_style(rc={"axes.facecolor": background})
        num_queries = len(df["standard-rag"][query_type][measure][0])

        data_mrag = list(
            np.array(df["multirag"][query_type][measure][docs_from:docs_to:docs_step]).flatten())
        data_standard_rag = list(
            np.array(df["standard-rag"][query_type][measure][docs_from:docs_to:docs_step]).flatten())
        data_split_rag = list(
            np.array(df["split-rag"][query_type][measure][docs_from:docs_to:docs_step]).flatten())

        flat_x = [x for xs in [[x] * num_queries for x in doc_range] for x in xs]

        box_df = pd.DataFrame({
            "rsr": data_mrag + data_standard_rag,
            "articles_fetched": flat_x + flat_x,
            "rag_type": (["MRAG"] * len(data_mrag)) + (["Standard RAG"] * len(data_split_rag))
        })

        plt.figure(figsize=(2.85, 2.4))
        ax = sns.boxplot(
            data=box_df,
            x="articles_fetched",
            y="rsr",
            hue="rag_type",
            width=0.7,
            gap=0.35,
            zorder=101,
            notch=True,
            showmeans=True,
            flierprops={"marker": "o", "markersize": 3},
            medianprops={"color": "navy", "linewidth": 1},
            meanprops={"marker": "p", "markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
            legend=visible["legend"],
        )

        sns.pointplot(
            data=box_df,
            x="articles_fetched",
            y="rsr",
            hue="rag_type",
            markersize=0,
            dodge=True,
            linestyle=(0, (1, 1)),
            linewidth=1,
            alpha=0.8,
            err_kws={"linewidth": 0},
            legend=False,
            zorder=100,
            ax=ax
        )

        plt.yticks(np.arange(11, dtype=float) / 10, visible=visible["y_ticks"])
        plt.xticks(visible=visible["x_ticks"])

        plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
        plt.ylabel("Retrieval Success Ratio" if visible["y_label"] else None)

        if visible["title"]:
            plt.title(f"Query Type {query_type}")

        if visible["legend"]:
            plt.legend(loc=visible["legend_location"])

        exporter.save_figure(f"absolute_{query_type}_{measure}")


def plot_absolute_retrieval_improvement_hist(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int = 10,
        doc_fetched: int = 20
) -> None:
    """
    Histogram - Absolute Retrieval Improvement

    A more detailed look into the distribution of a specific combination of query type and
    number of fetched documents.

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot. Defaults to 10.
    :type query_type: int
    :param doc_fetched: Number of documents. Defaults to 20.
    :type doc_fetched: int
    """
    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": True,
        "legend_location": "upper left"
    }

    def get_dist(data: list[float]):
        counter_data = Counter(data)
        _map = {}
        test = {}
        for i in range(query_type + 1):
            _map[round(i / query_type, 4)] = f"{i}/{query_type}"
            test[f"{i}/{query_type}"] = 0
        for key, value in counter_data.items():
            test[_map[round(key, 4)]] = value
        return test

    measures = ["success_ratio", "category_success_ratio"]
    backgrounds = ["#eaeaf2ff", "#d0d0d0ad"]

    for measure, background in zip(measures, backgrounds):
        sns.set_style(rc={"axes.facecolor": background})

        data_mrag = df["multirag"][query_type][measure][doc_fetched - 1]
        data_standard = df["standard-rag"][query_type][measure][doc_fetched - 1]

        bar_mrag = get_dist(data_mrag)
        bar_standard = get_dist(data_standard)

        bar_df = pd.DataFrame({
            "x_dist": list(bar_mrag.keys()) + list(bar_standard.keys()),
            "counts": list(bar_mrag.values()) + list(bar_standard.values()),
            "x": (["MRAG"] * (query_type + 1)) + (["Standard RAG"] * (query_type + 1))
        })

        plt.figure(figsize=(2.2, 2.4))
        ax = sns.barplot(bar_df, x="x_dist", y="counts", hue="x", legend=visible["legend"], dodge=True)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("none")

        plt.xticks(rotation=90, visible=visible["x_ticks"])
        plt.yticks(range(13), visible=visible["y_ticks"])

        plt.xlabel("Retrieval Success Ratio" if visible["x_label"] else None)
        plt.ylabel("Number of Queries" if visible["y_label"] else None)

        if visible["title"]:
            plt.title(f"Histogram - Query Type {query_type} | Document Fetched {doc_fetched}")

        if visible["legend"]:
            plt.legend(loc=visible["legend_location"])

        exporter.save_figure(f"histogram_{query_type}_{doc_fetched}_{measure}")


def plot_relative_retrieval_improvement_box(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0,
) -> None:
    """
    Box Plots - Relative Retrieval Improvement

    Box plots showing improvement of MRAG over Standard RAG using weighted
    retrieval success ratio (2:1 / document:category as default).

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """
    visible = {
        "title": False,
        "x_ticks": True,
        "x_label": False,
        "y_ticks": False,
        "y_label": False,
        "legend": False,
        "legend_location": "upper left",
        "avg_line": True
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step
    num_queries = len(df["standard-rag"][query_type]["success_ratio"][0])

    strategy_names = ["multirag", "split-rag", "split-rag-strategy-weighted",
                      "fusion-rag", "fusion-multirag", "standard-rag"]

    weighted_ratio = {}
    for name in strategy_names:
        rat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((rat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(value.flatten() - weighted_ratio["standard-rag"].flatten())

    flat_x = [x for xs in [[x] * num_queries for x in doc_range] for x in xs]

    box_df = pd.DataFrame({
        "rsr": results["multirag"],
        "articles_fetched": flat_x,
        "rag_type": (["MRAG"] * len(results["multirag"]))
    })

    plt.figure(figsize=(1.35, 2.1))
    ax = sns.boxplot(
        data=box_df,
        x="articles_fetched",
        y="rsr",
        hue="rag_type",
        width=0.7,
        gap=0.35,
        zorder=102,
        notch=True,
        showmeans=True,
        flierprops={"marker": "o", "markersize": 3},
        medianprops={"color": "navy", "linewidth": 1},
        meanprops={"marker": "p", "markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
        legend=visible["legend"],
    )

    if visible["avg_line"]:
        avg_mrag = np.mean(weighted_ratio["multirag"] - weighted_ratio["standard-rag"], axis=1)
        sns.pointplot(
            x=doc_range, y=avg_mrag,
            linewidth=2.0,
            err_kws={"linewidth": 0},
            legend=False,
            linestyle=(0, (1, 1)),
            zorder=101,
            markers=None,
            ax=ax,
        )

    plt.yticks(np.arange(-3, 6, dtype=float) / 10, visible=visible["y_ticks"])
    plt.xticks(visible=visible["x_ticks"])

    plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
    plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")

    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])

    exporter.save_figure(f"relative_{query_type}_weighted")


def plot_relative_baselines_low_cost(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0
) -> None:
    """
    Box Plots - Relative Retrieval Improvement - Baselines

    Relative retrieval improvement of MRAG over Standard RAG when compared with Split RAG.
    The weighted retrieval success ratio (2:1 / document:category as default) is used.

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """
    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": True,
        "legend_location": "upper left",
        "avg_line": False
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step
    num_queries = len(df["standard-rag"][query_type]["success_ratio"][0])

    strategy_names = ["multirag", "split-rag", "split-rag-strategy-weighted",
                      "fusion-rag", "fusion-multirag", "standard-rag"]

    weighted_ratio = {}
    for name in strategy_names:
        rat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((rat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(value.flatten() - weighted_ratio["standard-rag"].flatten())

    flat_x = [x for xs in [[x] * num_queries for x in doc_range] for x in xs]

    box_df = pd.DataFrame({
        "rsr": results["multirag"] + results["split-rag"],
        "articles_fetched": flat_x + flat_x,
        "rag_type": (["MRAG"] * len(results["multirag"])) + (["Split RAG"] * len(results["split-rag"]))
    })

    plt.figure(figsize=(1.4, 2.1))
    ax = sns.boxplot(
        data=box_df,
        x="articles_fetched",
        y="rsr",
        hue="rag_type",
        width=0.7,
        gap=0.35,
        zorder=101,
        notch=True,
        showmeans=True,
        flierprops={"marker": "o", "markersize": 3},
        medianprops={"color": "navy", "linewidth": 1},
        meanprops={"marker": "p", "markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
        legend=visible["legend"],
        palette=["#3174a2", "#c44e52"],
    )

    if visible["avg_line"]:
        avg_mrag = np.mean(weighted_ratio["multirag"] - weighted_ratio["standard-rag"], axis=1)
        sns.pointplot(
            x=doc_range, y=avg_mrag,
            linewidth=1.2,
            err_kws={"linewidth": 0},
            legend=False,
            linestyle=(0, (1, 1)),
            zorder=102,
            ax=ax
        )

    plt.yticks(np.arange(-5, 9, dtype=float) / 10, visible=visible["y_ticks"])
    plt.xticks(visible=visible["x_ticks"])

    plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
    plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")

    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])

    exporter.save_figure(f"relative_baselines_low_cost_{query_type}_weighted")


def plot_relative_baselines_high_cost(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0
) -> None:
    """
    Box Plots - Relative Retrieval Improvement - Baselines

    Relative retrieval improvement of Fusion MRAG over Standard RAG when compared with Fusion RAG.
    The weighted retrieval success ratio (2:1 / document:category as default) is used.

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """

    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": True,
        "legend_location": "upper left",
        "avg_line": False
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step
    num_queries = len(df["standard-rag"][query_type]["success_ratio"][0])

    strategy_names = ["multirag", "split-rag", "split-rag-strategy-weighted",
                      "fusion-rag", "fusion-multirag", "standard-rag"]

    weighted_ratio = {}
    for name in strategy_names:
        rat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((rat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(value.flatten() - weighted_ratio["standard-rag"].flatten())

    flat_x = [x for xs in [[x] * num_queries for x in doc_range] for x in xs]

    box_df = pd.DataFrame({
        "rsr": results["fusion-multirag"] + results["fusion-rag"],
        "articles_fetched": flat_x + flat_x,
        "rag_type": (["Fusion MRAG"] * len(results["fusion-multirag"])) + (["Fusion RAG"] * len(results["fusion-rag"]))
    })

    plt.figure(figsize=(1.4, 2.1))
    ax = sns.boxplot(
        data=box_df,
        x="articles_fetched",
        y="rsr",
        hue="rag_type",
        width=0.7,
        gap=0.35,
        zorder=101,
        notch=True,
        showmeans=True,
        flierprops={"marker": "o", "markersize": 3},
        medianprops={"color": "navy", "linewidth": 1},
        meanprops={"marker": "p", "markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
        legend=visible["legend"],
        palette=["#2c5a7b", "#8c8c8c"],
    )

    if visible["avg_line"]:
        avg_mrag = np.mean(weighted_ratio["fusion-multirag"] - weighted_ratio["standard-rag"], axis=1)
        sns.pointplot(
            x=doc_range, y=avg_mrag,
            linewidth=1.2,
            err_kws={"linewidth": 0},
            legend=False,
            linestyle=(0, (1, 1)),
            zorder=102,
            ax=ax
        )

    plt.yticks(np.arange(-5, 9, dtype=float) / 10, visible=visible["y_ticks"])
    plt.xticks(visible=visible["x_ticks"])

    plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
    plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")

    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])

    exporter.save_figure(f"relative_baselines_high_cost_{query_type}_weighted")


def plot_relative_retrieval_improvement_line(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0
) -> None:
    """
    Line Plots - Relative Retrieval Improvement

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """
    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": False,
        "legend_location": "upper left",
        "avg_line": False
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step

    weighted_ratio = {}
    for name in ["multirag", "split-rag", "split-rag-strategy-weighted", "standard-rag"]:
        cat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((cat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

    x_vals = doc_range

    plt.figure(figsize=(0.8, 1.6))
    sns.lineplot(x=x_vals, y=results["multirag"], label="MRAG", marker="o", legend=visible["legend"])
    sns.lineplot(x=x_vals, y=results["split-rag"], label="Split RAG",
                 marker="p", legend=visible["legend"], color="#c44e52")

    plt.yticks(np.arange(-2, 13, 2, dtype=float) / 100, visible=visible["y_ticks"])
    plt.xticks(x_vals[::5], visible=visible["x_ticks"])

    plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
    plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")

    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])

    exporter.save_figure(f"line_{query_type}")


def plot_voting_strategies(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0
) -> None:
    """
    Line Plots - Voting Strategies

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """
    visible = {
        "title": True,
        "x_ticks": True,
        "x_label": True,
        "y_ticks": True,
        "y_label": True,
        "legend": True,
        "legend_location": "upper left",
        "avg_line": False
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step

    weighted_ratio = {}
    for name in ["multirag", "multirag-strategy-decay", "multirag-strategy-distance", "standard-rag"]:
        cat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((cat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

    x_vals = doc_range

    plt.figure(figsize=(0.8, 1.6))
    sns.lineplot(x=x_vals, y=results["multirag"], label="MRAG", marker="o", legend=visible["legend"])
    sns.lineplot(x=x_vals, y=results["multirag-strategy-decay"], label="MRAG (Decay)",
                 marker="p", legend=visible["legend"], color="#ca7613ff")
    sns.lineplot(x=x_vals, y=results["multirag-strategy-distance"], label="MRAG (Distance)",
                 marker="X", legend=visible["legend"], color="#dd9138ff")

    plt.yticks(np.arange(-4, 5, 1, dtype=float) / 10, visible=visible["y_ticks"])
    plt.xticks(x_vals[::5], visible=visible["x_ticks"])

    plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)
    plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")
    if visible["legend"]:
        plt.legend(loc=visible["legend_location"], ncol=2)

    exporter.save_figure(f"voting_{query_type}")


def plot_split_rag_voting_strategies(
        exporter: PlotExporter,
        df: pd.DataFrame,
        query_type: int,
        doc_range: range,
        weight: float = 2.0
) -> None:
    """
    Line Plots - Relative Retrieval Improvement - Split RAG Voting Strategies

    :param exporter: Exporter to store the plots in a file.
    :type exporter: PlotExporter
    :param df: Data frame to plot.
    :type df: pd.DataFrame
    :param query_type: Query type to plot.
    :type query_type: int
    :param doc_range: Document range to plot.
    :type doc_range: range
    :param weight: Weight ratio for document:category. Defaults to 2.0.
    :type weight: float
    """
    visible = {
        "title": False,
        "x_ticks": True,
        "x_label": False,
        "y_ticks": False,
        "y_label": False,
        "legend": True,
        "legend_location": "upper left",
        "avg_line": False
    }

    sns.set_style(rc={"axes.facecolor": "#eaeaf2ff"})
    docs_from, docs_to, docs_step = doc_range.start, doc_range.stop, doc_range.step

    weighted_ratio = {}
    for name in ["multirag", "split-rag", "split-rag-strategy-weighted", "standard-rag"]:
        rat = np.array(df[name][query_type]["success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        cat_rat = np.array(df[name][query_type]["category_success_ratio"][docs_from:docs_to:docs_step], dtype=float)
        weighted_ratio[name] = ((rat * weight) + cat_rat) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

    x_vals = doc_range

    plt.figure(figsize=(0.8, 1.6))
    sns.lineplot(x=x_vals, y=results["multirag"], label="MRAG", marker="o", legend=visible["legend"])
    sns.lineplot(x=x_vals, y=results["split-rag"], label="Split RAG (1)",
                 marker="p", legend=visible["legend"], color="#c44e52")
    sns.lineplot(x=x_vals, y=results["split-rag-strategy-weighted"], label="Split RAG (2)",
                 marker="X", legend=visible["legend"], color="#937860")

    plt.yticks(np.arange(0, 22, 4, dtype=float) / 100, visible=visible["y_ticks"])
    plt.xticks(x_vals[::5], visible=visible["x_ticks"])

    if visible["x_label"]:
        plt.xlabel("Number of Documents Fetched" if visible["x_label"] else None)

    if visible["y_label"]:
        plt.ylabel("Improvement over\nStandard RAG" if visible["y_label"] else None)

    if visible["title"]:
        plt.title(f"Query Type {query_type}")

    if visible["legend"]:
        plt.legend(loc=visible["legend_location"], ncol=1)

    exporter.save_figure(f"split_rag_{query_type}")


def plot_all(data_path: str, export_dir: str, file_format: str) -> None:
    """
    Plot all kinds of figures.

    :param data_path: Path to the input file.
    :type data_path: str
    :param export_dir: Directory to write the plot files into.
    :type export_dir: str
    :param file_format: File format to store the plots into.
    :type file_format: str
    """
    df = pd.read_json(data_path)
    exporter = PlotExporter(export_dir, file_format)

    plot_absolute_retrieval_improvement_hist(exporter, df)
    plot_absolute_retrieval_improvement_box(exporter, df, 10, range(10, 21, 5))

    for query_type in (5, 10, 15, 20):
        doc_range = range(query_type, query_type + 11, 2)
        plot_voting_strategies(exporter, df, query_type, doc_range)
        plot_split_rag_voting_strategies(exporter, df, query_type, doc_range)
        plot_relative_retrieval_improvement_line(exporter, df, query_type, doc_range)

    for query_type in (5, 10, 15, 20):
        doc_range = range(query_type, query_type + 11, 5)
        plot_relative_baselines_low_cost(exporter, df, query_type, doc_range)
        plot_relative_baselines_high_cost(exporter, df, query_type, doc_range)
        plot_relative_retrieval_improvement_box(exporter, df, query_type, doc_range)
