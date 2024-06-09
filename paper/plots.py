# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek
#
# contributions: Roman Niggli


"""
This is script reproduces the plots presented in the paper.

The data format for the json files with the results is as follows:
```
{
    "standard-rag": {
        "1": {
            "rat": [[...25 queries (float ratio)...], [...25 queries...], ...1-32 doc fetches],
            "category rat": [[...25 queries...], [...25 queries...], ...1-32 doc fetches]
        },
        "2": {...}
    },
    "mrag": {...},
    "split-rag": {...},
    "fusion-rag": {...},
    "fusion-mrag": {...},
    "mrag-strategy-decay": {...},
    "mrag-strategy-distance": {...},
    "split-rag-strategy-weighted": {...},
```
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


### Box Plots - Absolute Retrieval Improvement

model = "sfr"
query_type = 10

docs_from = query_type
docs_to = query_type + 20 # including
docs_step = 2

visible = {
    "title": True,
    "x_ticks": True,
    "x_label": True,
    "y_ticks": True,
    "y_label": True,
    "legend": True,
    "legend_location": "upper left"
}

df = pd.read_json(f"results/25_random_sfr.json")

for measure in ["rat", "category_rat"]:
    background = "#d0d0d0ad" if measure == "category_rat" else "#eaeaf2ff"
    sns.set_style(rc = {"axes.facecolor": background})

    num_queries = len(df["standard-rag"][query_type][measure][0])

    data_mrag = list(np.array(df["mrag"][query_type][measure][docs_from:docs_to+1:docs_step]).flatten())
    data_standard_rag = list(np.array(df["standard-rag"][query_type][measure][docs_from:docs_to+1:docs_step]).flatten())
    data_split_rag = list(np.array(df["split-rag"][query_type][measure][docs_from:docs_to+1:docs_step]).flatten())
    data_fusion_rag = list(np.array(df["fusion-rag"][query_type][measure][docs_from:docs_to+1:docs_step]).flatten())

    flat_x = [x for xs in [[x] * num_queries for x in range(docs_from, docs_to+1, docs_step)] for x in xs]

    box_df = pd.DataFrame({
        "rsr": data_mrag + data_standard_rag ,
        "articles_fetched": flat_x + flat_x,
        "rag_type": (["MRAG"] * len(data_mrag)) + (["Standard RAG"] * len(data_split_rag))
    })

    plt.figure(figsize=(7.5, 2.4))
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
        meanprops={"marker": "p","markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
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
        errwidth=0,
        legend=False,
        zorder=100,
        ax=ax)

    plt.yticks(np.arange(11, dtype=float) / 10, visible=visible["y_ticks"])
    plt.xticks(visible=visible["x_ticks"])

    if visible["x_label"]:
        plt.xlabel("Number of Documents Fetched")
    else:
        plt.xlabel(None)
    if visible["y_label"]:
        plt.ylabel("Retrieval Success Ratio")
    else:
        plt.ylabel(None)
    if visible["title"]:
        plt.title(f"Query Type {query_type}")
    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])
    plt.savefig(f"absolute_{query_type}_{measure}_{model}.pdf", format="pdf", bbox_inches="tight")


### Histogram - Absolute Retrieval Improvement
# A more detailed look into the distribution of concrete columns from the above (absolute retrieval improvement) plot.

from collections import Counter

def get_dist(data):
    counter_data = Counter(data)
    map = {}
    test = {}
    for i in range(query_type + 1):
        map[round(i/query_type, 4)] = f"{i}/{query_type}"
        test[f"{i}/{query_type}"] = 0
    for key, value in counter_data.items():
        test[map[round(key, 4)]] = value
    return test


model = "sfr"
query_type = 10
doc_fetched = 30

visible = {
    "title": True,
    "x_ticks": True,
    "x_label": True,
    "y_ticks": True,
    "y_label": True,
    "legend": True,
    "legend_location": "upper left"
}

for measure in ["rat", "category_rat"]:
    background = "#d0d0d0ad" if measure == "category_rat" else "#eaeaf2ff"
    sns.set_style(rc = {"axes.facecolor": background})

    df = pd.read_json(f"results/25_random_{model}.json")

    data_mrag = df["mrag"][query_type][measure][doc_fetched-1]
    data_standard = df["standard-rag"][query_type][measure][doc_fetched-1]

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
    if visible["x_label"]:
        plt.xlabel("Retrieval Success Ratio")
    else:
        plt.xlabel(None)
    if visible["y_label"]:
        plt.ylabel(f"Number of Queries")
    else:
        plt.ylabel(None)
    if visible["title"]:
        plt.title(f"Histogram - Query Type {query_type} | Document Fetched {doc_fetched}")
    if visible["legend"]:
        plt.legend(loc=visible["legend_location"])
    plt.savefig(f"histogram_{query_type}_{doc_fetched}_{measure}_{model}.pdf", format="pdf", bbox_inches="tight")


### Box Plots - Relative Retrieval Improvement
# Box plots showing the improvement of MRAG over Standard RAG using a weighted retrieval success ratio (2:1 / document:category)

models = ["e5", "sfr"]
weight = 2.0

sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

for model in models:
    df = pd.read_json(f"results/25_random_{model}.json")

    (fig, ax) = plt.subplots(1, 4, figsize=(5.2, 2.1))

    idx = 0
    for query_type in [5, 10, 15, 20]:
        docs_from = query_type
        docs_to = query_type+10 # including
        docs_step = 5

        num_queries = len(df["standard-rag"][query_type]["rat"][0])

        weighted_ratio = {}
        for name in ["mrag", "split-rag", "split-rag-strategy-weighted", "fusion-rag", "fusion-mrag", "standard-rag"]:
            weighted_ratio[name] = ((np.array(df[name][query_type]["rat"][docs_from:(docs_to+1):docs_step], dtype=float) * weight)
                                    + np.array(df[name][query_type]["category_rat"][docs_from:(docs_to+1):docs_step], dtype=float)) / (weight + 1)

        results = {}
        for key, value in weighted_ratio.items():
            results[key] = list(value.flatten() - weighted_ratio["standard-rag"].flatten())

        flat_x = [x for xs in [[x] * num_queries for x in range(docs_from, docs_to+1, docs_step)] for x in xs]

        box_df = pd.DataFrame({
            "rsr": results["mrag"],
            "articles_fetched": flat_x,
            "rag_type": (["MRAG"] * len(results["mrag"]))
        })

        sns.boxplot(
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
            meanprops={"marker": "p","markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
            legend=False,
            ax=ax[idx]
        )

        avg_mrag = np.mean(weighted_ratio["mrag"] - weighted_ratio["standard-rag"], axis=1)
        sns.pointplot(
            x=range(docs_from, docs_to+1, docs_step), y=avg_mrag,
            linewidth=2.0,
            errwidth=0,
            legend=False,
            linestyle=(0, (1, 1)),
            zorder=101,
            markers=None,
            ax=ax[idx],
        )

        ax[idx].set_ylim(-0.3, 0.6)

        if idx == 0:
            ax[idx].set_yticks(np.arange(-3, 6, dtype=float) / 10)
            ax[idx].set_ylabel("Improvement over\nStandard RAG")
        else:
            ax[idx].set_yticks([])
            ax[idx].set_ylabel(None)
        ax[idx].set_xlabel(None)

        ax[idx].set_title(f"{query_type} Aspects", fontsize=10)

        idx += 1

    fig.text(0.5, -0.06, "Number of Documents Fetched", ha='center')
    plt.savefig(f"relative_weighted_{model}.pdf", format="pdf", bbox_inches="tight")


### Box Plots - Relative Retrieval Improvement - Baselines
# Relative retrieval improvement of MRAG over Standard RAG when compared with Split RAG. Additionally, relative retrieval improvement of Fusion MRAG over Standard RAG when compared with Fusion RAG.

model = "sfr"
weight = 2.0

df = pd.read_json(f"results/25_random_{model}.json")

sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

(fig, ax) = plt.subplots(1, 8, figsize=(11.2, 2.1))

idx = 0
for query_type in [5, 10, 15, 20]:
    docs_from = query_type
    docs_to = query_type+10 # including
    docs_step = 5

    num_queries = len(df["standard-rag"][query_type]["rat"][0])
    sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

    weighted_ratio = {}
    for name in ["mrag", "split-rag", "split-rag-strategy-weighted", "fusion-rag", "fusion-mrag", "standard-rag"]:
        weighted_ratio[name] = ((np.array(df[name][query_type]["rat"][docs_from:(docs_to+1):docs_step], dtype=float) * weight)
                                + np.array(df[name][query_type]["category_rat"][docs_from:(docs_to+1):docs_step], dtype=float)) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(value.flatten() - weighted_ratio["standard-rag"].flatten())

    flat_x = [x for xs in [[x] * num_queries for x in range(docs_from, docs_to+1, docs_step)] for x in xs]

    box_df = pd.DataFrame({
        "rsr": results["mrag"] + results["split-rag"],
        "articles_fetched": flat_x + flat_x,
        "rag_type": (["MRAG"] * len(results["mrag"])) + (["Split RAG"] * len(results["split-rag"]))
    })

    if idx == 6:
        display_legend = True
    else:
        display_legend = False

    sns.boxplot(
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
        meanprops={"marker": "p","markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
        legend=False,
        palette=["#3174a2", "#c44e52"],
        ax=ax[idx]
    )

    avg_mrag = np.mean(weighted_ratio["mrag"] - weighted_ratio["standard-rag"], axis=1)
    sns.pointplot(
        x=range(docs_from, docs_to+1, docs_step), y=avg_mrag,
        linewidth=1.2,
        errwidth=0,
        legend=False,
        linestyle=(0, (1, 1)),
        zorder=102,
        ax=ax[idx]
    )

    box_df = pd.DataFrame({
        "rsr": results["fusion-mrag"] + results["fusion-rag"],
        "articles_fetched": flat_x + flat_x,
        "rag_type": (["Fusion MRAG"] * len(results["fusion-mrag"])) + (["Fusion RAG"] * len(results["fusion-rag"]))
    })

    sns.boxplot(
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
        meanprops={"marker": "p","markerfacecolor": "navy", "markeredgecolor": "navy", "markersize": 2},
        legend=False,
        palette=["#2c5a7b", "#8c8c8c"],
        ax=ax[idx+1]
    )

    avg_mrag = np.mean(weighted_ratio["fusion-mrag"] - weighted_ratio["standard-rag"], axis=1)
    sns.pointplot(
        x=range(docs_from, docs_to+1, docs_step), y=avg_mrag,
        linewidth=1.2,
        errwidth=0,
        legend=False,
        linestyle=(0, (1, 1)),
        zorder=102,
        ax=ax[idx+1]
    )

    ax[idx].set_ylim(-0.5, 0.9)
    ax[idx+1].set_ylim(-0.5, 0.9)

    if idx == 0:
        ax[idx].set_yticks(np.arange(-5, 9, dtype=float) / 10)
        ax[idx].set_ylabel("Improvement over\nStandard RAG")
    else:
        ax[idx].set_yticks([])
        ax[idx].set_ylabel(None)
    ax[idx+1].set_yticks([])
    ax[idx+1].set_ylabel(None)
    ax[idx].set_xlabel(None)
    ax[idx+1].set_xlabel(None)

    ax[idx].set_title(f"{query_type} Aspects\nNo extra cost", fontsize=10)
    ax[idx+1].set_title(f"{query_type} Aspects\nExtra cost", fontsize=10)

    idx += 2

handles = [mpatches.Patch(color="#3174a2", label='MRAG'), mpatches.Patch(color="#c44e52", label='Split RAG'), mpatches.Patch(color="#2c5a7b", label='FusionMRAG'), mpatches.Patch(color="#8c8c8c", label='Fusion RAG')]
plt.legend(handles=handles, loc='lower right', ncol=2, bbox_to_anchor=(0.85,-0.025))
fig.text(0.5, -0.06, "Number of Documents Fetched", ha='center')
plt.savefig(f"relative_baselines_weighted_{model}.pdf", format="pdf", bbox_inches="tight")


### Line Plots - Relative Retrieval Improvement

model = "sfr"
weight = 2.0

for case_study in ["legal", "chemical_manufacturing"]:

    plt.figure(figsize=(0.8, 1.6))
    (fig, ax) = plt.subplots(1, 4, figsize=(3.2, 1.6))

    df = pd.read_json(f"results/25_random_{model}_{case_study}.json")

    sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

    idx = 0
    for query_type in [5, 10, 15, 20]:
        docs_from = query_type
        docs_to = query_type+10 # including
        docs_step = 2

        num_queries = len(df["standard-rag"][query_type]["rat"][0])

        weighted_ratio = {}
        for name in ["mrag", "split-rag", "split-rag-strategy-weighted", "standard-rag"]:
            weighted_ratio[name] = ((np.array(df[name][query_type]["rat"][docs_from:(docs_to+1):docs_step], dtype=float) * weight)
                                    + np.array(df[name][query_type]["category_rat"][docs_from:(docs_to+1):docs_step], dtype=float)) / (weight + 1)

        results = {}
        for key, value in weighted_ratio.items():
            results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

        x_vals = range(docs_from, docs_to+1, docs_step)

        sns.lineplot(x=x_vals, y=results["mrag"], label="MRAG", marker="o", legend=False, ax=ax[idx])
        sns.lineplot(x=x_vals, y=results["split-rag"], label="Split RAG", marker="p", legend=False, color="#c44e52", ax=ax[idx])

        ax[idx].set_ylim(-0.02, 0.13)

        if idx == 0:
            ax[idx].set_yticks(np.arange(-2, 13, 2, dtype=float) / 100)
            ax[idx].set_yticklabels(np.arange(-2, 13, 2, dtype=float) / 100, fontsize=7)
            ax[idx].set_ylabel("Average Improvement\nover Standard RAG", fontsize=7)
        else:
            ax[idx].set_yticks([])
            ax[idx].set_ylabel(None)
        ax[idx].set_xticks(x_vals[::5])
        ax[idx].set_xticklabels(x_vals[::5], fontsize=7)
        ax[idx].set_xlabel(None)

        ax[idx].set_title(f"{query_type} Aspects", fontsize=7)

        idx += 1

    plt.legend(loc='upper right', ncol=2, fontsize=7)
    fig.text(0.5, -0.06, "Number of Documents Fetched", ha='center', fontsize=7)

    plt.savefig(f"line_{case_study}_{model}.pdf", format="pdf", bbox_inches="tight")


### Line Plots - Voting Strategies

model = "sfr"
weight = 2.0

df = pd.read_json(f"results/25_random_{model}.json")

sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

(fig, ax) = plt.subplots(1, 4, figsize=(3.2, 1.6))

idx = 0
for query_type in [5, 10, 15, 20]:
    docs_from = query_type
    docs_to = query_type+10 # including
    docs_step = 2

    num_queries = len(df["standard-rag"][query_type]["rat"][0])

    weighted_ratio = {}
    for name in ["mrag", "mrag-strategy-decay", "mrag-strategy-distance", "standard-rag"]:
        weighted_ratio[name] = ((np.array(df[name][query_type]["rat"][docs_from:(docs_to+1):docs_step], dtype=float) * weight)
                                + np.array(df[name][query_type]["category_rat"][docs_from:(docs_to+1):docs_step], dtype=float)) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

    x_vals = range(docs_from, docs_to+1, docs_step)

    sns.lineplot(x=x_vals, y=results["mrag"], label="MRAG", marker="o", legend=False, ax=ax[idx])
    sns.lineplot(x=x_vals, y=results["mrag-strategy-decay"], label="MRAG (1)", marker="p", legend=False, color="#ca7613ff", ax=ax[idx])
    sns.lineplot(x=x_vals, y=results["mrag-strategy-distance"], label="MRAG (2)", marker="X", legend=False, color="#dd9138ff", ax=ax[idx])

    ax[idx].set_ylim(-0.4,0.5)

    if idx == 0:
        ax[idx].set_yticks(np.arange(-4, 5, 1, dtype=float) / 10)
        ax[idx].set_yticklabels(np.arange(-4, 5, 1, dtype=float) / 10, fontsize=7)
        ax[idx].set_ylabel("Improvement over\nStandard RAG", fontsize=7)
    else:
        ax[idx].set_yticks([])
        ax[idx].set_ylabel(None)
    ax[idx].set_xticks(x_vals[::5])
    ax[idx].set_xticklabels(x_vals[::5], fontsize=7)
    ax[idx].set_xlabel(None)

    ax[idx].set_title(f"{query_type} Aspects", fontsize=7)

    idx += 1

plt.legend(loc='lower right', ncol=3, fontsize=6, columnspacing = 0.4, labelspacing = 0.2)
fig.text(0.5, -0.06, "Number of Documents Fetched", ha='center', fontsize=7)

plt.savefig(f"voting_mrag_{model}.pdf", format="pdf", bbox_inches="tight")


### Line Plots - Relative Retrieval Improvement - Split RAG Voting Strategies

model = "sfr"
weight = 2.0

df = pd.read_json(f"results/25_random_{model}.json")

sns.set_style(rc = {"axes.facecolor": "#eaeaf2ff"})

(fig, ax) = plt.subplots(1, 4, figsize=(3.2, 1.6))

idx = 0
for query_type in [5, 10, 15, 20]:
    docs_from = query_type
    docs_to = query_type+10 # including
    docs_step = 2

    num_queries = len(df["standard-rag"][query_type]["rat"][0])

    weighted_ratio = {}
    for name in ["mrag", "split-rag", "split-rag-strategy-weighted", "standard-rag"]:
        weighted_ratio[name] = ((np.array(df[name][query_type]["rat"][docs_from:(docs_to+1):docs_step], dtype=float) * weight)
                                + np.array(df[name][query_type]["category_rat"][docs_from:(docs_to+1):docs_step], dtype=float)) / (weight + 1)

    results = {}
    for key, value in weighted_ratio.items():
        results[key] = list(np.mean(value - weighted_ratio["standard-rag"], axis=1))

    x_vals = range(docs_from, docs_to+1, docs_step)

    sns.lineplot(x=x_vals, y=results["mrag"], label="MRAG", marker="o", legend=False, ax=ax[idx])
    sns.lineplot(x=x_vals, y=results["split-rag"], label="Split RAG (1)", marker="p", legend=False, ax=ax[idx], color="#c44e52")
    sns.lineplot(x=x_vals, y=results["split-rag-strategy-weighted"], label="Split RAG (2)", marker="X", legend=False, ax=ax[idx], color="#937860")

    ax[idx].set_ylim(0,0.22)

    if idx == 0:
        ax[idx].set_yticks(np.arange(0, 22, 4, dtype=float) / 100)
        ax[idx].set_yticklabels(np.arange(0, 22, 4, dtype=float) / 100, fontsize=7)
        ax[idx].set_ylabel("Improvement over\nStandard RAG", fontsize=7)
    else:
        ax[idx].set_yticks([])
        ax[idx].set_ylabel(None)
    ax[idx].set_xticks(x_vals[::5])
    ax[idx].set_xticklabels(x_vals[::5], fontsize=7)
    ax[idx].set_xlabel(None)

    ax[idx].set_title(f"{query_type} Aspects", fontsize=7)

    idx += 1

plt.legend(loc='lower right', ncol=3, fontsize=6, columnspacing = 0.4, labelspacing = 0.2, bbox_to_anchor=(0.95,-0.025))
fig.text(0.5, -0.06, "Number of Documents Fetched", ha='center', fontsize=7)

plt.savefig(f"voting_split_rag_{model}.pdf", format="pdf", bbox_inches="tight")
