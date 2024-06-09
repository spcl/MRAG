# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Lucas Weitzendorf
# Roman Niggli

import json
import heapq
import os

import numpy as np

from abc import abstractmethod
from typing import Callable, Union
from operator import itemgetter
from dataclasses import dataclass
from tqdm import tqdm

from multirag.dataset import (
    Article
)
from multirag.storage import (
    VectorDB
)
from multirag.embed import (
    FullEmbeddings,
    QueryEmbeddings,
    FusionQueryEmbeddings,
    load_embeddings
)


@dataclass(frozen=True)
class StrategyResult:
    """
    Data class to represent the different results of a strategy.

    success: binary indicator for whether all aspects were identified
    success_ratio: relative percentage of correctly identified aspects
    category_success: binary indicator for whether all categories were identified
    category_success_ratio: relative percentage of aspects from correct categories
    """
    success: list[list[int]]
    success_ratio: list[list[float]]
    category_success: list[list[int]]
    category_success_ratio: list[list[float]]


class Strategy:
    """
    Abstract base class that defines the interface for the evaluation strategies.
    """
    def __init__(self, name: str, db: VectorDB) -> None:
        """
        Initialize the Strategy instance with a name and a vector database instance.

        :param name: Name of the strategy class.
        :type name: str
        :param db: Vector database instance.
        :type db: VectorDB
        """
        self.name = name
        self.db = db

    @abstractmethod
    def _get_picks(self, query_embs: QueryEmbeddings, n: int) -> Union[tuple[Article, ...], list[tuple[Article, ...]]]:
        """
        Retrieve the n documents closest to query_embs.

        :param query_embs: Query embeddings to evaluate.
        :type query_embs: QueryEmbedding
        :param n: Number of documents to retrieve.
        :type n: int
        :return: Either the n retrieved documents, or a list of n retrieval selections.
        :type: Union[tuple[Article, ...], list[tuple[Article, ...]]]
        """
        pass

    def run(self, query_embs: list[QueryEmbeddings], n: int) -> StrategyResult:
        """
        Run the evaluation of the chosen strategy.

        :param query_embs: Query embeddings to evaluate.
        :type query_embs: QueryEmbedding
        :param n: Maximum number of documents retrieved per query.
        :type n: int
        :return: Results for the strategy.
        :type: StrategyResult
        """
        success = np.ndarray(shape=(len(query_embs), n), dtype=int)
        success_ratio = np.ndarray(shape=(len(query_embs), n), dtype=float)
        category_success = np.ndarray(shape=(len(query_embs), n), dtype=int)
        category_success_ratio = np.ndarray(shape=(len(query_embs), n), dtype=float)

        for i, query_emb in enumerate(query_embs):
            rel = query_emb.query.topics
            category_rel = {a.label for a in rel}
            picks = self._get_picks(query_emb, n)

            for j in range(1, n + 1):
                sub_picks: tuple[Article, ...] = picks[j-1] if isinstance(picks, list) else picks[:j]

                fetched: set[Article] = set(sub_picks)
                success[i][j-1] = 1 if rel.issubset(fetched) else 0
                success_ratio[i][j-1] = len(rel & fetched) / len(rel)

                category_fetched = {a.label for a in fetched}
                category_success[i][j-1] = 1 if category_rel.issubset(category_fetched) else 0
                category_success_ratio[i][j-1] = len(category_rel & category_fetched) / len(category_rel)

        # transpose
        return StrategyResult(
            success.T.tolist(),
            success_ratio.T.tolist(),
            category_success.T.tolist(),
            category_success_ratio.T.tolist(),
        )


class StandardStrategy(Strategy):
    """
    The StandardStrategy class uses the standard search for the evaluation.

    Inherits from the Strategy class and implements its abstract methods.
    """

    def _get_picks(self, query_embs: QueryEmbeddings, n: int) -> tuple[Article, ...]:
        """
        Retrieve the n documents closest to query_embs within the standard
        embedding space.

        :param query_embs: Query embeddings to evaluate.
        :type query_embs: QueryEmbedding
        :param n: Number of documents to retrieve.
        :type n: int
        :return: Tuple with the retrieved documents from closest to furthest.
        :type: tuple[Article, ...]
        """
        return tuple(doc for (score, doc) in self.db.standard_search(query_embs.embeddings, n))


class MultiHeadStrategy(Strategy):
    """
    The MultiHeadStrategy class uses the Multi-Head RAG strategy for the evaluation.

    Inherits from the Strategy class and implements its abstract methods.
    """

    def __init__(
            self,
            name: str,
            db: VectorDB,
            layer: int,
            weight_fn: Callable[[float, int, float], float],
    ) -> None:
        """
        Initialize the MultiHeadStrategy instance with a name, a vector database instance,
        layer information as well as a vote function.

        :param name: Name of the strategy class.
        :type name: str
        :param db: Vector database instance.
        :type db: VectorDB
        :param layer: Layer to use embeddings from.
        :type layer: int
        :param weight_fn: Function to compute votes for a document based on head-scale,
            rank, and distance between query and document.
        :type weight_fn: Callable[[float, int, float], float]
        """
        super().__init__(name, db)
        self.weight_fn = weight_fn
        self.layer = layer

    def _search(self, emb: FullEmbeddings, n: int) -> list[list[tuple[float, Article]]]:
        """
        Search for closet neighbors of emb within the space of each attention head.

        :param emb: Query embeddings.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List with search results (ordered list of (distance, Article) pairs)
            for each attention head.
        :rtype: list[list[tuple[float, Article]]]
        """
        return self.db.attention_search(emb, self.layer, n)

    def _get_head_scales(self) -> list[float]:
        """
        Get the scales for each attention head. The scale of an attention head is the
        product of the mean pairwise distance between documents for that head, and the mean
        embedding norm of all documents of that head.

        :return: List with the attention scales.
        :rtype: list[float]
        """
        return self.db.attention_scales

    def _multi_vote(self, emb: FullEmbeddings, n: int) -> list[tuple[float, Article]]:
        """
        Accumulate all votes over all attention heads. Each head votes for its n closest
        documents for the provided embedding, with the i-th closest receiving 2**-i votes.
        All votes are scaled with the respective head's head-scale.

        :param emb: Query embedding to retrieve documents for.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: Sorted list of the top n (votes, Article) pairs (most to least votes).
        :rtype: list[tuple[float, Article]]
        """
        votes: dict[Article, float] = {}
        ranking = self._search(emb, n)
        head_scales: list[float] = self._get_head_scales()

        for i, head in enumerate(ranking):
            for rank, (dist, voted) in enumerate(head[:n]):
                votes[voted] = votes.get(voted, 0.0) + self.weight_fn(head_scales[i], rank, dist)

        top_picks: list[tuple[Article, float]] = heapq.nlargest(n, votes.items(), key=itemgetter(1))
        return [(votes, article) for (article, votes) in top_picks]

    def _get_picks(self, query_embs: QueryEmbeddings, n: int) -> tuple[Article, ...]:
        """
        Use _multi_vote to pick the top n documents to retrieve, return the documents
        in order from the first to the nth pick.

        :param query_embs: Query embeddings to evaluate.
        :type query_embs: QueryEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List of the n retrieved documents.
        :type: tuple[Article, ...]
        """
        return tuple(doc for (votes, doc) in self._multi_vote(query_embs.embeddings, n))


class SplitStrategy(MultiHeadStrategy):
    """
    The SplitStrategy class uses the Split-RAG strategy for the evaluation. It is similar
    to Multi-Head RAG, but instead of attention-embeddings uses segments of the standard
    embedding.

    Inherits from the MultiHeadStrategy class and overwrites some of its abstract methods.
    """

    def _search(self, emb: FullEmbeddings, n: int) -> list[list[tuple[float, Article]]]:
        """
        Use search within the segments of the standard embedding.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List with search results (ordered list of (distance, Article) pairs)
            for each segment.
        :rtype: list[list[tuple[float, Article]]]
        """
        return self.db.cut_standard_search(emb, n)

    def _get_head_scales(self) -> list[float]:
        """
        Get the scales for each segment of the standard embedding. The scale of each segment
        is the product of the mean pairwise distance between documents within that segment,
        and the mean embedding norm of all documents in that segment.

        :return: List with the segment scales.
        :rtype: list[float]
        """
        return self.db.cut_standard_scales


class FusionStrategy(Strategy):
    """
    The FusionStrategy class uses additionally the RAG Fusion approach for the evaluation.

    Inherits from the Strategy class and implements its abstract methods.
    """

    def _score(self, emb: FullEmbeddings, n: int) -> list[tuple[float, Article]]:
        """
        Pick the n closest documents for the provided query embedding in the standard
        embedding space. For each, return both the document and its distance to the query.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List with the chosen documents and their distance to the query.
        :rtype: list[tuple[float, Article]]
        """
        return self.db.standard_search(emb, n)

    @staticmethod
    def _reciprocal_rank_fusion(search_results: list[list[tuple[float, Article]]], k: int = 60) -> list[Article]:
        """
        This function is adapted from a function of the same name in RAG-Fusion by
        Zackary Rackauckas, see the file `rag_fusion.py` for more info.

        :param search_results: List with the search results for each fusion query.
        :type search_results: list[list[tuple[float, Article]]]
        :param k: Ranking constant. Defaults to 60.
        :type k: int
        :return: Fusion of the individual rankings.
        :rtype: list[Article]
        """
        fused_scores: dict[Article, float] = {}
        for doc_scores in search_results:
            for rank, (score, doc) in enumerate(sorted(doc_scores)):
                fused_scores[doc] = fused_scores.get(doc, 0.0) + 1 / (rank + k)

        reranked_results = []
        for doc, score in sorted(fused_scores.items(), key=itemgetter(1), reverse=True):
            reranked_results.append(doc)

        return reranked_results

    def _get_picks(self, query_emb: FusionQueryEmbeddings, n: int) -> list[tuple[Article, ...]]:
        """
        Adaptation of the RAG fusion algorithm by Zackary Rackauckas.
        This strategy does not return a tuple, but instead a list of tuples. Each
        element of the returned list represents the selection for the respective
        number of documents to fetch.

        :param query_emb: Query embedding to evaluate.
        :type query_emb: FusionQueryEmbedding
        :param n: Maximum number of documents to retrieve.
        :type n: int
        :return: List with selections for retrieving [1, ..., n] documents.
        :type: list[tuple[Article, ...]]
        """
        if not isinstance(query_emb, FusionQueryEmbeddings):
            raise Exception("RAG fusion requires fusion embeddings")

        picks = []
        # start with large n to optimize for DB caching
        for i in range(n, 0, -1):
            scores = [self._score(q, i) for q in query_emb.fusion_embeddings]
            fused: list[Article] = self._reciprocal_rank_fusion(scores)
            picks.append(tuple(fused[:i]))

        return picks[::-1]


class MultiHeadFusionStrategy(FusionStrategy, MultiHeadStrategy):
    """
    The MultiHeadFusionStrategy class uses additionally the RAG Fusion approach for
    the MultiHeadStrategy for the evaluation.

    Inherits from the FusionStrategy and MultiHeadStrategy classes and overwrites some of their abstract methods.
    """

    def _score(self, emb: FullEmbeddings, n: int) -> list[tuple[float, Article]]:
        """
        Pick the top n documents for a given query embedding, based on the MultiHeadStrategy.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List with the chosen documents and their scores.
        :rtype: list[tuple[float, Article]]
        """
        return [(-votes, doc) for (votes, doc) in self._multi_vote(emb, n)]


def run_strategies(
    embedding_path: str,
    vector_db: VectorDB,
    num_picks: int,
    layer: int,
    export_path: str
) -> dict[str, dict[int, StrategyResult]]:
    """
    Run various evaluation strategies on the query embeddings in combination with the document embeddings.

    :param embedding_path: Path to the file with the embeddings.
    :type embedding_path: str
    :param vector_db: Vector database instance to use.
    :type vector_db: VectorDB
    :param num_picks: Number of picks.
    :type num_picks: int
    :param layer: Layer to run the evaluation on.
    :type layer: int
    :param export_path: Path to the output file.
    :type export_path: str
    :return: Results of the various evaluation strategies.
    :type: dict[str, dict[int, StrategyResult]]
    """
    strategies: list[Strategy] = [
        StandardStrategy("standard-rag", vector_db),
        MultiHeadStrategy("multirag", vector_db, layer, lambda h, r, d: h * (2 ** -r)),
        MultiHeadStrategy("multirag-strategy-decay", vector_db, layer, lambda h, r, d: (2 ** -r)),
        MultiHeadStrategy("multirag-strategy-distance", vector_db, layer, lambda h, r, d: 1 / d),
        SplitStrategy("split-rag", vector_db, layer, lambda h, r, d: 2 ** -r),
        SplitStrategy("split-rag-strategy-weighted", vector_db, layer, lambda h, r, d: h * (2 ** -r)),
    ]

    _, query_embeddings = load_embeddings(embedding_path)

    if all(isinstance(e, FusionQueryEmbeddings) for e in query_embeddings):
        print("Detected fusion queries, added fusion strategies to schedule.")
        strategies.extend([
            FusionStrategy("fusion-rag", vector_db),
            MultiHeadFusionStrategy("fusion-multirag", vector_db, layer, lambda h, r, d: h * (2 ** -r)),
        ])

    queries_by_num_topics: dict[int, list[QueryEmbeddings]] = {}
    for query_emb in query_embeddings:
        n_rel = len(query_emb.query.topics)
        if n_rel not in queries_by_num_topics:
            queries_by_num_topics[n_rel] = []
        queries_by_num_topics[n_rel].append(query_emb)

    res: dict[str, dict[int, StrategyResult]] = {}
    for strategy in strategies:
        res[strategy.name] = {}
        for n_rel, query_embs in tqdm(queries_by_num_topics.items(), strategy.name):
            res[strategy.name][n_rel] = strategy.run(query_embs, num_picks)

    if export_path is None:
        return res

    print(f"Saving data in {export_path}...")
    if export_dir := os.path.dirname(export_path):
        os.makedirs(export_dir, exist_ok=True)
    with open(export_path, 'w') as file:
        json.dump(res, file, indent=4, default=lambda o: o.__dict__)

    return res
