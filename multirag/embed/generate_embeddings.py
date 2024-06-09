# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Lucas Weitzendorf
# Roman Niggli

import os
import json

from typing import Any, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from multirag.dataset import (
    Article,
    load_articles,
    Query,
    FusionQuery,
    QueryEncoder,
    load_queries
)

Embedding = list[float]


@dataclass(frozen=True)
class LayerEmbeddings:
    """
    Data class to store the attention heads of a layer.
    """
    attention_heads: list[Embedding]

    @classmethod
    def from_dict(cls, emb_dict: dict):
        return cls(**emb_dict)


@dataclass(frozen=True)
class FullEmbeddings:
    """
    Data class to store the standard embedding as well as embeddings from the
    different attention heads for the various layers.
    """
    standard_embedding: Embedding
    layer_embeddings: dict[int, LayerEmbeddings]

    @classmethod
    def from_dict(cls, emb_dict: dict) -> 'FullEmbeddings':
        layer_embeddings: dict[int, LayerEmbeddings] = {}
        for layer_idx, l_emb_dict in emb_dict["layers"].items():
            layer_embeddings[int(layer_idx)] = LayerEmbeddings.from_dict(l_emb_dict)
        return cls(
            emb_dict["standard"],
            layer_embeddings
        )


@dataclass(frozen=True)
class ArticleEmbeddings:
    """
    Data class to store the article information as well as the full embeddings
    (standard embedding, attention head embeddings).
    """
    article: Article
    embeddings: FullEmbeddings

    @classmethod
    def from_dict(cls, emb_dict: dict) -> 'ArticleEmbeddings':
        return cls(
            Article.from_dict(emb_dict["article"]),
            FullEmbeddings.from_dict(emb_dict["embeddings"]),
        )


@dataclass(frozen=True)
class QueryEmbeddings:
    """
    Data class to store the query information as well as the full embeddings
    (standard embedding, attention head embeddings).
    """
    query: Query
    embeddings: FullEmbeddings


@dataclass(frozen=True)
class FusionQueryEmbeddings(QueryEmbeddings):
    """
    Data class to store the fusion query information as well as the full embeddings
    (standard embedding, attention head embeddings).
    """
    query: FusionQuery
    fusion_embeddings: list[FullEmbeddings]


class EmbeddingModel:
    """
    Class that defines the interface for the Salesforce/SFR-Embedding-Mistral embedding model.
    """
    class CachingModule(torch.nn.Module):
        """
        Custom wrapper around an instance of :class:`nn.Module` that caches its inputs.
        """
        def __init__(self, module: torch.nn.Module) -> None:
            """
            Initialize the cache instance with the Torch module.

            :param module: Torch module.
            :type module: torch.nn.Module
            """
            super().__init__()
            self._module = module
            self.last_input: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Store x and forward to wrapped :class:`nn.Module` instance.

            :param x: Input.
            :type x: torch.nn.Tensor
            :return: Output of the wrapped module class instance.
            :rtype: torch.nn.Tensor
            """
            self.last_input = x
            return self._module.forward(x)

    def __init__(self, target_layers: Optional[set[int]], device: str) -> None:
        """
        Initialize the embedding model.

        :param target_layers: Layers to target.
        :type target_layers: Optional[set[int]]
        :param device: The device to load the model on.
        :type device: str
        """
        self._tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
        self._model = AutoModel.from_pretrained(
            "Salesforce/SFR-Embedding-Mistral",
            device_map=device
        )
        self.target_layers = target_layers or {len(self._model.layers) - 1}

        for layer in self._model.layers:
            layer.self_attn.o_proj = self.CachingModule(layer.self_attn.o_proj)

    def generate_embeddings(self, text: str) -> FullEmbeddings:
        """
        Generate embeddings (standard embedding, attention head embeddings) for the input text.

        :param text: Input text.
        :type text: str
        :return: Embeddings.
        :rtype: FullEmbeddings
        """
        all_layer_embeddings: dict[int, LayerEmbeddings] = {}

        def split(_embedding: Embedding, _interval: int) -> list[Embedding]:
            _sub_embeddings: list[Embedding] = []
            for i in range(0, len(_embedding), _interval):
                _sub_embeddings.append(_embedding[i:(i + _interval)])
            return _sub_embeddings

        with torch.no_grad():
            batch_dict = self._tokenizer([text], padding=True, return_tensors="pt")
            iids = batch_dict["input_ids"].to(self._model.device)
            hidden_states = self._model.embed_tokens(iids)

            for layer_idx, layer in enumerate(self._model.layers):
                # run inference for this layer
                hidden_states = layer(hidden_states)[0]

                if layer_idx in self.target_layers:
                    # retrieve cached input from custom module
                    attn_heads = layer.self_attn.o_proj.last_input[0][-1]
                    layer_embeddings = LayerEmbeddings(
                        attention_heads=split(attn_heads.tolist(), 128),
                    )
                    all_layer_embeddings[layer_idx] = layer_embeddings

            # apply last layer, followed by the RMS norm
            standard_embedding = self._model.norm(hidden_states)[0][-1].tolist()
            return FullEmbeddings(standard_embedding, all_layer_embeddings)


def embed_articles(articles: list[Article], model: EmbeddingModel) -> list[ArticleEmbeddings]:
    """
    Embed Article documents.

    :param articles: List of Article documents.
    :type articles: list[Article]
    :param model: Embedding model to use.
    :type model: EmbeddingModel
    :return: List of embeddings for the Article documents.
    :rtype: list[ArticleEmbeddings]
    """
    for article in tqdm(articles, "Generating article embeddings"):
        embeddings = model.generate_embeddings(article.text)
        article_embedding = ArticleEmbeddings(article, embeddings)
        yield article_embedding


def embed_queries(queries: list[Query], model: EmbeddingModel) -> list[QueryEmbeddings]:
    """
    Embed queries.

    :param queries: List of queries.
    :type queries: list[Query]
    :param model: Embedding model to use.
    :type model: EmbeddingModel
    :return: List of embeddings for the queries.
    :rtype: list[QueryEmbeddings]
    """
    def construct_input(_instruct: str, _query: str) -> str:
        """
        Construct prompt.

        :param _instruct: Instruction part of the prompt.
        :type _instruct: str
        :param _query: Query part of the prompt.
        :type _query: str
        :return: Constructed prompt.
        :rtype: str
        """
        return f"Instruct: {_instruct}\n\nQuery: {_query}"

    fusion_instruction = "Given a web search query, retrieve relevant passages that answer the query."
    query_instruction = (
        "Given a story, retrieve relevant documents that provide contextual information "
        "about topics brought up in the story."
    )

    for query in tqdm(queries, "Generating query embeddings"):
        text = construct_input(query_instruction, query.text)
        default_embeddings = model.generate_embeddings(text)

        if isinstance(query, FusionQuery):
            fusion_embeddings: list[FullEmbeddings] = []
            for prompt in query.fusion_prompts:
                text = construct_input(fusion_instruction, prompt)
                embeddings = model.generate_embeddings(text)
                fusion_embeddings.append(embeddings)
            query_embeddings = FusionQueryEmbeddings(query, default_embeddings, fusion_embeddings)
        else:
            query_embeddings = QueryEmbeddings(query, default_embeddings)

        yield query_embeddings


class EmbeddingEncoder(json.JSONEncoder):
    """
    Class to handle the JSON storage of the various embedding data classes.
    """
    def default(self, o: object) -> Any:
        """
        Extract the various attributes from the JSON storage.

        :param o: JSON storage.
        :type o: object
        :return: Attributes of the JSON storage.
        :rtype: Any
        """
        if isinstance(o, ArticleEmbeddings):
            return {
                "article": o.article.__dict__,
                "embeddings": self.default(o.embeddings)
            }
        elif isinstance(o, FullEmbeddings):
            return {
                "standard": o.standard_embedding,
                "layers": {k: e.__dict__ for k, e in o.layer_embeddings.items()},
            }
        if isinstance(o, FusionQueryEmbeddings):
            return {
                "query": QueryEncoder().default(o.query),
                "embeddings": self.default(o.embeddings),
                "fusion_embeddings": [self.default(e) for e in o.fusion_embeddings],
            }
        elif isinstance(o, QueryEmbeddings):
            return {
                "query": QueryEncoder().default(o.query),
                "embeddings": self.default(o.embeddings),
            }
        return super().default(o)


def _load_embeddings(
        file_path: str,
        articles: Optional[list[Article]]
) -> tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]:
    """
    Load the document and query embeddings from a JSON file.

    If the articles parameter is provided, some of the attributes are read from that
    data structure instead of from the JSON file.

    :param file_path: Path to the input JSON file.
    :type file_path: str
    :param articles: Article documents.
    :type articles: Optional[list[Article]]
    :return: Document and query embeddings.
    :rtype: tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]
    """
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    article_embeddings: list[ArticleEmbeddings] = []
    query_embeddings: list[QueryEmbeddings] = []

    for article_emb_dict in json_data["articles"]:
        article_emb = ArticleEmbeddings.from_dict(article_emb_dict)
        article_embeddings.append(article_emb)

    if articles:
        articles_by_title = {a.title: a for a in articles}
    else:
        articles_by_title = {e.article.title: e.article for e in article_embeddings}

    for query_emb_dict in json_data["queries"]:
        try:
            query_dict = query_emb_dict["query"]
            topics: set[Article] = {articles_by_title[title] for title in query_dict["topics"]}
        except KeyError:
            continue

        text: str = query_dict["text"]
        default_emb: FullEmbeddings = FullEmbeddings.from_dict(query_emb_dict["embeddings"])

        if "fusion" in query_dict:
            query = FusionQuery(topics, text, query_dict["fusion"])
            fusion_emb = [FullEmbeddings.from_dict(d) for d in query_emb_dict["fusion_embeddings"]]
            query_emb = FusionQueryEmbeddings(query, default_emb, fusion_emb)
        else:
            query = Query(topics, text)
            query_emb = QueryEmbeddings(query, default_emb)

        query_embeddings.append(query_emb)

    return article_embeddings, query_embeddings


def load_embeddings(file_path: str) -> tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]:
    """
    Load the document and query embeddings from a JSON file.

    :param file_path: Path to the input JSON file.
    :type file_path: str
    :return: Document and query embeddings.
    :rtype: tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]
    """
    return _load_embeddings(file_path, None)


def generate_embeddings(
        article_path: str,
        query_path: str,
        target_layers: Optional[set[int]],
        export_path: str
) -> tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]:
    """
    Generate embeddings of the documents in the dataset and the queries.

    :param article_path: Path to the JSON file containing the dataset.
    :type article_path: str
    :param query_path: Path to the JSON file containing the queries.
    :type query_path: str
    :param target_layers: Layers to target for the attention head embeddings.
    :type target_layers: Optional[set[int]]
    :param export_path: Path to the output file.
    :type export_path: str
    :return: Document and query embeddings.
    :rtype: tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]
    """
    articles: list[Article] = load_articles(article_path)
    queries: list[Query] = load_queries(query_path, articles)

    try:
        article_embeddings, query_embeddings = _load_embeddings(export_path, articles)
    except (FileNotFoundError, json.JSONDecodeError):
        article_embeddings, query_embeddings = [], []

    if article_embeddings or query_embeddings:
        print("Found existing usable embeddings:")
        print(f"  {len(article_embeddings)} article embeddings")
        print(f"  {len(query_embeddings)} query embeddings")
        choice = input("Do you want to overwrite them? [y/N] ")
        if choice.strip().lower() in ('y', 'yes'):
            article_embeddings.clear()
            query_embeddings.clear()

    existing_articles = {e.article for e in article_embeddings}
    articles = [a for a in articles if a not in existing_articles]

    existing_queries = {e.query for e in query_embeddings}
    queries = [q for q in queries if q not in existing_queries]

    if not (articles or queries):
        print("No new embeddings were added.")
        return article_embeddings, query_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingModel(target_layers, device)

    try:
        for article_emb in embed_articles(articles, model):
            article_embeddings.append(article_emb)

        for query_emb in embed_queries(queries, model):
            query_embeddings.append(query_emb)
    except KeyboardInterrupt:
        print("Embedding generation canceled.")

    json_data = {
        "articles": article_embeddings,
        "queries": query_embeddings
    }

    print(f"Saving data in {export_path}...")
    if export_dir := os.path.dirname(export_path):
        os.makedirs(export_dir, exist_ok=True)
    with open(export_path, 'w') as file:
        json.dump(json_data, file, indent=4, cls=EmbeddingEncoder)

    return article_embeddings, query_embeddings
