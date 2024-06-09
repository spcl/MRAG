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
import random

from enum import Enum
from typing import Any
from dataclasses import dataclass

from tqdm import tqdm
from openai import OpenAI

from multirag.dataset import Article, load_articles


@dataclass(frozen=True)
class Query:
    """
    Data class to represent the different aspects of a query.

    Each query has a set of Articles associated with it as well as the query itself in the member text.
    """
    topics: set[Article]
    text: str

    def __hash__(self):
        return hash(self.text)


@dataclass(frozen=True)
class FusionQuery(Query):
    """
    Data class to represent the different aspects of a fusion query.

    A fusion query consists of a list of prompts.
    """
    fusion_prompts: list[str]

    def __hash__(self):
        return hash(self.text + ''.join(self.fusion_prompts))


class QueryEncoder(json.JSONEncoder):
    """
    Class to handle the combination of FusionQueries and Queries.
    """
    def default(self, o: object) -> Any:
        if isinstance(o, FusionQuery):
            return {
                "topics": [a.title for a in o.topics],
                "text": o.text,
                "fusion": o.fusion_prompts,
            }
        elif isinstance(o, Query):
            return {
                "topics": [a.title for a in o.topics],
                "text": o.text,
            }
        return super().default(o)


class QueryGenerator:
    """
    Class that uses the OpenAI API to generate the queries.

    It supports GPT-3.5-Turo, GPT-4, GPT-4-Turbo and GPT-4o.
    """
    class Model(Enum):
        GPT_3_5_TURBO = "gpt-3.5-turbo"
        GPT_4 = "gpt-4"
        GPT_4_TURBO = "gpt-4-turbo"
        GPT_4O = "gpt-4o"

    def __init__(self, model: Model) -> None:
        self._openai_client = OpenAI()
        self.model = model

    @staticmethod
    def _construct_prompt(topics: set[Article]) -> str:
        """
        Based on a list of article titles, construct a prompt for an LLM.
        This prompt instructs the LLM to create a story about all articles.

        :param topics: Articles to be used to construct the prompt.
        :type topics: set[Article]
        :return: A prompt for an LLM.
        :rtype: str
        """
        titles = [a.title for a in topics]
        setup = (
            f"Please create a story about the attached {len(topics)} articles on the topics {', '.join(titles)}.\n"
            "It is very important that each of the attached articles is relevant to the story, in a way "
            "that references the content of the article, not just its title. But please also mention each "
            "title at least once. Please make sure that all of the attached articles are relevant to your "
            "story, and that each article is referenced in at least two sentences! They do not necessarily "
            "have to be referenced in the same order, but make sure no article is forgotten.\n"
            "Important: Output only the story, no additional text. And do not use bullet points, or "
            "paragraphs.\n\n"
            "Articles:\n"
        )
        postfix = (
            "---------\n" +
            f"Again, make sure that you reference all the following topics in your story: {', '.join(titles)}\n"
        )

        articles = ""  # attach articles
        for article in topics:
            articles += (
                "---------\n" +
                f"{article.title}:\n" +  # article title
                article.text + '\n'  # article body
            )

        return setup + articles + postfix

    def query_from_topics(self, topics: set[Article]) -> Query:
        """
        Generate a synthetic query via the OpenAI API.

        :param topics: Articles to be used in the query.
        :type topics: set[Article]
        :return: Synthetic query for the synthetic dataset.
        :rtype: Query
        """
        response = self._openai_client.chat.completions.create(
            model=self.model.value,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a very curious scholar asking about possible "
                        "connections between different topics."
                    )
                },
                {
                    "role": "user",
                    "content": self._construct_prompt(topics)
                }
            ]
        ).choices[0].message.content.strip()
        return Query(topics, response)

    def fusion_from_query(self, query: Query, num_queries: int) -> FusionQuery:
        """
        Generate a synthetic fusion query via the OpenAI API.
        A fusion query asks additional questions about a query to an LLM to produce better results.

        :param query: A synthetic query for the synthetic dataset.
        :type query: Query
        :param num_queries: Number of fusion prompts to generate for the given query.
        :type num_queries: int
        :return: A fusion query with additional LLM prompts.
        :rtype: FusionQuery
        """
        response = self._openai_client.chat.completions.create(
            model=self.model.value,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates multiple search queries "
                        "based on a single input query."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate multiple search queries related to: {query.text}"},
                {
                    "role": "user",
                    "content": f"OUTPUT ({num_queries} queries):"
                }
            ]
        ).choices[0].message.content.strip()
        return FusionQuery(query.topics, query.text, response.split('\n'))


def _sample_query_topics(articles: list[Article], k: int, n: int) -> list[set[Article]]:
    """
    Sort list of articles by category and pick k topics for n queries.

    :param articles: List of articles to choose from for the queries.
    :type articles: list[Article]
    :param k: Number of articles per query.
    :type k: int
    :param n: Number of queries.
    :type n: int
    :return: List of articles for the queries.
    :rtype: list[set[Article]]
    """
    query_topics: list[set[Article]] = []

    articles_by_category = {}
    for article in articles:
        if article.label not in articles_by_category:
            articles_by_category[article.label] = []
        articles_by_category[article.label].append(article)

    categories = list(articles_by_category.keys())

    while len(query_topics) < n:
        topics: set[Article] = set()
        for category in random.sample(categories, k):
            pick: Article = random.choice(articles_by_category[category])
            topics.add(pick)

        query_topics.append(topics)

    return query_topics


def load_queries(file_path: str, articles: list[Article]) -> list[Query]:
    """
    Load queries stored in a JSON file.

    :param file_path: Path to the query file.
    :type file_path: str
    :param articles: List of articles.
    :type articles: list[Article]
    :return: List of queries.
    :rtype: list[Query]
    """
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    queries: list[Query] = []
    articles_by_title = {a.title: a for a in articles}

    for query_dict in json_data:
        try:
            topics = {articles_by_title[title] for title in query_dict["topics"]}
        except KeyError:
            continue

        text = query_dict["text"]
        if fusion := query_dict.get("fusion", None):
            query = FusionQuery(topics, text, fusion)
        else:
            query = Query(topics, text)

        queries.append(query)

    return queries


def _check_query(query: Query) -> bool:
    """
    Check whether the queries adhere to certain constraints.
    - query should be at least 100 characters
    - check whether all article title are mentioned

    :param query: Query to check.
    :type query: Query
    :return: True if query fufills constraints. False otherwise.
    :rtype: bool
    """
    if len(query.text) < 100:
        return False

    def is_mentioned(_article: Article) -> bool:
        """
        Check if the specified title is mentioned in the provided text.
        This is a very primitive implementation that can fail quite easily.
        For instance, the title may appear in its plural form (title: "Fairy",
        text contains: "Fairies"), which won't be caught by this function.

        :param _article: Article to check for.
        :type _article: Article
        :return: True if article title is found in the query. False otherwise.
        :rtype: bool
        """
        title = _article.title
        title = title.removeprefix("the")
        title = title.split('(')[0].strip()
        return all(word.lower() in query.text.lower() for word in title.split())

    return all(map(is_mentioned, query.topics))


def _save_to_file(queries: list[Query], file_path: str) -> None:
    """
    Store list of queries in a JSON file.

    :param queries: List of queries.
    :type queries: list[Query]
    :param file_path: Path to the output file.
    :type file_path: str
    """
    if export_dir := os.path.dirname(file_path):
        os.makedirs(export_dir, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(queries, file, indent=4, cls=QueryEncoder)


def _review_queries(
        queries: list[Query],
        default_generator: QueryGenerator,
        export_path: str
) -> list[Query]:
    """
    Iterate through all generated queries. For the ones with "text" field == "" do:
    - generate new query
    - ask for manual decision: Is this query good?
    - if not, retry

    Note that for high numbers of related topics, it can become practically impossible
    to obtain a query that contains every single specified topic. In that case, it is
    recommended to try a couple of times, then try with gpt-4o a few times, and maybe
    pick one result where only very few topics are missing.

    :param queries: List of already generated queries.
    :type queries: list[Query]
    :param default_generator: Query generator class to use.
    :type default_generator: QueryGenerator
    :param export_path: Path to the output file.
    :type export_path: str
    :return: List of queries.
    :rtype: list[Query]
    """
    to_be_reviewed = [(i, query) for i, query in enumerate(queries) if not _check_query(query)]
    if not to_be_reviewed:
        print("No queries require manual review.")
        return queries

    advanced_generator = QueryGenerator(QueryGenerator.Model.GPT_4O)
    print()

    for i, (q_idx, query) in enumerate(to_be_reviewed):
        while not _check_query(query):
            print(f"{i + 1} of {len(to_be_reviewed)}")
            print("----------------------")
            print(f"topics: {query.topics}")
            print(f"length: {len(query.text)}")
            print()
            print(query.text)
            print("---")

            choice = input("[a]ccept, [r]egenerate, use gpt-4[o], [c]ancel?\n")
            print()

            if choice == 'a':
                break
            elif choice == 'r':
                query = default_generator.query_from_topics(query.topics)
            elif choice == 'o':
                query = advanced_generator.query_from_topics(query.topics)
            elif choice == 'c':
                return queries

        queries[q_idx] = query
        _save_to_file(queries, export_path)

    print("Done reviewing.")

    return queries


def generate_queries(
        aspects: list[int],
        article_path: str,
        num_queries: int,
        num_attempts: int,
        manual_review: bool,
        num_fusion_queries: int,
        export_path: str,
) -> list[Query]:
    """
    Function to generate queries from a specified file.
    The function opens the specified file, and for each query description found in the file
    generates a query, and stores that query in the "text" field.
    If there are already queries present, the user will be prompted, whether to regenerate them.

    :param aspects: List of number of aspects that should be incorporated in the query generation.
    :type aspects: list[int]
    :param article_path: Path to the synthetic dataset, where the Wikipedia articles are stored.
    :type article_path: str
    :param num_queries: Number of queries to generate.
    :type num_queries: int
    :param num_attempts: Number of attempts per query to generate a functional query.
    :type num_attempts: int
    :param manual_review: Choice whether the queries will be manually reviewed.
    :type manual_review: bool
    :param num_queries: Number of fusion prompts to generate per query.
    :type num_queries: int,
    :param export_path: Path to the output file.
    :type export_path: str
    :param num_fusion_queries: Number of fusion queries to generate per standard query.
    :type num_fusion_queries: int
    :return: List of queries.
    :rtype: list[Query]
    """
    assert num_attempts >= 1
    assert len(aspects) > 0

    articles: list[Article] = load_articles(article_path)
    generator = QueryGenerator(QueryGenerator.Model.GPT_3_5_TURBO)

    try:
        queries: list[Query] = load_queries(export_path, articles)
    except (FileNotFoundError, json.JSONDecodeError):
        queries = []

    if queries:
        choice = input(f"Found {len(queries)} existing usable queries. Do you want to replace them? [y/N] ")
        if choice.strip().lower() in ('y', 'yes'):
            queries.clear()

    missing_queries: dict[int, int] = {}
    for k in aspects:
        existing_queries = sum(len(q.topics) == k for q in queries)
        if existing_queries < num_queries:
            missing_queries[k] = num_queries - existing_queries

    def generate_single(_topics: set[Article]) -> Query:
        _query = None
        for _ in range(num_attempts):
            _query = generator.query_from_topics(_topics)
            if _check_query(_query):
                break
        return _query

    if missing_queries:
        pbar = tqdm(total=sum(missing_queries.values()), desc="Generating queries")
        for k, n in missing_queries.items():
            for topics in _sample_query_topics(articles, k, n):
                query = generate_single(topics)
                queries.append(query)
                _save_to_file(queries, export_path)
                pbar.update(1)
    else:
        print(f"No new queries were added.")

    if manual_review:
        print("Reviewing generated queries...")
        queries = _review_queries(queries, generator, export_path)

    if num_fusion_queries > 0:
        print()
        for i, query in enumerate(tqdm(queries, desc="Adding fusion prompts")):
            if not isinstance(query, FusionQuery):
                queries[i] = generator.fusion_from_query(query, num_fusion_queries)
                _save_to_file(queries, export_path)

    queries.sort(key=lambda q: len(q.topics))

    print(f"Saving data to {export_path}...")
    _save_to_file(queries, export_path)

    return queries
