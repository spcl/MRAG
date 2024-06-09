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
import os
import random
import itertools

from tqdm import tqdm
from typing import Any, Optional
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

from wikipediaapi import WikipediaPage, Wikipedia
from requests import ReadTimeout, ConnectionError, JSONDecodeError


@dataclass(frozen=True)
class CategoryConfig:
    """
    Data class to represent the different aspects of a synthetic dataset category.

    label is the title of the category
    starting points are the Wikipedia (list) articles that are used as the starting point
    for sampling articles for the category.
    category_pattern, explicit_category and title_prefix are helper variables for matching
    articles to a category during the scraping process.
    """
    label: str
    starting_points: list[str]
    category_pattern: Optional[str] = None
    explicit_category: Optional[str] = None
    title_prefix: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CategoryConfig':
        return cls(**config)

    def __hash__(self):
        return hash(self.label)


@dataclass(frozen=True)
class Article:
    """
    Data class to represent the different aspects of a Wikipedia article.

    title represents the title of the article.
    text contains the summary, the first section, of an article.
    label states the category to which the article belongs to.
    """
    title: str
    text: str
    label: str

    @classmethod
    def from_dict(cls, article: dict[str, str]) -> 'Article':
        return cls(**article)

    def __repr__(self):
        return f'<Article {self.title}>'


class MultiRagWiki(Wikipedia):
    """
    Class to handle the Wikipedia access.

    Inherits from the Wikipedia class.
    """
    def __init__(self) -> None:
        project_name = "Foobar"
        project_url = "https://foo.com/bar"
        contact_email = "foo@bar.org"
        user_agent = f"{project_name} ({project_url}; {contact_email})"
        super().__init__(user_agent)


def load_articles(file_path: str) -> list[Article]:
    """
    Load articles from input JSON file.

    :param file_path: Path to the input JSON file.
    :type file_path: str
    :return: List of articles.
    :rtype: list[Article]
    """
    with open(file_path, 'r') as file:
        json_data: list[dict[str, str]] = json.load(file)

    return list(map(Article.from_dict, json_data))


def _match_page(page: WikipediaPage, config: CategoryConfig) -> bool:
    """
    Check if the provided page fulfills the conditions specified in the config.

    :param page: Linked Wikipedia page to check for matches.
    :type page: WikipediaPage
    :param config: Category config to match against.
    :type config: CategoryConfig
    :return: True if matches, False otherwise
    :rtype: bool
    """
    # is it a list?
    if page.title.startswith("List "):
        return False

    # check if pattern appears in any category
    if config.category_pattern is not None:
        for category in page.categories:
            if config.category_pattern.lower() in category.lower():
                return True

    # check if explicit category appears
    if config.explicit_category is not None:
        if f"Category:{config.explicit_category}" in page.categories:
            return True

    # check if title begins with specified prefix
    if config.title_prefix is not None:
        if page.title.lower().startswith(config.title_prefix.lower()):
            return True

    return False


def _fetch_articles_for_group(
        pages: list[WikipediaPage],
        config: CategoryConfig,
        sample_size: int,
        min_length: int,
        pbar: tqdm = None,
        retries: int = 3
) -> set[Article]:
    """
    For a given category, retrieve a random set of `sample_size` articles matching
    the criteria specified in `config` and a minimum length of `min_length`.

    :param pages: List of Wikipedia pages.
    :type pages: list[WikipediaPage]
    :param config: Category config to match against.
    :type config: CategoryConfig
    :param sample_size: Number of articles to sample.
    :type sample_size: int
    :param min_length: Minimum number of characters per article.
    :type min_length: int
    :param pbar: Optional tqdm progress bar to be incremented `sample_size` times.
    :type pbar: tqdm
    :param retries: Maximum number of retries after encountering :exception:`HTTPError`. Defaults to 3.
    :type retries: int
    :return: Set of articles referenced from `pages` matching the criteria.
    :rtype: set[Article]
    """
    if sample_size < 1:
        return set()

    candidates: list[WikipediaPage] = []
    for page in pages:
        for linked_page in page.links.values():
            if page.title != linked_page.title:
                candidates.append(linked_page)

    random.shuffle(candidates)
    selected: set[Article] = set()

    for linked_page in candidates:
        rem_retries = retries
        is_match = False

        while rem_retries > 0:
            try:
                is_match = _match_page(linked_page, config)
                is_match &= len(linked_page.summary) >= min_length
                rem_retries = 0
            except (ReadTimeout, ConnectionError, JSONDecodeError):
                rem_retries -= 1

        if not is_match:
            # doesn't fit matching criteria for this group
            continue

        article = Article(linked_page.title, linked_page.summary, config.label)
        if article in selected:
            # duplicate of previous article
            continue

        if pbar is not None:
            pbar.update(1)

        selected.add(article)
        if len(selected) == sample_size:
            return selected

    raise Exception(f"only {len(selected)} articles of desired length for label {config.label}")


def fetch_articles(
        config_path: str,
        num_categories: int,
        samples_per_category: int,
        min_article_length: int,
        export_path: str
) -> list[Article]:
    """
    Retrieve summaries of Wikipedia articles in each category.

    :param config_path: Path to the configuration file, that contains the category information.
    :type config_path: str
    :param num_categories: Number of categories for the synthetic dataset.
    :type num_categories: int
    :param samples_per_category: Number of documents in each category of the synthetic dataset.
    :type samples_per_category: int
    :param min_article_length: Minimum number of characters in the each document.
    :type min_article_length: int
    :param export_path: Path to the output file.
    :type export_path: str
    :return: List of documents (including title and category).
    :rtype: list[Article]
    """
    with open(config_path, 'r') as file:
        config_dicts: list[dict[str, Any]] = json.load(file)

    if num_categories > len(config_dicts):
        raise Exception(f"Config file {config_path} only contains {len(config_dicts)} categories, but {num_categories} were requested")
    config_dicts = config_dicts[:num_categories]

    wiki: Wikipedia = MultiRagWiki()
    group_pages: dict[CategoryConfig, list[WikipediaPage]] = {}

    for config_dict in config_dicts:
        config: CategoryConfig = CategoryConfig.from_dict(config_dict)
        group_pages[config] = list(map(wiki.page, config.starting_points))

    num_groups = len(group_pages)

    print(f"Retrieving {samples_per_category} samples for {num_groups} groups...")
    pbar = tqdm(total=num_groups * samples_per_category)

    def fetch_routine(_config: CategoryConfig) -> set[Article]:
        pages: list[WikipediaPage] = group_pages[_config]
        return _fetch_articles_for_group(pages, _config, samples_per_category, min_article_length, pbar)

    with ThreadPool() as pool:
        configs: list[CategoryConfig] = list(group_pages.keys())
        articles: list[set[Article]] = pool.map(fetch_routine, configs)

    all_articles: list[Article] = list(itertools.chain(*articles))
    if export_path is None:
        return all_articles

    print(f"Saving data in {export_path}...")

    if export_dir := os.path.dirname(export_path):
        # os.makedirs does not like an empty path
        os.makedirs(export_dir, exist_ok=True)
    with open(export_path, 'w',) as file:
        json.dump(all_articles, file, indent=4, default=lambda o: o.__dict__)

    return all_articles
