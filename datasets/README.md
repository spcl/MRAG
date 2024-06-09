## Datasets

### Wikipedia Articles

`wikipedia_articles.json.bz2` contains the compressed JSON file, that we used as the synthetic Wikipedia article dataset during the evaluation of MRAG.
It was generated using our synthetic dataset generator: `multirag datagen`.
It consists of 25 categories with 50 distinct documents in each category.
The documents are based on the summary of Wikipedia articles and each document contains at least 800 characters.

Please uncompess `wikipedia_articles.json.bz2` using `bunzip2` before using it.

For the format of the JSON file see the respective [section](../multirag/dataset/README.md#dataset-output-format) in the dataset module documention.

## Queries

### Wikipedia Article Queries

`wikipedia_queries.json.bz2` contains the compressed JSON file, that stores the queries that were used to query the synthetic Wikipedia article dataset during the evaluation of MRAG.
It was generated using our synthetic query generator: `multirag querygen`.
It contains 25 queries with 1, 2, 3, 4, 5, 6, 10, 15, 20 and 25 aspects respectively, so a total of 250 queries.

Please uncompess `wikipedia_queries.json.bz2` using `bunzip2` before using it.

For the format of the JSON file see the respective [section](../multirag/dataset/README.md#query-output-format) in the dataset module documention.
