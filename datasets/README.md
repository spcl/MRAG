## Datasets

### Wikipedia Articles

`wikipedia_articles_v1.json.bz2` contains the compressed JSON file, that we used as the synthetic Wikipedia article dataset during the evaluation of MRAG.
It was generated using our synthetic dataset generator: `multirag datagen`.
It consists of 25 categories with 50 distinct documents in each category.
The documents are based on the summary of Wikipedia articles and each document contains at least 800 characters.

Please uncompess `wikipedia_articles_v1.json.bz2` using `bunzip2` before using it.

For the format of the JSON file see the respective [section](../multirag/dataset/README.md#dataset-output-format) in the dataset module documention.

We also provide an extended version of the Wikipedia dataset in `wikipedia_articles_v2.json.bz2`, which contains 80 categories with 50 documents in each category for a total of 4000 documents.


### Legal Documents

`legal_documents_v1.tar.bz2` contains the legal documents dataset used during the evaluation of MRAG.
It consists of 25 categories with 25 topics.
Each topic is stored in an individual JSON file and contains 10 documents.
The dataset consists of 6,250 documents in total.
We provide an improved version of the dataset as `legal_documents_v2.tar.bz2` with longer document summaries.


### Chemical Plant Accident Dataset

`chem_documents_v1.tar.bz2` contains the dataset created for the analysis of causes of chemical plant accidents.
Similar to the legal document datasets, this dataset consists of 25 categories with 25 topics, where each topic has 10 documents and is stored in a single JSON file.
The dataset consists of 6,250 documents in total.
We also provide an improved version of the dataset as `chem_documents_v2.tar.bz2` with longer document summaries.


## Queries

### Wikipedia Article Queries

`wikipedia_queries_v1.json.bz2` contains the compressed JSON file, that stores the queries that were used to query the synthetic Wikipedia article dataset during the evaluation of MRAG.
It was generated using our synthetic query generator: `multirag querygen`.
It contains 25 queries with 1, 2, 3, 4, 5, 6, 10, 15, 20 and 25 aspects respectively, so a total of 250 queries.

Please uncompess `wikipedia_queries_v1.json.bz2` using `bunzip2` before using it.

For the format of the JSON file see the respective [section](../multirag/dataset/README.md#query-output-format) in the dataset module documention.

We also provide queries for the extended version of the Wikipedia dataset in `wikipedia_queries_v2.json.bz2`.

Please use the documents and queries of the same version if you want to recreate our results.


### Legal Documents

`legal_queries.tar.bz2`  contains the compressed JSON file, that stores the queries that were used to query legal document dataset during the evaluation of MRAG.


### Chemical Plant Accident Queries

`chem_queries.tar.bz2`  contains the compressed JSON file, that stores the queries that were used to query legal document dataset during the evaluation of MRAG.
