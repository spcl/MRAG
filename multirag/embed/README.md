# Embedding Generation

You can use the `multirag-cli embed` command to generate embeddings for the dataset and the queries with the following command line interface:
```
usage: multirag-cli embed [-h] [-d [DOCUMENT_PATH]] [-l LAYERS [LAYERS ...]] [-o [OUTPUT]] [-q [QUERY_PATH]]

Embedding Generation

optional arguments:
  -h, --help            show this help message and exit
  -d [DOCUMENT_PATH], --document-path [DOCUMENT_PATH]
                        Path to the dataset. (default: articles.json)
  -l LAYERS [LAYERS ...], --layers LAYERS [LAYERS ...]
                        Layers to target for the attention heads. (default: [31])
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output file. (default: embeddings.json)
  -q [QUERY_PATH], --query-path [QUERY_PATH]
                        Path to the queries. (default: queries.json)
```
