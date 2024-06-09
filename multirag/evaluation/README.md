# Evaluation

You can use the `multirag-cli evaluate` command to generate the synthetic Wikipedia-based dataset, which comes with the following command line interface:
```
usage: multirag-cli evaluate [-h] [-e [EMBEDDING_PATH]] [-l [LAYER]] [-o [OUTPUT]] [-p [PICKS]] [-c [CONFIG]] [-m [METRIC]]

Evaluation

optional arguments:
  -h, --help            show this help message and exit
  -e [EMBEDDING_PATH], --embedding-path [EMBEDDING_PATH]
                        Path to the embedding file. (default: embeddings.json)
  -l [LAYER], --layer [LAYER]
                        Layer to evaluate. (default: 31)
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output file. (default: test-results.json)
  -p [PICKS], --picks [PICKS]
                        Number of picks. (default: 32)
  -c [CONFIG], --config [CONFIG]
                        Path to the database Docker compose file. (default: config/docker-compose.yaml)
  -m [METRIC], --metric [METRIC]
                        Distance metric for the vector database, one of cosine, dot, manhattan, euclidean. (default: cosine)
```


## Strategies
The file `evaluate.py` contains several different retrieval strategies, as well as an abstract class for defining your own strategies. We now describe the already implemented strategies.

### Standard
The class `StandardStrategy` implements the _"standard"_ retrieval strategy. It retrieves documents for a given prompt by ordering them based on their distance to the prompt in the standard embedding space, and selecting the documents closest to the prompt.

### Multi-Head RAG
The class `MultiHeadStrategy` implements the _"Multi-Head RAG"_ retrieval strategy, which we fully explain in the paper. We now briefly describe the necessary steps for this retrieval strategy:
1. For each attention head's embedding space, perform a similarity search.
2. Assign scores to the documents for each attention head, based on the head itself, the document's similarity ranking, and the distance between the document's and the prompt's embedding.
3. Accumulate the score for each document across all attention heads.
4. Retrieve the documents with the highest cumulative scores.

### Split-RAG
The class `SplitStrategy` implements the _"Split-RAG"_ retrieval strategy.
It works identical to _Multi-Head RAG_, but uses the segments of the standard embedding instead of the attention embeddings.

### RAG-Fusion
The class `FusionStrategy` implements Zackary Rackauckas' [RAG-Fusion](https://arxiv.org/abs/2402.03367) retrieval strategy.
Instead of directly performing standard retrieval based on a prompt, it lets an LLM generate a number of questions about the prompt in question, and performs retrieval for those questions.
The results of these retrievals are then combined through reciprocal rank fusion to produce the final selection of documents to return.

### MRAG-Fusion
The class `MultiHeadFusionStrategy` implements the _"MRAG-Fusion"_ retrieval strategy.
It is a blend of _Multi-Head RAG_ and _RAG-Fusion_ that uses the same techniques as _RAG-Fusion_, but replaces the standard embedding-based retrievals with retrievals through _Multi-Head RAG_.
