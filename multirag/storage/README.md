# Vector Database

## Set Up

The Postgres-based vector database is isolated inside a Docker container and can be started by issuing the following command:
```
multirag-cli db start
```

The vector database can be shut down with the command:
```
multirag-cli db stop
```


## Command Line Interfaces

You can use the `multirag-cli db` command to interact with our Postgres-based vector database, which comes with the following command line interface:
```
usage: multirag-cli db [-h] [-c [CONFIG]] [-m [METRIC]] {start,stop,clear,import} ...

Database

optional arguments:
  -h, --help            show this help message and exit
  -c [CONFIG], --config [CONFIG]
                        Path to the database Docker compose file. (default: config/docker-compose.yaml)
  -m [METRIC], --metric [METRIC]
                        Distance metric for the vector database, one of cosine, dot, manhattan, euclidean. (default: cosine)

db commands:
  {start,stop,clear,import}
```

The start command (`multirag-cli db start`) allows the user to start the vector database, so that can be used subsequently.
```
usage: multirag-cli db start [-h]

Start database Docker container

optional arguments:
  -h, --help  show this help message and exit
```

The stop command (`multirag-cli db stop`) allows the user to shutdown the vector database.
```
usage: multirag-cli db stop [-h]

stop database Docker container

optional arguments:
  -h, --help  show this help message and exit
```

The clear command (`multirag-cli db clear`) allows the user to remove all content from the vector database, so that a new embedding can be loaded into the database.
```
usage: multirag-cli db clear [-h]

Clear database

optional arguments:
  -h, --help  show this help message and exit
```

The import command (`multirag-cli db import`) allows the user to import the embeddings into the vector database from an input file.
```
usage: multirag-cli db import [-h] [-e [EMBEDDING_PATH]]

Import data into the database

optional arguments:
  -h, --help            show this help message and exit
  -e [EMBEDDING_PATH], --embedding-path [EMBEDDING_PATH]
                        Path to the embedding data file. (default: embeddings.json)
```
