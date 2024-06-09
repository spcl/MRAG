# Synthetic Dataset and Query Generation

The dataset module handles the generation of the synthetic Wikipedia article-based dataset as well as the generation of queries for that dataset.

## Synthetic Dataset Generation

You can use the `multirag-cli datagen` command to generate the synthetic Wikipedia-based dataset, which comes with the following command line interface:
```
usage: multirag-cli datagen [-h] [-c [CONFIG]] [-m [MIN_LENGTH]] [-n [NUM_CATEGORIES]] [-o [OUTPUT]] [-s [SAMPLES]]

Synthetic Dataset Generation

optional arguments:
  -h, --help            show this help message and exit
  -c [CONFIG], --config [CONFIG]
                        Path to the config file. (default: multirag/dataset/categories.json)
  -m [MIN_LENGTH], --min-length [MIN_LENGTH]
                        Minimum number of characters in each document. (default: 800)
  -n [NUM_CATEGORIES], --num-categories [NUM_CATEGORIES]
                        Number of categories. (default: 25)
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output file. (default: articles.json)
  -s [SAMPLES], --samples [SAMPLES]
                        Number of documents per category. (default: 50)
```

The synthetic dataset consists of the summaries of Wikipedia articles, that are grouped into categories.
The number of categories (default: 25) and documents per category (default: 50) can be specified by the user.
The number of categories specified as the command line argument should not exceed the number of categories defined in the configuration file.
We provide an example configuration file for the categories, that we used during the evaluation of MRAG, in `multirag/dataset/categories.json`.
We describe the format of the configuration file in the section [Configuration File Format](#configuration-file-format).
Additionally the user can set a minimum of characters that each document contains (default: 800) as well as the path to the output JSON file, where the synthetic dataset is stored.

You can find the synthetic Wikipedia article dataset, that we used during the evaluation of MRAG, in the [datasets](/datasets) directory.
Generating a synthetic dataset with the standard paramaters takes roughly 10 minutes.


### Configuration File Format

```JSON
[
    {
        "label": "countries",
        "starting_points": [
            "Member states of the United Nations"
        ],
        "explicit_category": "Member states of the United Nations"
    },
]
```
`label` is the name of the category. `starting_points` references the Wikipedia articles that serve as a staring point for the article discovery.
The `explicit_category` attribute helps with the scraping of Wikipedia articles. Every articles that contains `explicit_category` as a substring will be accepted as an article of the category.

### Dataset Output Format

```JSON
[
    {
        "title": "...",
        "text": "...",
        "label": "..."
    },
]
```

Each document in the output file of the synthetic dataset has three attributes: `title`, `text` and `label`.
The `title` attribute reflects the title of the respective Wikipedia article.
`text` contains the summary, the first section, of the Wikipedia articles and serves as the document during the following stages of the pipeline.
The value of that attribute contains at least the minimum number of characters, that were specified during the synthetic dataset generation (default: 800).
`label` specifies the category to which that `document` belongs to.

The `text` attribute is used during the subsequent embedding stage, while all three attributes are used during the query generation for the synthetic dataset.


## Synthetic Query Generation

You can use the `multirag-cli querygen` command to generate synthetic queries for the synthetic Wikipedia-based dataset, which comes with the following command line interface:
```
usage: multirag-cli querygen [-h] [-a ASPECTS [ASPECTS ...]] [-d [DATASET_PATH]] [-m] [-n [NUM_ATTEMPTS]] [-f [NUM_FUSION_QUERIES]] [-o [OUTPUT]] [-q [NUM_QUERIES]]

Synthetic Query Generation

optional arguments:
  -h, --help            show this help message and exit
  -a ASPECTS [ASPECTS ...], --aspects ASPECTS [ASPECTS ...]
                        List of aspect numbers to incorporate into the queries. (default: [1, 2, 3, 4, 5, 6, 10, 15, 20, 25])
  -d [DATASET_PATH], --dataset-path [DATASET_PATH]
                        Path to the dataset. (default: articles.json)
  -m, --manual-review   Enable manual review of failed queries. (default: False)
  -n [NUM_ATTEMPTS], --num-attempts [NUM_ATTEMPTS]
                        Number of attempts to retry the generation of a query that previously failed. (default: 2)
  -f [NUM_FUSION_QUERIES], --num-fusion-queries [NUM_FUSION_QUERIES]
                        Number of fusion queries to generate per standard query. (default: 4)
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output file. (default: queries.json)
  -q [NUM_QUERIES], --num-queries [NUM_QUERIES]
                        Number of queries to generate. (default: 25)
```
The synthetic query generation is based on stories, which cover the various aspects of a query.
The number of queries (default: 25) as well as the number of aspects per query can be specified by the user.
The number of aspects is a list, where for each specific number of aspects that number of queries is generated.
So the total number of generated queries is $length(\text{aspect list}) * \text{number of queries}$.
The path to the synthetic dataset can be specified with the `--dataset-path` parameter (default: `articles.json`).
The format for input file can be found in the [output format](#dataset-output-format) section of the synthetic dataset generation.
It is possible to manually review the repeated query generation in case the first attempt fails.
The number of attempts to generate a specific query can be specified with the `--num-attempts` parameter (default: 2).
The user can also control the number of prompts for the fusion queries (default 4). If the user specifies zero prompts, no fusion queries are generated.
Additionally the user can state the path to the output JSON file, where the synthetic queries are stored.

The query generator uses the OpenAI API for the query generation, which will cost you.
Please export your OpenAI API key using `export OPENAI_API_KEY=<api key>` before using the synthetic query generator.

You can find the synthetic queries that we used with synthetic Wikipedia article dataset for the evaluation fo MRAG in the [datasets](/datasets) directory.
Generating synthetic queries with the standard paramaters takes roughly 30 minutes.


### Fusion Queries

Fusion queries are based on [RAG Fusion](https://arxiv.org/abs/2402.03367) and are an optional mechanism that is used to further enhance the benefits of MRAG at the cost of additional tokens.
A fusion query uses an LLM to create a fixed number of questions about the standard query.
Each question is separately applied through an embedding model.


### Query Output Format

```
[
    {
        "topics": ["...", "..." ...],
        "text: "...",
        "fusion": ["...", "...", ...]
    }
]
```
Each query has at least two attributes: `topics` and `text`.
The `text` attribute contains the actual query text.
`topics` contains the article titles that were used during the generation of the query.
`topics` is used during the evaluation to check, whether all necessary articles were fetched.
The attribute `fusion` is used only for fusion queries and contains the prompts, which are used to query an LLM to improve the results.


### Example Prompts
The prompts were formatted for better readability.
The following example shows the prompt for a query generation with two aspects.
```
{
    "role": "system",
    "content":
        "You are a very curious scholar asking about possible connections between different topics."
},
{
    "role": "user",
    "content":
        "Please create a story about the attached 2 articles on the topics Impostor syndrome, Alhambra
        (board game). It is very important that each of the attached articles is relevant to the story,
        in a way that references the content of the article, not just its title. But please also
        mention each title at least once. Please make sure that all of the attached articles are
        relevant to your story, and that each article is referenced in at least two sentences! They do
        not necessarily have to be referenced in the same order, but make sure no article is forgotten.
        Important: Output only the story, no additional text. And do not use bullet points, or
        paragraphs.

        Articles:
        ---------
        Impostor syndrome:
        Impostor syndrome, also known as impostor phenomenon or impostorism, is a psychological
        occurrence. Those who have it may doubt their skills, talents, or accomplishments. They may
        have a persistent internalized fear of being exposed as frauds. Despite external evidence of
        their competence, those experiencing this phenomenon do not believe they deserve their success
        or luck. They may think that they are deceiving others because they feel as if they are not as
        intelligent as they outwardly portray themselves to be.
        Impostor syndrome can stem from and result in strained personal relationships and can hinder
        people from achieving their full potential in their fields of interest. The term
        \"impostorization\" shifts the source of the phenomenon away from the supposed impostor to
        institutions whose policies, practices, or workplace cultures \"either make or intend to make
        individuals question their intelligence, competence, and sense of belonging.\"
        ---------
        Alhambra (board game):
        Alhambra (German: Der Palast von Alhambra, literally \"The Palace of Alhambra\") is a 2003
        tile-based German-style board game designed by Dirk Henn. It was originally published in
        Germany by Queen Games in a language-interdependent version; an English-specific version was
        released in North America by the now-defunct \u00dcberplay. The game is a Muslim-themed
        update, set during the construction of the Alhambra palace in 14th century Granada, of the
        1998 stock trading board game Stimmt So!, which in turn was an update of the 1992 mafia
        influence board game Al Capone; the original version was subsequently released as Alhambra:
        The Card Game. Upon its release, Alhambra won numerous awards, including the Spiel des Jahres
        award. Its success has led to the release of numerous expansion packs and spin-off games, and
        is becoming Queen Games' flagship franchise.
        ---------
        Again, make sure that you reference all the following topics in your story: Impostor syndrome,
        Alhambra (board game)"
}
```
The following example shows an example prompt for the generation of fusion questions for the same query that was generated with the prompt above.
```
{
    "role": "system",
    "content":
        "You are a helpful assistant that generates multiple search queries based on a single input
        query."
},
{
    "role": "user",
    "content":
        "Generate multiple search queries related to: Alice sat at the bustling board game caf\u00e9,
        her hands trembling as she arranged the tiles in her Alhambra game. Despite her keen strategic
        mind, doubt gnawed at her heart. The pervasive feeling of impostor syndrome crept in,
        whispering that her victories were mere luck, not skill. She felt like an outsider in the
        world of competitive board gaming, always fearing being exposed as a fraud.

        As she pondered her next move, a fellow player, Sarah, noticed Alice's hesitance. Sensing her
        unease, Sarah offered a kind smile and words of encouragement. \"Don't underestimate yourself,
        Alice. Your talent shines through in every game we play together,\" Sarah reassured her. Those
        words struck a chord deep within Alice, a glimmer of belief piercing through the veil of
        self-doubt.

        Emboldened by Sarah's support, Alice made bold decisions in the game, surprising even herself
        with her strategic prowess. With each successful move, the shadows of impostor syndrome faded,
        replaced by a newfound confidence. As the game drew to a close, Alice emerged victorious, her
        skills undeniable to all who witnessed her triumph.

        The Alhambra board game became a symbolic battleground where Alice confronted her inner
        impostor, emerging stronger and more self-assured than ever before. In that moment of victory,
        she realized that true success is not about luck but about belief in one's abilities, a lesson
        that transcended the boundaries of the game and resonated in her journey towards overcoming
        impostor syndrome."
},
{
    "role": "user",
    "content": "OUTPUT (4 queries):"
}
```

### Example Queries
The queries were formatted for better readability.
The following example query was generated using the prompt in section [Example Prompts](#example-prompts).

```
Alice sat at the bustling board game caf\u00e9, her hands trembling as she arranged the tiles in her
Alhambra game. Despite her keen strategic mind, doubt gnawed at her heart. The pervasive feeling of
impostor syndrome crept in, whispering that her victories were mere luck, not skill. She felt like
an outsider in the world of competitive board gaming, always fearing being exposed as a fraud.

As she pondered her next move, a fellow player, Sarah, noticed Alice's hesitance. Sensing her unease,
Sarah offered a kind smile and words of encouragement. \"Don't underestimate yourself, Alice. Your
talent shines through in every game we play together,\" Sarah reassured her. Those words struck a
chord deep within Alice, a glimmer of belief piercing through the veil of self-doubt.

Emboldened by Sarah's support, Alice made bold decisions in the game, surprising even herself with
her strategic prowess. With each successful move, the shadows of impostor syndrome faded, replaced by
a newfound confidence. As the game drew to a close, Alice emerged victorious, her skills undeniable
to all who witnessed her triumph. The Alhambra board game became a symbolic battleground where Alice
confronted her inner impostor, emerging stronger and more self-assured than ever before. In that
moment of victory, she realized that true success is not about luck but about belief in one's
abilities, a lesson that transcended the boundaries of the game and resonated in her journey towards
overcoming impostor syndrome.
```
The following example questions are intended for the fusion approach of the query above.
```
1. Strategies for overcoming impostor syndrome in competitive environments
2. Impact of supportive mentors in addressing feelings of self-doubt and impostor syndrome
3. Psychological effects of impostor syndrome on performance in board games
4. Alhambra board game tips for building confidence and strategic thinking
```
