CREATE EXTENSION IF NOT EXISTS vector;

-- table for standard embedding
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT UNIQUE,
    content TEXT,
    label TEXT
);

-- table for standard embedding
CREATE TABLE IF NOT EXISTS standard (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES articles (id) UNIQUE,
    embedding VECTOR(4096)
);

-- table for attention embeddings:
-- each attention head's embedding space is one column
CREATE TABLE IF NOT EXISTS attention (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES articles (id),
    layer_index INT,
    head00 vector(128), head01 vector(128), head02 vector(128), head03 vector(128),
    head04 vector(128), head05 vector(128), head06 vector(128), head07 vector(128),
    head08 vector(128), head09 vector(128), head10 vector(128), head11 vector(128),
    head12 vector(128), head13 vector(128), head14 vector(128), head15 vector(128),
    head16 vector(128), head17 vector(128), head18 vector(128), head19 vector(128),
    head20 vector(128), head21 vector(128), head22 vector(128), head23 vector(128),
    head24 vector(128), head25 vector(128), head26 vector(128), head27 vector(128),
    head28 vector(128), head29 vector(128), head30 vector(128), head31 vector(128),
    UNIQUE (article_id, layer_index)
);

-- table to store the scales
CREATE TABLE IF NOT EXISTS attention_scales (
    id SERIAL PRIMARY KEY,
    scales VECTOR(32),
    layer_index INT UNIQUE
);

-- table for cut-standard embeddings:
-- each segment's embedding space is one column
CREATE TABLE IF NOT EXISTS cut_standard (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES articles (id) UNIQUE,
    segment00 vector(128), segment01 vector(128), segment02 vector(128), segment03 vector(128),
    segment04 vector(128), segment05 vector(128), segment06 vector(128), segment07 vector(128),
    segment08 vector(128), segment09 vector(128), segment10 vector(128), segment11 vector(128),
    segment12 vector(128), segment13 vector(128), segment14 vector(128), segment15 vector(128),
    segment16 vector(128), segment17 vector(128), segment18 vector(128), segment19 vector(128),
    segment20 vector(128), segment21 vector(128), segment22 vector(128), segment23 vector(128),
    segment24 vector(128), segment25 vector(128), segment26 vector(128), segment27 vector(128),
    segment28 vector(128), segment29 vector(128), segment30 vector(128), segment31 vector(128)
);

-- table to store the scales: should only contain one entry
CREATE TABLE IF NOT EXISTS cut_standard_scales (
    id SERIAL PRIMARY KEY,
    scales VECTOR(32)
);
