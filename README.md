# pubmed-rag
A playground for RAG development from pubmed API (starting with bkg-review project)


## Getting Started

### Installation and Set up

1. Install [poetry](https://python-poetry.org/docs/#installation) on your device
2. Clone this repository
3. Set up environment by 
    - open terminal
    - `cd` into repo directory
    - running `poetry install`
4. activate the environment by running `poetry shell` in repo dir

## Retrieving the pubmed articles

run `python get_embeddings.py <path to config file>`
    e.g. `python get_embeddings.py demo/config.yaml`

## Putting the embeddings into a vector database
run `python create_db.py <path to config file>`
    e.g. `python create_db.py demo/config/yaml`

