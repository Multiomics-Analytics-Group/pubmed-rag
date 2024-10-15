# pubmed-rag
A playground for RAG development from pubmed API (starting with bkg-review project)

Work in progress..

## 1. Getting Started

### Installation and Set up

1. Install [poetry](https://python-poetry.org/docs/#installation) on your device
2. Clone this repository
3. Set up environment by 
    - open terminal
    - `cd` into repo directory
    - running `poetry install`
4. activate the environment by running `poetry shell` in repo dir
5. download ollama and set up account

## 2. Creating the Vector Database

#### 2.a) Retrieving the pubmed articles

Using get_embeddings.py
`python get_embeddings.py --config <path to config file>`

Args:
- `--config` or `-c <path to config file>` 
- `--files_downloaded` or `-fd <path to folder with biocjson files>`

Examples:
- `python get_embeddings.py -c demo/config.yaml -fd biocjson`
- `python get_embeddings.py --config config.yaml`

#### 2.b) Putting the embeddings into a vector database

Using create_db.py
`python create_db.py --config <path to config file>`

Args:
- `--config` or `-c <path to config file>` 

Examples:
- `python create_db.py -c demo/config.yaml`
- `python create_db.py --config /users/blah/what/config.yaml`

## Querying the database

#### Find nearest vectors

Using run_search.py
`python run_search.py --config <path to config file> --query <the LLM prompt>`

Args:
- `--config` or `-c <path to config file>` 
- `--query` or `--q <text to embed>` 

Examples:
- `python run_search.py -c demo/config.yaml -q "Can you please tell me what nodes and edges I should include in a biological knowledge graph for drug repurposing?"`
- `python run_search.py --config demo/config.yaml --query "Best databases to use for a knowledge graph for biological question answering?"`

#### RAG model

Using use_rag.py
`python use_rag.py --config <path to config file> --query <the LLM prompt>`

Args:
- `--config` or `-c <path to config file>` 
- `--query` or `--q <text to embed>` 

Examples:
- `python use_rag.py -c demo/config.yaml -q "Can you please tell me what nodes and edges I should include in a biological knowledge graph for drug repurposing?"`
- `python use_rag.py --config demo/config.yaml --query "Best databases to use for a knowledge graph for biological question answering?"`