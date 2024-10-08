# import 
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from pubmed_rag.utils import (
    get_basename,
    generate_log_filename,
    init_log,
    get_config_path,
    config_loader,
    assert_nonempty_keys,
    assert_nonempty_vals,
)

from pubmed_rag.model import (
    get_tokens,
    get_sentence_embeddings
)

def main(
        path_to_config:str,
        query:str,
        logging:bool=False
):
    # allow logging if ran as script
    if logging:
        verbose = logger.info
    else:
        verbose = print

    ## LOAD CONFIG PARAMETERS
    # getting the config filepath
    config_filepath = path_to_config
    # log it
    verbose(f'Path to config file: {config_filepath}')
    # load config params
    verbose("Loading config params ... ")
    config = config_loader(config_filepath)
    assert_nonempty_keys(config)
    assert_nonempty_vals(config)
    output_path = config['biocjson output path']
    chosen_model = config['transformer_model']
    db_name = config['db name']
    n_results = config['n_results']
    verbose(f'Configuration: {config}')

    ## MAIN
    verbose(f"Accessing {db_name}")
    client = MilvusClient(db_name)

    verbose(f"Loading model {chosen_model} from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    model = AutoModel.from_pretrained(chosen_model)

    # GET EMBEDDINGS
    verbose(f"Embedding query: {query}")
    # Tokenize sentences
    encoded_input = get_tokens(
        tokenizer,
        [query]
    )
    # get embeddings
    sentence_embeddings = get_sentence_embeddings(
        model,
        encoded_input
    )           

    ## RUN QUERY
    verbose('Searching against vector database')
    result = client.search(
        collection_name="main",
        data=sentence_embeddings.detach().numpy(),
        #filter="section == 'abstract'",
        limit=n_results,
        output_fields=["text", "pmid", 'section'],
    )

    return result[0]

if __name__ == "__main__":

    ## GET ARGS
    # init
    parser = argparse.ArgumentParser(
        prog='run_search',
        description='gets config path and query'
    )
    parser.add_argument(
        '-c', '--config', 
        action='store',
        default='demo/config.yaml'
        )
    parser.add_argument(
        '-q', '--query',
        action='store',
    )
    args = parser.parse_args()

    ## START LOG FILE 
    # get log suffix, which will be the current script's base file name
    log_suffix = get_basename()
    # generate log file name
    log_file = generate_log_filename(suffix=log_suffix)
    # init logger
    logger = init_log(log_file, display=True)
    # log it
    logger.info(f'Path to log file: {log_file}')

    result = main(
        path_to_config=args.config,
        query=args.query,
        logging=True, 
    )

    logger.info(f"Query: {args.query}")
    logger.info(f"Results:")
    for x in (result):
        logger.info(json.dumps(x, indent=4))

    logger.info('Complete.')

else:
    print('Warning: run_query.py imported as a module. Script not ran.')

