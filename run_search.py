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
    log_suffix = get_basename(args.config)
    # generate log file name
    log_file = generate_log_filename(suffix=log_suffix)
    # init logger
    logger = init_log(log_file, display=True)
    # log it
    logger.info(f'Path to log file: {log_file}')

    ## LOAD CONFIG PARAMETERS
    # getting the config filepath
    config_filepath = args.config
    # log it
    logger.info(f'Path to config file: {config_filepath}')
    # load config params
    logger.info("Loading config params ... ")
    config = config_loader(config_filepath)
    assert_nonempty_keys(config)
    assert_nonempty_vals(config)
    output_path = config['biocjson output path']
    chosen_model = config['transformer_model']
    db_name = config['db name']
    n_results = config['n_results']
    logger.info(f'Configuration: {config}')

    ## MAIN
    logger.info(f"Accessing {db_name}")
    client = MilvusClient(db_name)

    logger.info(f"Loading model {chosen_model} from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    model = AutoModel.from_pretrained(chosen_model)

    # GET EMBEDDINGS
    logger.info(f"Embedding query: {args.query}")
    # Tokenize sentences
    encoded_input = get_tokens(
        tokenizer,
        [args.query]
    )
    # get embeddings
    sentence_embeddings = get_sentence_embeddings(
        model,
        encoded_input
    )           

    ## RUN QUERY
    logger.info('Searching against vector database')
    result = client.search(
        collection_name="main",
        data=sentence_embeddings.detach().numpy(),
        #filter="section == 'abstract'",
        limit=n_results,
        output_fields=["text", "pmid", 'section'],
    )

    logger.info(f"Query: {args.query}")
    logger.info(f"Results:")
    for x in (result[0]):
        logger.info(json.dumps(x, indent=4))

    logger.info('Complete.')

else:
    print('run_query.py imported. Script not ran.')
