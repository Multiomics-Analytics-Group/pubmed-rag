# import 
import argparse
import pandas as pd
from ast import literal_eval
import os
from pymilvus import MilvusClient
from pubmed_rag.utils import (
    get_basename,
    generate_log_filename,
    init_log,
    config_loader,
    assert_nonempty_keys,
    assert_nonempty_vals,
)

if __name__ == "__main__":

    ## GET ARGS
    # init
    parser = argparse.ArgumentParser(
        prog='get sentence embeddings',
    )
    parser.add_argument(
        '-c', '--config', 
        action='store',
        default='demo/config.yaml'
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
    pmid_path = config['pmid file path']
    output_path = config['biocjson output path']
    max_tokens = config['max_tokens']
    chosen_model = config['transformer_model']
    db_name = config['db name']
    out_dim = config['output_dimensions']
    metric = config['metric_type']
    logger.info(f'Configuration: {config}')

    ## MAIN
    logger.info(f"Creating {db_name}")

    client = MilvusClient(f'{db_name}')
    if client.has_collection(collection_name="main"):#TODO add to config?
        client.drop_collection(collection_name="main")
    client.create_collection(
        collection_name="main",
        dimension=out_dim, 
        metric_type=metric
    )

    logger.info(f'Reading in embeddings from folder: {output_path} ...')
    df = pd.read_csv(
        os.path.join(output_path, 'all_embeddings.csv'), 
        sep='\t'
        )
    # rename id and vector cols
    df = df.rename(
        columns={
            'Unnamed: 0':'id',
            'embedding':'vector'
        }, 
    )
    # ensure vector is a list
    df['vector'] = df['vector'].apply(literal_eval)
    # to list of dicts
    data = df.to_dict(orient='records')
    logger.info(f"Number of embeddings is: {len(data)}")

    # put data into db
    result = client.insert(collection_name="main", data=data)
    logger.info(f"Inserted data into db: {result}")

    logger.info('Complete.')

else:
    print('create_db.py imported. Script not ran.')
