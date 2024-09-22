# import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import time, os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
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

from pubmed_rag.bioc import (
    collapse_sections,
    get_smaller_texts, 
    get_biocjson,
    passages_to_df, 
    mean_pooling
)

if __name__ == "__main__":

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
    config_filepath = get_config_path()
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
    logger.info(f'Configuration: {config}')

    ## MAIN
    logger.info(f"Creating {db_name} in current dir {os.getcwd}")

    client = MilvusClient(f'{db_name}.db')
    if client.has_collection(collection_name="main"):
        client.drop_collection(collection_name="main")
    client.create_collection(
        collection_name="main",
        dimension=768,  # The vectors we will use in this demo has 768 dimensions
    )

    logger.info(f'Reading in embeddings from folder: {output_path} ...')
    df = pd.read_csv(os.path.join(output_path, 'all_embeddings.csv'), )
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
    print('create_db.py imported. scholarly queries not initiated.')
