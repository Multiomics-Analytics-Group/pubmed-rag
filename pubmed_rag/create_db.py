# import
import os
from ast import literal_eval

import pandas as pd
from pymilvus import MilvusClient

from pubmed_rag.helpers.utils import (
    assert_nonempty_keys,
    assert_nonempty_vals,
    config_loader,
    get_args,
    get_logger,
)

if __name__ == "__main__":
    ## GET ARGS
    # init
    args = get_args(
        prog_name="create_vector_db",
        others=dict(description="puts embeddings into vector database"),
    )

    ## START LOG FILE
    logger = get_logger()
    logger.info(f"Arguments: {args}")

    ## LOAD CONFIG PARAMETERS
    # getting the config filepath
    config_filepath = args.config
    # log it
    logger.info(f"Path to config file: {config_filepath}")
    # load config params
    logger.info("Loading config params ... ")
    config = config_loader(config_filepath)
    assert_nonempty_keys(config)
    assert_nonempty_vals(config)
    pmid_path = config["pmid file path"]
    output_path = config["biocjson output path"]
    chosen_model = config["transformer_model"]
    db_name = config["db name"]
    col_name = config["collection name"]
    out_dim = config["output_dimensions"]
    metric = config["metric_type"]
    logger.info(f"Configuration: {config}")

    ## MAIN
    logger.info(f"Creating {db_name}")

    # access or create database
    client = MilvusClient(f"{db_name}")
    # overwrite the collection in the database of exists
    if client.has_collection(collection_name=col_name):
        client.drop_collection(collection_name=col_name)
    # put in the data
    client.create_collection(
        collection_name=col_name, dimension=out_dim, metric_type=metric
    )

    logger.info(f"Reading in embeddings from folder: {output_path} ...")
    # read in embeddings from get_embeddings.py
    df = pd.read_csv(os.path.join(output_path, "all_embeddings.csv"), sep="\t") #TODO put naming in config?
    # rename id and vector cols
    df = df.rename(
        columns={"Unnamed: 0": "id", "embedding": "vector"},
    )
    # ensure vector is a list
    df["vector"] = df["vector"].apply(literal_eval)
    # to list of dicts
    data = df.to_dict(orient="records")
    logger.info(f"Number of embeddings is: {len(data)}")

    # put data into db
    result = client.insert(collection_name=col_name, data=data)
    logger.info(f"Inserted data into db: {result}")

    logger.info("Complete.")

else:
    print("create_db.py imported. Script not ran.")
