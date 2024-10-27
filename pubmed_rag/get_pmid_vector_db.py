import subprocess

from pubmed_rag.helpers.utils import (
    get_args,
    get_logger,
)

if __name__ == "__main__":
    ## GET ARGS
    # init
    args = get_args(
        prog_name="create_pmid_db",
        others=dict(description="gets pmid articles as embeddings in vector database"),
    )

    ## START LOG FILE
    logger = get_logger()
    logger.info(f"Arguments: {args}")

    ## LOAD CONFIG PARAMETERS
    # getting the config filepath
    config_filepath = args.config
    # log it
    logger.info(f"Path to config file: {config_filepath}")

    get_embed_cmd = [
        "python",
        "get_embeddings.py",
        "-c",
        config_filepath,
        "-fd",
        args.files_downloaded,
    ]
    logger.info(f"Preparing command for 'get_embeddings.py': {get_embed_cmd}")
    create_db_cmd = [
        "python",
        "create_db.py",
        "-c",
        config_filepath,
    ]
    logger.info(f"Preparing command for 'create_db.py': {create_db_cmd}")

    # getting the embeddings
    logger.info("Initiating 'get_embeddings.py'...")
    subprocess.call(get_embed_cmd)
    # putting in vector db collection
    logger.info("Initiating 'create_db.py'...")
    subprocess.call(create_db_cmd)

else:
    print("get_pmid_vector_db.py imported. Script not ran.")
