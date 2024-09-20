# import 
import numpy as np
import pandas as pd
import time, os
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
    passages_to_df
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
    logger.info(f'Configuration: {config}')

    ## MAIN
    logger.info(f"Preparing embedding model: {}")
    

    logger.info(f'Reading in pmids list from file: {pmid_path} ...')
    df = pd.read_csv(pmid_path, header=None)
    pmids = df[0].astype(str).to_list()
    logger.info(f"pmids are: {pmids}")

    # for each pmid
    for pmid in pmids:
        
        logger.info(f"Getting pubtator3 biocjson from api for {pmid}")
        result = get_biocjson(pmid, output_path)
        logger.info(f"Saved to {output_path}.")

        if result is not None:
            logger.info(f"Now, passages to df and saved to: {output_path}")
            df_test = passages_to_df(result, output_path)

            # cleaning?
            logger.info(f"Light cleaning")
            # lower case section names
            df_test['section'] = df_test['section'].str.lower().str.strip()
            # pmids to object
            df_test['pmid'] = df_test['pmid'].astype(str)
            df_test['date'] = pd.to_datetime(df_test['date'])
            # also stripping sentences in case?
            df_test['sentence'] = df_test['sentence'].str.strip()
            punctuations = ('!',',','.','?',',','"', "'")
            # lol adding a . to the end for now? if no punc
            df_test['sentence'] = np.where(
                df_test['sentence'].str.endswith(punctuations), 
                df_test['sentence'], 
                df_test['sentence']+'.'
            )
            # which sections to keep? 
            keep_sections = ['title', 'abstract', 'intro', 'results', 'discuss', 'methods']
            # filter 
            df_filtered = df_test[df_test['section'].isin(keep_sections)]

            # grouping by section
            logger.info(f"Grouping by section...")
            collapsed = collapse_sections(df_filtered, 'biocjson')
            # smaller texts within section
            logger.info(
                f"Smaller texts with max {max_tokens} tokens within section..."
            )
            for i, section in enumerate(collapsed['text']):
                smaller = get_smaller_texts(section, max_tokens)
                collapsed.at[i, 'text'] = smaller
            exploded = collapsed.explode('text')
            # fix this later, want ot save over section
            exploded.to_csv(
                os.path.join(
                    output_path, 
                    f'sectioned_{pmid}.csv',
                    index=False
                )
            )

            # get embeddings 

        time.sleep(1)

    logger.info('Complete.')

else:
    print('main.py imported. scholarly queries not initiated.')
