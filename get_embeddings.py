# import 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import time, os
from transformers import AutoTokenizer, AutoModel
from pubmed_rag.utils import (
    get_basename,
    generate_log_filename,
    init_log,
    config_loader,
    assert_nonempty_keys,
    assert_nonempty_vals,
)

from pubmed_rag.bioc import (
    collapse_sections,
    get_smaller_texts, 
    get_biocjson,
    passages_to_df, 
)

from pubmed_rag.model import (
    get_tokens,
    get_sentence_embeddings
)

if __name__ == "__main__":

    ## GET ARGS
    # init
    parser = argparse.ArgumentParser(
        prog='embedder',
    )
    parser.add_argument(
        '-c', '--config', 
        action='store',
        default='demo/config.yaml',
        help='provide path to config yaml file'
        )
    parser.add_argument(
        '-fd', '--files_downloaded',
        action='store_true',
        help='add this flag if the biocjson files already exist in the "pmid file path" in the config yaml'
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
    tsne_fpath = config['tsne']
    logger.info(f'Configuration: {config}')

    ## MAIN
    df_embeddings = {}

    logger.info(f"Loading model {chosen_model} from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    model = AutoModel.from_pretrained(chosen_model)

    logger.info(f'Reading in pmids list from file: {pmid_path} ...')
    df = pd.read_csv(pmid_path, header=None)
    pmids = df[0].astype(str).to_list()
    logger.info(f"pmids are: {pmids}")

    # for each pmid
    for i, pmid in enumerate(pmids):
        
        logger.info(f"Getting pubtator3 biocjson from api for {i}: {pmid}")
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
            # df_test['sentence'] = np.where(
            #     df_test['sentence'].str.endswith(punctuations), 
            #     df_test['sentence'], 
            #     df_test['sentence']+'.'
            # )
            # which sections to keep? 
            keep_sections = ['abstract', 'intro', 'results', 'discuss', 'methods', 'concl']
            # filter 
            df_filtered = df_test[df_test['section'].isin(keep_sections)].copy()
            df_filtered = df_filtered[df_filtered['sentence'].apply(lambda x: len(x.split())>5)]
            df_filtered = df_filtered.drop('index', axis=1)

            # # grouping by section
            # logger.info(f"Grouping by section...")
            # collapsed = collapse_sections(df_filtered, 'biocjson')
            # # smaller texts within section
            # logger.info(
            #     f"Smaller texts with max {max_tokens} tokens within section..."
            # )
            # for i, section in enumerate(collapsed['text']):
            #     smaller = get_smaller_texts(section, max_tokens)
            #     collapsed.at[i, 'text'] = smaller
            # exploded = collapsed.explode('text')
            # # fix this later, want ot save over section
            df_filtered.to_csv(
                os.path.join(
                    output_path, 
                    f'sectioned_{pmid}.csv'
                ),
                index=False,
                sep='\t'    
            )

            # GET EMBEDDINGS
            logger.info(f"Getting embeddings for {pmid}")

            # Tokenize sentences
            encoded_input = get_tokens(
                tokenizer,
                df_filtered['sentence'].to_list()
                #exploded['text'].to_list()
            )

            # get embeddings
            sentence_embeddings = get_sentence_embeddings(
                model,
                encoded_input
            ) 

            # append back to df
            df_filtered['embedding'] = pd.Series(
            #exploded['embedding'] = pd.Series(
                sentence_embeddings.detach().numpy().tolist()
            ).values

            # save to csv
            logger.info(f"Saving embeddings to {output_path}")
            df_filtered.to_csv(
            #exploded.to_csv(
                os.path.join(
                    output_path, 
                    f'embed_{pmid}.csv'
                ),
                index=False,
                sep='\t'
            )
            # store for tsne
            df_embeddings[pmid] = df_filtered#exploded
        else:
            f'Result for {pmid} was None.'

        time.sleep(1)

    # SAVE ALL EMBEDDINGS IN ONE FILE
    # put all in one df
    all_dfs = pd.concat(df_embeddings.values(), ignore_index=True)
    num_emb = len(all_dfs)
    out_all_emb_fpath = os.path.join(output_path, 'all_embeddings.csv')
    logger.info(f"Saving all {num_emb} embeddings to: {out_all_emb_fpath}")
    # save df to csv
    all_dfs.to_csv(out_all_emb_fpath, sep='\t')

    # if nonempty string / not False
    if tsne_fpath:
        # GENERATE tsne
        logger.info(f"Generating tsne for {num_emb} embeddings")
        
        # testing 
        # all_dfs = pd.read_csv(os.path.join(output_path, 'all_embeddings.csv'), )
        # all_dfs['embedding'] = all_dfs['embedding'].apply(literal_eval)
        # all_dfs['embedding'] = all_dfs['embedding'].apply(lambda x: np.array(x))
        # num_emb = len(all_dfs)

        # perplexity must be less than n_samples
        if num_emb > 50:
            perp = 50
        elif num_emb > 25:
            perp = 25
        else:
            perp = num_emb-0.5

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        entity_embeddings_2d = tsne.fit_transform(np.vstack(all_dfs['embedding']))

        # Get labels
        label_encoder = LabelEncoder()
        sections_enc = label_encoder.fit_transform(all_dfs['section'])

        # Plot the embeddings
        plt.figure()
        scatter = plt.scatter(
            entity_embeddings_2d[:, 0], 
            entity_embeddings_2d[:, 1], 
            alpha=0.5, 
            c=sections_enc, 
        )
        # Mapping encoded labels back to original labels in the legend
        handles, _ = scatter.legend_elements()
        original_labels = label_encoder.inverse_transform(range(len(handles)))

        # Add a legend with the original (non-encoded) labels
        plt.legend(handles, original_labels, title="sections")

        plt.title('t-SNE Visualization of Embeddings by section')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(tsne_fpath)
        logger.info(f"Tsne plot saved to {tsne_fpath}")
    else:
        logger.info(f"No tsne filepath provided in config file. Tsne plot not generated.")

    logger.info('Complete.')

else:
    print('get_embeddings.py imported. Script not ran.')
