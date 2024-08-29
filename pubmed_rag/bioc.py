# imports
import os, time, json, requests 
from utils import assert_path
import pandas as pd

def get_biocjson(id:str, out_path:str, prefix:str='biocjson_', wait:int|float=0)->dict:
    """
    Given a pmid or pmcid retrieves full text (if available) or abstract only from PubMed/Central

    PARAMS
    -----
    - id (str): must be a pmid or pmcid
    - out_path (str): where to save the json files
    - prefix (str): a prefix for the json filenames
    - wait (int): how many seconds to wait between each request

    OUTPUTS
    -----
    - json to the out_path
    - new_result (dict): the files in a dictionary where the keys are the pmid id and values are biocjson

    EXAMPLES
    -----
    TODO

    """

    ### PRECONDITIONS
    assert isinstance(id, str), f"id must be a str: {id}"
    assert_path(out_path)
    assert isinstance(prefix, str), f'prefix must be a string: {prefix}'
    assert (isinstance(wait, int) | isinstance(wait, float)),\
        f"wait must be an integer or float: {wait}"

    ### MAIN FUNCTION

    # Define the PubTator API URL
    pubtator_url = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

    # Send a GET request to the API with the list of PMIDs
    response = requests.get(pubtator_url, params={"pmids": id, "full": True})

    # Check if the request was successful
    if response.status_code == 200:
        # Return the response in JSON format
        result = response.json() 
        # light clean?
        new_result = result['PubTator3'][0]

        # output to json
        with open(os.path.join(out_path, f'{prefix}{id}.json'), 'w') as file:
            json.dump(new_result, file, indent=4)
        
        return new_result
    else:
        print(f"Unable to retrieve {id}: \n Error {response.status_code}: {response.text}")

    # add delay before next request
    time.sleep(wait)        


def passages_to_df(result:dict, out_path:str, prefix:str='df_')->pd.DataFrame:
    """
    Given a biocjson result from get_biocjson() parses to return df of sentences with metadata

    PARAMS
    -----
    - result (dict): from get_biocjson()
    - out_path (str): where to save the json files
    - prefix (str): a prefix for the json filenames

    OUTPUTS
    -----
    - df csv to the out_path
    - df (pd.DataFrame): each observation is a sentence with meta data

    EXAMPLES
    -----
    TODO

    """
    ### PRECONDITIONS
    assert isinstance(result, dict), f"result must be a dict: {result}"
    assert_path(out_path)
    assert isinstance(prefix, str), f'prefix must be a string: {prefix}'


    ### MAIN FUNCTION
    # get ids
    id = result['pmid']
    pmcid = result['pmcid']

    # columns
    columns = ['index', 'section', 'sentence']

    if pmcid is None:
        pas  = [
            [
                i, 
                x['infons']['type'], 
                x['text']
            ]
            for i, x in enumerate(result['passages'])
        ]

    else:
        pas  = [
            [
                i, 
                x['infons']['section_type'], 
                x['text']
            ]
            for i, x in enumerate(result['passages'])
        ]

    # put into a dataframe with metadata columns
    df = pd.DataFrame(pas, columns=columns)
    df['pmid'] = id
    df['pmcid'] = pmcid
    df['date'] = result['date']
    df['authors'] = ' and '.join(result['authors'])
    df['journal'] = result['journal']

    # save to csv
    df.to_csv(os.path.join(out_path, f'{prefix}{id}.csv'), index=False)

    return df
    