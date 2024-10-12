# these functions are pretty general (file that can be reused across projects)
import argparse
from datetime import datetime
import yaml
import os, sys
import logging
import glob
import warnings

## CHECKS
def assert_path(filepath:str):
    """
    Checks that fpath is a string and that it exists
    
    PARAMS
    -----
    - filepath (str): the filepath or folderpath

    OUTPUTS
    -----
    - raises assertion error if filepath is not a string or doesn't exist
    """

    assert isinstance(filepath, str), \
        f'filepath must be a string: {filepath}'
    assert os.path.exists(os.path.abspath(filepath)), \
        f'filepath does not exist: {os.path.abspath(filepath)}'

def assert_nonempty_keys(dictionary:dict):
    """
    - Checks that the keys are not empty strings
    - Can they be numbers? I guess 

    PARAMS
    -----
    - dictionary (dict): a dictionary e.g. config file
    """

    # PRECONDITIONS
    assert isinstance(dictionary, dict), 'dictionary must be a dict'

    # MAIN FUNCTION
    for key in dictionary:
        if type(key) is str:
            assert key, f'There is an empty key (e.g., ""): {key, dictionary.keys()}'
            assert key.strip(), f'There is a blank key (e.g., space, " "): {key, dictionary.keys()}'

def assert_nonempty_vals(dictionary:dict):
    """
    - Checks that the dict values are not empty strings

    PARAMS
    -----
    - dictionary (dict): a dictionary e.g. config file
    """

    # PRECONDITIONS
    assert isinstance(dictionary, dict), 'dictionary must be a dict'

    # MAIN FUNCTION
    for v in dictionary.items():
        if type(v) is str:
            assert v, f'There is an empty key (e.g., ""): {v, dictionary.items()}'
            assert v.strip(), f'There is a blank key (e.g., space, " "): {v, dictionary.items()}'

def warn_folder(
        folderpath:str, 
        warning_message:str=f'Warning: There are existing files in the given folderpath.'
    ):
    """
    Checks if the folder is empty. If not empty raise a warning. 

    PARAMETERS
    -----
    - folderpath (str): path to folder
    - warning_message (str): what you warning message should be if folder is not empty

    OUTPUTS
    -----
    - no return
    - prints a warning message if files exist in the folderpath

    EXAMPLES
    -----
    1)
    >>> warn_folder('/Users/this/that/empty_folder')
    
    2)
    >>> warn_folder('/Users/this/that/non_empty_folder')
    Warning: There are existing files in the given folderpath.
    """

    # PRECONDITIONS
    assert_path(folderpath)
    assert isinstance(warning_message, str), f'warning message must be string: {warning_message}'

    if len(glob.glob(os.path.join(folderpath,'*'))) > 0: 
        warnings.warn(
            warning_message,
            UserWarning
        )

        return warning_message

# FUNCTIONS FOR CONFIG 
def config_loader(filepath:str) -> dict:
    """
    Loads in yaml config file as dict

    PARAMETERS
    -----
    - filepath (str): path to the config file

    RETURNS
    -----
    - contents (dict): configuration parameters as a dictionary

    EXAMPLE
    -----
    >>> config_dict = config_loader('config/config.yaml')
    
    """

    # PRECONDITIONS
    assert_path(filepath)

    # MAIN FUNCTION
    with open(filepath, 'r') as f:
        contents = yaml.safe_load(f)

    # POSTCONDITIONS
    assert isinstance(contents, dict), "content not returned as a dict"

    return contents

def get_args(prog_name:str, others:dict={}):
    """
    Initiates argparse.ArugmentParser() and adds common arguments.

    :param prog_name: The name of the program. 
    :type prog_name: str

    :returns: 
    :rtype: 
    """

    ### PRECONDITIONS
    assert isinstance(prog_name, str), \
        f"prog_name should be a string: {prog_name}"
    assert isinstance(others, dict), \
        f"other kwargs must be a dict: {others}"
    ## MAIN FUNCTION
    # init
    parser = argparse.ArgumentParser(
        prog=prog_name,
        **others
    )
    # config file path
    parser.add_argument(
        '-c', '--config',
        action='store',
        default='demo/config.yaml',
        help='provide path to config yaml file'
    )
    # used in run_search.py
    parser.add_argument(
        '-q', '--query',
        action='store',
        help='text to embed and search in vector db'
    )
    # used in get_embeddings.py
    parser.add_argument(
        '-fd', '--files_downloaded',
        action='store_true',
        help='add this flag if the biocjson files already exist in the "pmid file path" in the config yaml'
    )
    args = parser.parse_args()
    return args


## FOR LOGGING
def get_basename(fname:None|str=None)->str:
    """
    - For a given filename, returns basename WITHOUT file extension
    - If no fname given (i.e., None) then return basename that the function is called in

    PARAMS
    -----
    - fname (None or str): the filename to get basename of, or None 

    OUTPUTS
    -----
    - basename of given filepath or the current file the function is executed

    EXAMPLES
    -----
    1)
    >>> get_basename()
    utils

    2) 
    >>> get_basename('this/is-a-filepath.csv')
    is-a-filepath
    """
    if fname is not None: 
        # PRECONDITION
        assert_path(fname)
        # MAIN FUNCTIONS
        return os.path.splitext(os.path.basename(fname))[0]
    else:
        return os.path.splitext(os.path.basename(sys.argv[0]))[0]

def get_time(incl_time:bool=True, incl_timezone:bool=True) -> str:
    """
    Gets current date, time (optional) and timezone (optional) for file naming

    PARAMETERS
    -----
    - incl_time (bool): whether to include timestamp in the string
    - incl_timezone (bool): whether to include the timezone in the string

    RETURNS
    -----
    - fname (str): includes date, timestamp and/or timezone 
        connected by '_' in one string e.g. yyyyMMdd_hhmm_timezone 

    EXAMPLES
    -----
    1)
    >>> get_time()
    '20231019_101758_CEST'

    2)
    >>> get_time(incl_time=False)
    '20231019_CEST'

    """

    # PRECONDITIONALS
    assert isinstance(incl_time, bool), 'incl_time must be True or False'
    assert isinstance(incl_timezone, bool), 'incl_timezone must be True or False'

    # MAIN FUNCTION
    # getting current time and timezone
    the_time = datetime.now()
    timezone = datetime.now().astimezone().tzname()
    # convert date parts to string
    y = str(the_time.year)
    M = str(the_time.month)
    d = str(the_time.day)
    h = str(the_time.hour)
    m = str(the_time.minute)
    s = str(the_time.second)
    # putting date parts into one string
    if incl_time and incl_timezone:
        fname = '_'.join([y+M+d, h+m+s, timezone])
    elif incl_time:
        fname = '_'.join([y+M+d, h+m+s])
    elif incl_timezone:
        fname = '_'.join([y+M+d, timezone])
    else:
        fname = y+M+d

    # POSTCONDITIONALS
    parts = fname.split('_')
    if incl_time and incl_timezone:
        assert len(parts) == 3, f'time and/or timezone inclusion issue: {fname}'
    elif incl_time or incl_timezone:
        assert len(parts) == 2, f'time/timezone inclusion issue: {fname}'
    else:
        assert len(parts) == 1, f'time/timezone inclusion issue: {fname}'

    return fname

def generate_log_filename(folder:str='logs', suffix:str='') -> str:
    """
    Creates log file name and path

    PARAMETERS
    -----
    folder (str): name of the folder to put the log file in
    suffix (str): anything else you want to add to the log file name

    RETURNS
    -----
    log_filepath (str): the file path to the log file
    """
    # PRECONDITIONS
    assert_path(folder)
    
    # MAIN FUNCTION
    log_filename = get_time(incl_timezone=False) + '_' + suffix +'.log'
    log_filepath = os.path.join(folder, log_filename)
    
    return log_filepath

def init_log(filename:str, display:bool=False, logger_id:str|None=None):
    """
    - Custom python logger configuration (basicConfig())
        with two handlers (for stdout and for file)
    - from: https://stackoverflow.com/a/44760039
    - Keeps a log record file of the python application, with option to
        display in stdout 
    
    PARAMETERS
    -----
    - filename (str): filepath to log record file
    - display (bool): whether to print the logs to whatever standard output
    - logger_id (str): an optional identifier for yourself, 
        if None then defaults to 'root'

    RETURNS
    -----
    - logger object

    EXAMPLE
    -----
    >>> logger = init_log('logs/tmp.log', display=True)
    >>> logger.info('Loading things')
    [2023-10-20 10:38:03,074] root: INFO - Loading things
    """
    # PRECONDITIONALS
    assert isinstance(filename, str), 'filename must be a string'
    assert (isinstance(logger_id, str) or logger_id is None), \
        'logger_id must be a string or None'

    # MAIN FUNCTION
    # init handlers
    file_handler = logging.FileHandler(filename=filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    if display: 
        handlers = [file_handler, stdout_handler]
    else: 
        handlers = [file_handler]

    # logger configuration
    logging.basicConfig(
        #level=logging.DEBUG, 
        format='[%(asctime)s] %(name)s: %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    # instantiate the logger
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.DEBUG)

    return logger


## OTHERS
def get_chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def filter_filepaths(
        fpath:str|list, 
        identifiers:list=[''], 
        exclude:None|list=None
    )->list:
    """
    Isolating files to iterate through. Can provide multiple identifiers.
    if list given, then filters list. 
    if str/path given, then acquires list first. 
    """

    # PRECONDITIONALS
    assert isinstance(identifiers, list), 'exclude must be None or a list of strngs'
    for id in identifiers:
        assert isinstance(id, str), f'must all be strings: {id} not a string'
       
    if exclude is not None:
        assert isinstance(exclude, list), 'exclude must be None or a list of strngs'
        for ex in exclude:
            assert isinstance(ex, str), f'must all be strings: {ex} not a string'
    
    # MAIN FUNCTION

    # if path (str) given then get a list of files first
    if type(fpath) is str:
        assert_path(fpath)
        filepaths = glob.glob(f'{fpath}/**', recursive=True)
    # if a list of filenames then continue
    elif type(fpath) is list:
        filepaths = fpath
    else: 
        raise TypeError(f'fpath must be a string or list: {type(fpath)}')

    # check
    for path in filepaths:
        assert_path(path)

    # filtering for files that have those identifiers
    in_filtered = [file for file in filepaths if \
                all([id in os.path.basename(file) for id in identifiers])]
    
    if exclude is None:
        return in_filtered
    else:
        # filter out the files that match in the exclusion list
        ex_filtered = [file for file in in_filtered if \
                    all([ex not in os.path.basename(file) for ex in exclude])]

        return ex_filtered
    
def pipe(input, functions:list=[]):
    '''
    - Pipes output of one function into another like '|' in linux
    - https://stackoverflow.com/questions/28252585/functional-pipes-in-python-like-from-rs-magrittr
    - Limitation: 
        - All functions must only expect one argument
        - Defaults will have to be used for all other arguments
    
    PARAMS
    -----
    - input (Any): whatever input the first function is expecting
    - functions (list): a list of functions IN ORDER OF EXECUTION 
        do not include () at end of functions

    OUTPUTS
    -----
    - Any or None (whatever the expected output is from the final function in the list)

    '''
    # PRECONDITIONALS
    assert isinstance(functions, list), 'functions should be in a list'
    for f in functions: 
        assert callable(f), f'{f} is not a function'

    # MAIN FUNCTION
    for f in functions:
        input = f(input)
    return input
