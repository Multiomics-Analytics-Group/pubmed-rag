# these functions are pretty general (file that can be reused across projects)
import argparse
import glob
import logging
import os
import sys
import warnings
from datetime import datetime
from urllib.parse import urlparse, urlunsplit
import yaml


## CHECKS
def assert_path(filepath: str):
    """
    Checks that the given filepath is a string and that it exists.

    :param str filepath: The filepath or folder path to check.
    :raises TypeError: If the filepath is not a string.
    :raises FileNotFoundError: If the filepath does not exist.
    """

    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string: {type(filepath)}")
    if not os.path.exists(os.path.abspath(filepath)):
        raise FileNotFoundError(f"The specified path does not exist: {filepath}")


def create_folder(directory_path: str, is_nested: bool = False) -> bool:
    """
    alternative to assert_path to create folder if it doesn't exist
    :param directory_path: The path of the directory to create.
    :type directory_path: str
    :param is_nested: A flag indicating whether to create nested directories (True uses os.makedirs, False uses os.mkdir).
    :type is_nested: bool
    :returns: True if the folder was created or False if it already existed.
    :rtype: bool
    :raises OSError: If there is an error creating the directory.
    """
    # PRECONDITION CHECK
    if not isinstance(directory_path, str):
        raise TypeError(f"filepath must be a string: {type(directory_path)}")
    abs_path = os.path.abspath(directory_path)

    # make sure it is a folder not a file
    if os.path.isfile(abs_path):
        raise ValueError(
            f"directory_path is an existing file when it should be a folder/foldername: {abs_path}"
        )
    # if folder already exists
    elif os.path.isdir(abs_path):
        return False
    # create the folder(s)
    else:
        try:
            if is_nested:
                # Create the directory and any necessary parent directories
                os.makedirs(directory_path, exist_ok=True)
                return True
            else:
                # Create only the final directory (not nested)
                os.mkdir(directory_path)
                return True
        except OSError as e:
            raise OSError(f"Error creating directory '{directory_path}': {e}")


def assert_nonempty_keys(dictionary: dict):
    """
    - Checks that the keys are not empty strings
    - Can they be numbers? I guess

    PARAMS
    -----
    - dictionary (dict): a dictionary e.g. config file
    """

    # PRECONDITIONS
    assert isinstance(dictionary, dict), "dictionary must be a dict"

    # MAIN FUNCTION
    for key in dictionary:
        if type(key) is str:
            assert key, f'There is an empty key (e.g., ""): {key, dictionary.keys()}'
            assert (
                key.strip()
            ), f'There is a blank key (e.g., space, " "): {key, dictionary.keys()}'


def assert_nonempty_vals(dictionary: dict):
    """
    - Checks that the dict values are not empty strings

    PARAMS
    -----
    - dictionary (dict): a dictionary e.g. config file
    """

    # PRECONDITIONS
    assert isinstance(dictionary, dict), "dictionary must be a dict"

    # MAIN FUNCTION
    for v in dictionary.items():
        if type(v) is str:
            assert v, f'There is an empty key (e.g., ""): {v, dictionary.items()}'
            assert (
                v.strip()
            ), f'There is a blank key (e.g., space, " "): {v, dictionary.items()}'


def warn_folder(
    folderpath: str,
    warning_message: str = "Warning: There are existing files in the given folderpath.",
):
    """
    Checks if the folder is empty. If not empty, raises a warning.

    :param folderpath: Path to the folder.
    :type folderpath: str
    :param warning_message: Warning message to display if the folder is not empty, defaults to "Warning: There are existing files in the given folderpath."
    :type warning_message: str, optional

    :raises FileNotFoundError: If the specified path does not exist.
    :raises AssertionError: If the warning message is not a string.

    :return: Warning message if files exist in the folder.
    :rtype: str

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
    assert isinstance(
        warning_message, str
    ), f"warning message must be string: {warning_message}"

    if len(glob.glob(os.path.join(folderpath, "*"))) > 0:
        warnings.warn(warning_message, UserWarning)

        return warning_message


def normalize_url(host: str, port: int, scheme: str = "http") -> str:
    """
    Normalize the given URL. Ensure the URL starts with 'http://'

    This function takes a URL and normalizes it by ensuring it has a scheme,
    converting it to lowercase, and removing any trailing slashes.

    :param host: The host to be normalized.
    :type host: str
    :param port: The port
    :type port: int
    :param scheme: the scheme
    :type scheme: str
    :return: The normalized URL.
    :rtype: str

    """
    ## PRECONDITIONS
    if not isinstance(host, str):
        raise TypeError(f"host should be a str e.g., 'localhost': {type(host)}")
    if not isinstance(port, int):
        raise TypeError(f"port must be int e.g., '7474': {type(port)}")
    if not isinstance(scheme, str):
        raise TypeError(f"scheme must be str: {type(scheme)}")

    ## MAIN FUNCTION
    if not urlparse(host).netloc:
        host = urlunsplit([scheme, host, "", "", ""])

    # Remove any trailing slashes
    url = host.rstrip("/")

    # Add the port
    url = f"{url}:{str(port)}"

    ## POSTCOND CHECKS
    if not urlparse(url).netloc:
        raise TypeError(f"Unable to normalize url: {url}")

    return url


# FUNCTIONS FOR CONFIG
def config_loader(filepath: str) -> dict:
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
    with open(filepath, "r") as f:
        contents = yaml.safe_load(f)

    # POSTCONDITIONS
    assert isinstance(contents, dict), "content not returned as a dict"

    return contents


def filter_config(config:dict, additional:list=["pubmed_rag"])->dict:
    """
    Filters the given configuration dictionary to only include specified keys.
    This function takes a configuration dictionary and filters it to only include
    keys specified in the `additional` list and the result of `get_basename()`.
    :param config: The configuration dictionary to filter.
    :type config: dict
    :param additional: A list of additional keys to keep in the filtered configuration.
                        Defaults to ["pubmed_rag"].
    :type additional: list
    :raises TypeError: If `config` is not a dictionary or `additional` is not a list.
    :return: A filtered configuration dictionary containing only the specified keys.
    :rtype: dict
    """

    ## PRECONDITION
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict: {type(config)}")
    if not isinstance(additional, list):
        raise TypeError(f"additional must be a list: {type(additional)}")
    
    ## MAIN FUNCTION
    # which keys in config to keep
    keep = [get_basename()] + additional

    filtered_config = {
        k:v for (k,v) in config.items() if k in keep
    }

    return filter_config


def get_args(prog_name: str, others: dict = {}):
    """
    Initiates argparse.ArugmentParser() and adds common arguments.

    :param prog_name: The name of the program.
    :type prog_name: str

    :returns:
    :rtype:
    """
    ### PRECONDITIONS
    if not isinstance(prog_name, str):
        raise TypeError(f"prog_name should be a string: {type(prog_name)}")
    if not isinstance(others, dict):
        raise TypeError(f"other kwargs must be a dict: {type(others)}")
    ## MAIN FUNCTION
    # init
    parser = argparse.ArgumentParser(prog=prog_name, **others)
    # config file path
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default="demo/config.yaml",
        help="provide path to config yaml file",
    )
    args = parser.parse_args()
    return args


## FOR LOGGING
def get_basename(fname: None | str = None) -> str:
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


def get_time(incl_time: bool = True, incl_timezone: bool = True) -> str:
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
    assert isinstance(incl_time, bool), "incl_time must be True or False"
    assert isinstance(incl_timezone, bool), "incl_timezone must be True or False"

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
        fname = "_".join([y + M + d, h + m + s, timezone])
    elif incl_time:
        fname = "_".join([y + M + d, h + m + s])
    elif incl_timezone:
        fname = "_".join([y + M + d, timezone])
    else:
        fname = y + M + d

    # POSTCONDITIONALS
    parts = fname.split("_")
    if incl_time and incl_timezone:
        assert len(parts) == 3, f"time and/or timezone inclusion issue: {fname}"
    elif incl_time or incl_timezone:
        assert len(parts) == 2, f"time/timezone inclusion issue: {fname}"
    else:
        assert len(parts) == 1, f"time/timezone inclusion issue: {fname}"

    return fname


def generate_log_filename(folder: str = "logs", suffix: str = "") -> str:
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
    log_filename = get_time(incl_timezone=False) + "_" + suffix + ".log"
    log_filepath = os.path.join(folder, log_filename)

    return log_filepath


def init_log(filename: str, display: bool = False, logger_id: str | None = None):
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
    assert isinstance(filename, str), "filename must be a string"
    assert (
        isinstance(logger_id, str) or logger_id is None
    ), "logger_id must be a string or None"

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
        # level=logging.DEBUG,
        format="[%(asctime)s] %(name)s: %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("matplotlib.font_manager").disabled = True

    # instantiate the logger
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.DEBUG)

    return logger


def get_logger():
    """
    Putting at all together to init the log file.
    """
    # get log suffix, which will be the current script's base file name
    log_suffix = get_basename()
    # generate log file name
    log_file = generate_log_filename(suffix=log_suffix)
    # init logger
    logger = init_log(log_file, display=True)
    # log it
    logger.info(f"Path to log file: {log_file}")

    return logger


## OTHERS
def get_chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def filter_filepaths(
    fpath: str | list, identifiers: list = [""], exclude: None | list = None
) -> list:
    """
    Isolating files to iterate through. Can provide multiple identifiers.
    if list given, then filters list.
    if str/path given, then acquires list first.
    """

    # PRECONDITIONALS
    assert isinstance(identifiers, list), "exclude must be None or a list of strngs"
    for id in identifiers:
        assert isinstance(id, str), f"must all be strings: {id} not a string"

    if exclude is not None:
        assert isinstance(exclude, list), "exclude must be None or a list of strngs"
        for ex in exclude:
            assert isinstance(ex, str), f"must all be strings: {ex} not a string"

    # MAIN FUNCTION

    # if path (str) given then get a list of files first
    if type(fpath) is str:
        assert_path(fpath)
        filepaths = glob.glob(f"{fpath}/**", recursive=True)
    # if a list of filenames then continue
    elif type(fpath) is list:
        filepaths = fpath
    else:
        raise TypeError(f"fpath must be a string or list: {type(fpath)}")

    # check
    for path in filepaths:
        assert_path(path)

    # filtering for files that have those identifiers
    in_filtered = [
        file
        for file in filepaths
        if all([id in os.path.basename(file) for id in identifiers])
    ]

    if exclude is None:
        return in_filtered
    else:
        # filter out the files that match in the exclusion list
        ex_filtered = [
            file
            for file in in_filtered
            if all([ex not in os.path.basename(file) for ex in exclude])
        ]

        return ex_filtered
