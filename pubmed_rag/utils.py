# these functions are pretty general (file that can be reused across projects)
from datetime import datetime
import yaml
import shutil, os, sys, re
import logging
import glob
import csv
import subprocess
import warnings

# CHECKS
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


# other
def get_chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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


def find_matches(filepath:str, sub:tuple=('-data\d*', '-header'), area:str|list|None=None):
    """
    Search `area` for files matching the given `filepath` name with replaced substrings specified in `sub` 

    PARAMS
    -----
    - filepath (str): path to an existing file 
    - sub (tuple): substring to replace in the filepath (<pattern>, <replacement>)
        - if no sub then give a tuple with 2 empty strings i.e., ('', '')
    - area (NoneType or list or str):
        - if None then search in the same location that filepath is located
        - if list of files then search for matches within this list
        - if string should be a folderpath and will search for matches within this folder
    
    OUTPUTS
    -----
    - matches (list): list with matching paths
    """
    
    # PRECONDITIONS
    assert_path(filepath)
    assert isinstance(sub, tuple), f'sub must be a tuple with 2 items (search, replace): {sub}'
    assert len(sub)==2, f'sub must be a tuple with 2 items (search, replace): {sub}'
    for term in sub:
        assert isinstance(term, str), f'items in sub tuple must be strings (can be empty stirngs): {term}'
    # check search area
    if area is None:
        # then search the folder that data_file is in 
        area = os.path.split(filepath)[0]
    elif isinstance(area, str):
        assert_path(area)
        assert os.path.isdir(area), f'area {area} is not a folder'
    elif isinstance(area, list):
        for file in area:
            assert_path(file)
    else:
        raise ValueError(f"area must be a folderpath as a str, or a list of files: {area}")
    
    # MAIN FUNCTION
    search_sub = re.sub(sub[0], sub[1], get_basename(filepath))

    matches = filter_filepaths(area, identifiers=[search_sub])

    return matches


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
        level=logging.DEBUG, 
        format='[%(asctime)s] %(name)s: %(levelname)s - %(message)s',
        handlers=handlers
    )

    # instantiate the logger
    logger = logging.getLogger(logger_id)

    return logger


def arg_loader() -> list:
    """
    Retreives all arguments after python/shell filename in command line and returns as a list. 
    If no args than returns an empty list.

    RETURNS
    -----
    - input_args (list): A list of command line arguments

    EXAMPLES
    -----
    Given files:
        main.py >
            from utils import arg_loader
            arg_loader()
        run_main.sh >
            python main.py "$@"

    1) 
    % python main.py something another
    ['something', 'another']

    2)
    % bash run_main.sh
    []

    3)
    % bash run_main.sh 'well okay' then
    ['well okay', 'then']
    """
    input_args = sys.argv[1:]
    return input_args


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


def get_config_path() -> str:
    """
    Retrieves path to config file from arg_loader and checks that it exists.

    RETURNS
    -----
    config_filepath (str): path to config file 
    """
    arg_list = arg_loader()
    assert len(arg_list) > 0, \
        'Please provide path to config file as an argument e.g. python main.py config/config.yaml' 
    config_filepath = arg_list[0]
    assert os.path.exists(os.path.abspath(config_filepath)), \
        f'filepath to config file does not exist: {os.path.abspath(config_filepath)}'
    return config_filepath


def copy_recursively(src:str, dest:str):
    """
    Copying all directory and subdirectory contents to a destination folder

    PARAMETERS
    -----
    src (str): absolute path to source folder/files
    dest (str): absolute path to destination folder/file

    OUTPUT
    -----
    - if single source file to copy:
        - if filename included in dest variable then this filename is used
        - if no filename given, then dest will be treated as the dir name and the original filename will be used
    - if multiple source files, all copied to given dest (dest auto treated as dir name)

    EXAMPLES
    -----
    1)
    >>> copy_recursively('something.ipynb', 'whatever')
    Dir created:
        /whatever/
    File created:
        /whatever/something.ipynb

    2) 
    >>> copy_recursively('here', 'there')
    Skipping (already exists): 
        /there/
    File created:
        /there/file1.csv
    Skipping (already exists):
        /there/file2.csv
    File created:
        /there/file3.csv

    """
    # PRECONDITIONS
    assert_path(src)
    assert isinstance(dest, str), 'dest path must be a string'

    # MAIN FUNCTION
    abs_src = os.path.abspath(src)
    abs_dest = os.path.abspath(dest)    
    # if a single file to copy
    if os.path.isfile(abs_src):
        
        try:
            # for checks:
            # get file extension to see if exists
            _, fext = os.path.splitext(abs_dest)
            # subdirs
            subdirs = os.path.split(abs_dest)[0]

            # if dest file already exists then skip
            if os.path.isfile(abs_dest):
                print(f'Skipping (already exists):\n\t{abs_dest}')
            # if dest an existing directory, copy file with same name
            elif os.path.isdir(abs_dest):
                print(f'{new_dest} is a directory')
                new_dest = os.path.join(abs_dest, os.path.basename(abs_src))
                shutil.copy2(abs_src, new_dest)
                print(f'File created:\n\t{new_dest}')
            # if no filename (no ext) given: 
            # create dir and copy file with same name
            elif fext == '':
                os.makedirs(abs_dest)
                print(f'Dir created:\n\t{abs_dest}')
                new_dest = os.path.join(abs_dest, os.path.basename(abs_src))
                shutil.copy2(abs_src, new_dest)
                print(f'File created:\n\t{new_dest}')
            # so we know a filename was given: if subdirs exist just copy file
            elif os.path.isdir(subdirs):
                shutil.copy2(abs_src, abs_dest)
                print(f'File created:\n\t{abs_dest}')     
            # if the subdirs don't exist: makedirs then copy file           
            else:
                os.makedirs(subdirs)
                print(f'Dir created:\n\t{subdirs}')
                shutil.copy2(abs_src, abs_dest)
                print(f'File created:\n\t{abs_dest}')
        except Exception as e: 
            print(f'Something went wrong at {abs_src}', e)

    # if a dir
    elif os.path.isdir(abs_src):
        # get list of files to copy
        all_files = glob.glob(f'{abs_src}/**', recursive=True)

        for item in all_files: 
            # destination folder naming as new_dest
            new_dest = item.replace(abs_src, abs_dest)
            # if dest file/folder already exists skip
            if os.path.exists(new_dest):
                # verbose
                print(f'Skipping (already exists):\n\t{new_dest}')
            # create folder or copy file if doesn't already exist in dest
            else: 
                if os.path.isfile(item):
                    shutil.copy2(item, new_dest)
                    print(f'File created:\n\t{new_dest}')
                elif os.path.isdir(item):
                    os.makedirs(new_dest)
                    print(f'Dir created:\n\t{new_dest}')
                else:
                    f'Something went wrong at {item}'
    else:
        f'{abs_src} was neither an existing file or dir'


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
    

def group_files(
        source_folder:str|list, 
        sub:tuple=('-datab*\d*.csv', ''),
        include_files_with:list=[''],
        exclude_files_with:None|list=None
    )->dict:
    '''
    - In a given folder, or list of files, groups the files based on naming
    - Main puropse is to group multicsvs in a list

    PARAMS
    -----
    - source_folder (str or list): path to folder with the datafiles or list of datafiles
    - sub (tuple): substring to replace in the filepath (<pattern>, <replacement>)
        - if no sub then give a tuple with 2 empty strings i.e., ('', '')
    - exclude_files_with (None or list): a list of strings for use in filter_filepaths(exclude=)
    - include_files_with (list): a list of strings for use in filter_filepaths(identifiers=)

    OUTPUTS
    -----
    - grouped_files (dict): a dictionary of lists where key is the group identifier, and 
        list contains files that matched


    '''
    # PRECONDITIONS
    if isinstance(source_folder, str):
        assert_path(source_folder)
    else:
        assert isinstance(source_folder, list), f'source_folder must be a folderpath or a list of filepaths: {source_folder}'
        for file in source_folder:
            assert_path(file)
    assert isinstance(sub, tuple), f'sub must be a tuple with 2 items (search, replace): {sub}'
    assert len(sub)==2, f'sub must be a tuple with 2 items (search, replace): {sub}'
    if exclude_files_with is not None:
        assert isinstance(exclude_files_with, list), f'exclude_files_with must be a list of strings: {exclude_files_with}'
        for ex in exclude_files_with:
            assert isinstance(ex, str), f'exclude_files_with must be a list of strings: {ex}'
    assert isinstance(include_files_with, list), f'include_files_with must be a list of strings: {include_files_with}'
    for i in include_files_with:
        assert isinstance(i, str), f'include_files_with must be a list of strings: {i}'

    # MAIN FUNCTION
    # init dict
    grouped_files = {}
    # getting all data files
    result = filter_filepaths(source_folder, identifiers=include_files_with, exclude=exclude_files_with)
    # isolating unique main file names
    files_only = [f for f in result if os.path.isfile(f)]
    file_groups = set([re.sub(sub[0], sub[1], fname) for fname in files_only])
    # generate grouped lists 
    for file in file_groups:
        grouped_files[file] = [f for f in files_only if file in f]
    
    return grouped_files


def get_header(filepath:str, row:int=0, sep:str='|', quotechar:str='"')-> list:
    """
    Gets the header row from a csv file and returns as a list.

    PARAMS
    -----
    - filepath (str): path to file, should be a csv file
    - row (int): the index of the row that contains the header
    - sep (str): the delimiter of the csv file
    - quotechar (str): quote character 

    OUTPUTS
    -----
    - header_list (list): the header as a list with each column as an item

    EXAMPLES
    -----
    """
    # PRECONDITIONS
    assert_path(filepath)
    assert isinstance(row, int), 'row number should be an integer'
    assert isinstance(sep, str), 'sep should be given as a string'
    assert isinstance(quotechar, str), 'quotechar should be given as a string'

    # MAIN FUNCTION  
    if row > 0:
        # read in header from csv (this takes a long time if its a big file ..)
        with open(filepath) as fp:
            reader = csv.reader(fp, delimiter=sep, quotechar=quotechar)
            all_rows = list(reader)
            assert row < len(all_rows), 'given row index outside of csv indices'
            header_list = all_rows[row]
    else:
        # read in first row from csv only (faster)
        with open(filepath) as fp:
            reader = csv.reader(fp, delimiter=sep, quotechar=quotechar)
            header_list = next(reader)    

    # POSTCONDITIONALS
    assert len(header_list) > 0, 'Empty header list!'

    return header_list


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


def get_line_count(filepath:str)->int:
    """
    Retreives the line count of a file
    https://stackoverflow.com/questions/64744161/best-way-to-find-out-number-of-rows-in-csv-without-loading-the-full-thing
    """
    # PRECONDITIONS
    assert_path(filepath)
    abspath = os.path.abspath(filepath)
    assert os.path.isfile(abspath), f'{abspath} is not a file.'

    # MAIN FUNCTION
    query_result = subprocess.check_output(f"wc -l {abspath}", shell=True)
    try:
        count_lines = int(query_result.split()[0])
    except ValueError as error:
        raise

    # POSTCONDITION
    assert isinstance(count_lines, int), 'line count result not returned as an int'

    return count_lines 


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