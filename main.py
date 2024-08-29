# import 
from pubmed_rag.utils import (
    get_basename,
    generate_log_filename,
    init_log,
    get_config_path,
    config_loader,
    assert_nonempty_keys,
    assert_nonempty_vals,

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
    logger.info(f'Configuration: {config}')
    logger.info('Complete.')


else:
    print('main.py imported. scholarly queries not initiated.')
