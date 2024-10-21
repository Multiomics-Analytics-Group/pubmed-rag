# import

import requests

from pubmed_rag.utils import (
    assert_nonempty_keys,
    assert_nonempty_vals,
    config_loader,
    get_args,
    get_logger,
)
from run_search import find_similar_vectors


def init_prompt(query: str, results: list) -> list:
    """
        Thank you to the following resources:
        - PanKB (B Sun, L Pashkova, PA Pieters, AS Harke, OS Mohite, BO Palsson, PV Phaneuf
    bioRxiv 2024.08.16.608241; doi: https://doi.org/10.1101/2024.08.16.608241)
        - https://www.llama.com/docs/how-to-guides/prompting
    """

    restrictions = """You can only use academic papers from pubmed or biorxiv or google scholar to support your answer. """
    role = """
    You are a biological/biomedical Knowledge graph (KG) LLM. You are a cautious assistant proficient in creating knowledge graphs for biological and biomedical use cases. 
    You are able to find answers to the questions from the contextual passage snippets provided and their affiliated pubmed articles.
    Please check the context information carefully and do not use information that is not relevant to the question.
    If the retrieved context does not provide useful information to answer the question, say that you do not know.
    """

    the_task = """
    Use the following pieces of information enclosed in <context> tags sourced from pubmed ids enclosed in <pmid> to provide an answer to the question enclosed in <question> tags.
    Give priority to the context in descending order.
    Please return the context quotes (and their pmids) that you used to support your answer.
    """

    question = f"<question>{query}</question>"

    the_context = ""
    metadata = [x["entity"] for x in results]

    for data in metadata:
        the_context += f"""<context>{data['text']}</context> <pmid>{str(data['pmid'])}</pmid>
        """

    system_content = f"{role} Restrictions: {restrictions}"
    user_content = the_task + question + the_context

    system_prompt = {"role": "system", "content": system_content}

    user_prompt = {"role": "user", "content": user_content}

    prompt = [system_prompt, user_prompt]

    ## POST CONDITIONS
    assert isinstance(prompt, list), "Was unable to save prompt as a list of dicts"
    for x in prompt:
        assert isinstance(x, dict), f"could not save x as a dict: {x}"

    return prompt


def llama3(prompt: list, model="llama3.1") -> str:
    """
    getting response from llama3 LLM
    https://www.llama.com/docs/llama-everywhere/running-meta-llama-on-mac/

    :param prompt: A list of dictionaries with the prompts to the model
    :type prompt: list
    :param model: The name of the llama LLM version
    :type model: str

    :param return: The LLMs response?
    :rtype: str
    """
    data = {
        "model": model,
        "messages": prompt,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(llama_api, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise ConnectionError(
            "There was an issue with the model and/or api. Please check ollama."
        )


if __name__ == "__main__":
    ## GET ARGS
    # init
    args = get_args(
        prog_name="run_search", others=dict(description="queries the vector db")
    )

    # ## START LOG FILE
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
    llama_model = config["llama model"]
    llama_api = config["llama api"]
    logger.info(f"Configuration: {config}")

    # ## MAIN
    # getting context from vector database
    logger.info("Embedding question and searching vector database...")
    result = find_similar_vectors(
        path_to_config=args.config,
        query=args.query,
        logging=False,
    )

    # preparing the prompt
    prompt = init_prompt(args.query, result)
    logger.info(f"The prompt going to the LLM: {prompt}")

    # prompt to the LLM
    response = llama3(
        prompt=prompt,
        model=llama_model,
    )
    logger.info(f"The response: {response}")

    logger.info("Complete.")

else:
    print("Warning: prep_prompt.py imported as a module. Script not ran.")
