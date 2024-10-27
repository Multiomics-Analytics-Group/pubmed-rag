# import

import requests

from pubmed_rag.helpers.utils import (
    assert_nonempty_keys,
    assert_nonempty_vals,
    config_loader,
    get_args,
    get_logger,
)
from pubmed_rag.run_search import find_similar_vectors


def init_prompt(
    query: str,
    results: list,
    role: str = """
    You are a cautious AI assistant. You are able to find concise answers to the questions from the contextual passage snippets provided and their affiliated pubmed articles.
    Please check the context information carefully and do not use information that is not relevant to the question.
    If the retrieved context does not provide useful information to answer the question, say that you do not know.
    """,
    task: str = """
    Use the following pieces of information enclosed in <context> tags sourced from pubmed ids enclosed in <pmid> to provide an answer to the question enclosed in <question> tags.
    """,
) -> list:
    """
    TODO: complete docstring ..
        Thank you to the following resources:
        - PanKB (B Sun, L Pashkova, PA Pieters, AS Harke, OS Mohite, BO Palsson, PV Phaneuf
    bioRxiv 2024.08.16.608241; doi: https://doi.org/10.1101/2024.08.16.608241)
        - https://www.llama.com/docs/how-to-guides/prompting
    """

    # prepare question
    question = f"<question>{query}</question>"

    # prepare context section
    the_context = ""
    metadata = [x["entity"] for x in results]
    for data in metadata:
        the_context += f"""<context>{data['text']}</context> <pmid>{str(data['pmid'])}</pmid>
        """
    # putting it together
    user_content = task + question + the_context
    # in format for llama
    system_prompt = {"role": "system", "content": role}
    user_prompt = {"role": "user", "content": user_content}
    prompt = [system_prompt, user_prompt]

    ## POST CONDITIONS
    assert isinstance(prompt, list), "Was unable to save prompt as a list of dicts"
    for x in prompt:
        assert isinstance(x, dict), f"could not save x as a dict: {x}"

    return prompt


def llama3(
        prompt: list, 
        api: str = "http://localhost:11434/api/chat", 
        model: str = "llama3.1",
        num_ctx: int = 4096  # Added parameter for context length
    ) -> str:
    """
    Getting response from llama3 LLM
    https://www.llama.com/docs/llama-everywhere/running-meta-llama-on-mac/

    :param prompt: A list of dictionaries with the prompts to the model
    :type prompt: list
    :param model: The name of the llama LLM version
    :type model: str
    :param num_ctx: The context length for handling larger prompts
    :type num_ctx: int
    :return: The LLM's response
    :rtype: str
    """
    data = {
        "model": model,
        "messages": prompt,
        "options": {
            "num_ctx": num_ctx  # Include the context length in the request
        },
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(api, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise ConnectionError(
            "There was an issue with the model and/or API. Please check Ollama."
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
