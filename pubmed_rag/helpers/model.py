# imports

import torch
import torch.nn.functional as F
import transformers


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    computes a single embedding (vector) for each sentence by averaging the token embeddings
    """
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def attention_pooling(model_output, attention_scores):
    token_embeddings = model_output[0]
    # Ensure attention_scores are of type float
    attention_scores = attention_scores.float()
    attention_weights = F.softmax(attention_scores, dim=-1)
    return torch.sum(token_embeddings * attention_weights.unsqueeze(-1), dim=1)


def map_pooling(pooling:str):
    """
    Retrieve the pooling function based on the given pooling type.

    This function maps a string representing the pooling type to the corresponding
    pooling function. It raises a TypeError if the input is not a string and a
    ValueError if the pooling type is not recognized.

    :param pooling: The type of pooling to be used. Must be one of 'mean_pooling' or 'attention_pooling'.
    :type pooling: str
    :raises TypeError: If the input is not a string.
    :raises ValueError: If the pooling type is not recognized.
    :return: The corresponding pooling function.
    :rtype: function

    :Example:

    >>> map_pooling('mean_pooling')
    <function mean_pooling at 0x...>
    >>> map_pooling('attention_pooling')
    <function attention_pooling at 0x...>
    """
    
    ## PRECONDITIONS
    # define options
    pooling_map = {
        'mean_pooling':mean_pooling,
        'attention_pooling':attention_pooling
    }
    if not isinstance(pooling, str):
        raise TypeError(f"pooling must be a str: {type(pooling)}")    
    if not pooling in pooling_map:
        raise ValueError(
            f"pooling of {pooling} not an option in {pooling_map.keys()}"
        )
    
    ## MAIN FUNCTION
    # retrieving pooling function
    pooling_function = pooling_map[pooling]
    return pooling_function

def get_tokens(
    tokenizer: transformers.AutoTokenizer,
    input: list,
    tokenizer_kwargs: dict = dict(
        padding=True, truncation=True, return_tensors="pt", max_length=512
    ),
) -> transformers.BatchEncoding:
    """
    Uses a tokenizer for a given model from Hugging Face to encode a list of sentences.

    :param tokenizer: Tokenizer for a given model loaded from Hugging Face
    :type tokenizer: A transformers class
    :param input: A list of sentences to be embedded
    :param type: list
    :param tokenizer_kwargs: Additional parameters to pass for

    :returns: The encoded inputs that can be used as a dictionary.
    :rtype: transformers.BatchEncoding
    """

    # PRECONDITION CHECKS
    # assert isinstance(tokenizer, transformers.AutoTokenizer), \
    #     f"Tokenizer must be from transformers: {type(tokenizer)}"
    assert isinstance(input, list), f"input must be a list of strings {input}"
    for item in input:
        assert isinstance(
            item, str
        ), f"input must be a list of strings: {item} is not str"
    assert isinstance(
        tokenizer_kwargs, dict
    ), f"tokenizer_kwargs must be a dict: {tokenizer_kwargs}"

    # MAIN FUNCTION

    # get tokens
    encoded_input = tokenizer(input, **tokenizer_kwargs)

    return encoded_input


def get_sentence_embeddings(
    model: transformers.AutoModel,
    encoded_input: transformers.BatchEncoding,
    pooling_function=attention_pooling
) -> torch.Tensor:
    """
    Uses a given model from Hugging Face to embed a list of sentences
    that have been encoded by pubmed_rag.model.get_tokens()

    :param model: The model loaded from Hugging Face
    :type model: transformers.AutoModel
    :param encoded_input: A list of sentences to be embedded
    :type encoded_input: transformers.BatchEncoding

    :returns: The embeddings
    :rtype: torch.Tensor
    """

    # PRECONDITION CHECKS
    assert isinstance(
        encoded_input, transformers.BatchEncoding
    ), f"tokens must be a transformers.BatchEncoding: {encoded_input}"

    # MAIN FUNCTION

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = pooling_function(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings
