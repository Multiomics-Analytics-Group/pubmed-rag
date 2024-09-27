# imports
import os, time, json, requests 
from pubmed_rag.utils import assert_path, get_chunks
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_tokens(
        model_name:str, 
        input:list,
        tokenizer_kwargs:dict
        )->torch.Tensor:
    """
    Uses a tokenizer for a given model from Hugging Face to embed a list of sentences.

    :param model_name: The model name from Hugging Face
    :type model_name: str
    :param input: A list of sentences to be embedded 
    :param type: list
    :param tokenizer_kwargs: Additional parameters to pass for
    
    :returns: The encoded inputs as a tensor
    :rtype: torch.Tensor
    """

    # PRECONDITION CHECKS
    assert isinstance(model_name, str), f"model_name must be a str: {model_name}"
    assert isinstance(input, list), f"input must be a list of strings {input}"
    for item in input:
        assert isinstance(item, str), \
            f"input must be a list of strings: {item} is not str"
    assert isinstance(tokenizer_kwargs, dict), \
        f"tokenizer_kwargs must be a dict: {tokenizer_kwargs}"    
    