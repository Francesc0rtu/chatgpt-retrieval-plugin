from typing import List
from typing import Union, List


from tenacity import retry, wait_random_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer



@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]):
    """
    Embed texts using LLM model.
    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    """
    # Log the call of the embedding 
    print("Calling OpenAI API to get embeddings...")

    return create_embedding(texts)

def create_embedding(inputs: List[str] ):
    """
    Embed texts using LLM model.
    The model is loaded from the HuggingFace model hub.
    Args:
        texts: The list of texts to embed.
        
    Returns:    
        A list of embeddings, each of which is a list of floats."""

    # model for embedding: It will be better to use a llama model, maybe the same
    # used for generating the question "teelinsan/camoscio-7b-llama"
    model_name = "efederici/mmarco-sentence-BERTino"   
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(inputs)
    return embeddings



