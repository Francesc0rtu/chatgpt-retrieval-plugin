from typing import List
import requests
import subprocess
import os
import tempfile
import sys
import torch

from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_context(prompt: str) -> List[str]:
    """
    Queries the data store using the retrieval plugin to get relevant context.

    Args:
        prompt: The user prompt to identify context for.

    Returns:
        A list of document chunks from the data store, sorted by proximity of vector similarity.
    """

    retrieval_endpoint = os.environ.get("DATASTORE_QUERY_URL", "http://0.0.0.0:8000/query")
    # get the bearer token from the environment
    bearer_token = os.environ.get("BEARER_TOKEN")

    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {bearer_token}"
    }

    data = {
        "queries": [
            {
                "query": prompt,
                "top_k": 3
            }
        ]
    }
    response = requests.post(url=retrieval_endpoint, json=data, headers=headers)
    response_json = response.json()

    results = response_json["results"][0]["results"]

    context = []

    # Iterate over the array and extract the "text" values
    for item in results:
        context.append(item["text"])

    return context

def generate_retrieval_prompt_Camoscio(prompt: str, context_array: List[str], token_limit: int) -> str:
    prompt_template=f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{prompt}

### Input:
<context>

### Risposta:"""
    limit = token_limit - len(prompt_template)
    context = "\n".join(context_array)
    token_limited_context = context[:limit]

    full_prompt = prompt_template.replace("<context>", token_limited_context)

    return full_prompt



from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def invoke_camoscio_with_context(prompt: str, token_limit: int) -> None:

    # get the context to attach to the prompt
    context_array = get_context(prompt)

    # generate the full prompt with the context and the question
    full_prompt = generate_retrieval_prompt_Camoscio(prompt, context_array, token_limit)

    # check if GPU is available
    assert torch.cuda.is_available(), "Change the runtime type to GPU"
    device = "cuda"

    # load the model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    model = LlamaForCausalLM.from_pretrained(
                        "decapoda-research/llama-7b-hf",
                        load_in_8bit=True,
                        device_map="auto",
                        )

    # load the model peft to optimize the generation using the 8bit compression
    model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-llama")

    generation_config = GenerationConfig(
    temperature=0.2,     # change the temperature to get more or less random results
    top_p=0.75,
    top_k=40,
    num_beams=4,
    )

    # tokenize the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # generate the output using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids = input_ids,
            generation_config = generation_config,
            max__new_tokens = 2048,
            ) # type: ignore
    
    # decode the output
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # split Risposta
    output = output.split("### Risposta:")[1].strip()
    return output
        
def invoke_bert_with_context(prompt: str, token_limit:int):
     # get the context to attach to the prompt
    context_array = get_context(prompt)

    # generate the full prompt with the context and the question
    context = "\n".join(context_array)
    context = context[:token_limit]
    # check if GPU is available
    print("context retrieved")
    from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
    
    model_name = "deepset/roberta-base-squad2"
    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
    'question': prompt,
    'context': context
    }
    res = nlp(QA_input)

    print(res)

#prompt = "How do I activate Conda for my project?"
prompt = sys.argv[1]

# Note: token_limit is set to 1600 to leave room for the response from LLaMa (7B model maxes out at 2048 tokens)
# Consider specifying this as an argument to the script
# invoke_camoscio_with_context(prompt, token_limit=1600)
invoke_bert_with_context(prompt, token_limit=1600)