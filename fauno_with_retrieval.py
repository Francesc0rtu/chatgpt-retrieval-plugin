from typing import List
import requests
import subprocess
import os
import tempfile
import sys
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from peft import PeftModel
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
    bearer_token = 100
    print(bearer_token)

    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {bearer_token}"
    }

    data = {
        "queries": [
            {
                "query": prompt,
                #"filter": { "document_id": "4827d5ac-2875-40ac-9279-dab0964cbf5a"},
                "top_k": 3
            }
        ]
    }
    print(data["queries"])
    response = requests.post(url=retrieval_endpoint, json=data, headers=headers)
    response_json = response.json()

    results = response_json["results"][0]["results"]

    context = []

    # Iterate over the array and extract the "text" values
    for item in results:
        context.append(item["text"])

    return context

def generate_retrieval_prompt(prompt: str, context_array: List[str], token_limit: int) -> str:
    prompt_template=f"""
<context>
"""

    limit = token_limit - len(prompt_template)
    context = "\n".join(context_array)
    token_limited_context = context[:limit]

    full_prompt = prompt_template.replace("<context>", token_limited_context)

    return full_prompt



def invoke_llama_with_context(prompt: str, token_limit: int) -> None:
    context_array = get_context(prompt)
    full_prompt = generate_retrieval_prompt(prompt, context_array, token_limit)

    print(full_prompt)

    prompt_file_path = ""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as prompt_file:
        # Write the prompt to the file
        prompt_file.write(full_prompt)
        prompt_file_path = prompt_file.name
        
    llama_cwd = os.environ.get("LLAMA_WORKING_DIRECTORY", "./llama.cpp")
    llama_cmd = os.environ.get("LLAMA_CMD", f"./main -m ./models/7B/ggml-model-q4_0.bin -c 1000")
    print("Command: " + f"{llama_cmd} -p \"{full_prompt}\"")
    # Call LLaMa with streaming responses
    process = subprocess.Popen(f"{llama_cmd} -p \"{full_prompt}\" ", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=llama_cwd)

    print(f"LLaMa process started...llama_cmd: {llama_cmd}, prompt_file_path: {prompt_file_path}, llama_cwd: {llama_cwd}")

    while True:
        # Read data from stdout and stderr streams
        stdout_data = process.stdout.readline()
        stderr_data = process.stderr.readline()

        # Check for end of stream
        if (not stdout_data) and (not stderr_data):
            break

        # Display the output
        if stdout_data:
            print(stdout_data.decode().strip())
        if stderr_data:
            print("STDERR: " + stderr_data.decode().strip())

    # Wait for the process to exit
    process.wait()



    

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def invoke_fauno_with_context(prompt:str, token_limit: int) -> None:
    context_array = get_context(prompt)
    full_prompt = generate_retrieval_prompt(prompt, context_array, 500)

    print(full_prompt)
    #CUDA check
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    prompt_file_path = ""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as prompt_file:
        # Write the prompt to the file
        prompt_file.write(full_prompt)
        prompt_file_path = prompt_file.name
    
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

    question = {
        "question": prompt,
        "context": full_prompt
    }

    answer = nlp(question)
    print(answer)




prompt = sys.argv[1]
# Note: token_limit is set to 1600 to leave room for the response from LLaMa (7B model maxes out at 2048 tokens)
# Consider specifying this as an argument to the script
# invoke_fauno_with_context(prompt, token_limit=1600)
invoke_fauno_with_context(prompt, token_limit=1600)