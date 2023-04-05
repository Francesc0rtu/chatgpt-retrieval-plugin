# LLaMA retrieval plugin

**Checkout our [blog post about this project here](https://blog.lastmileai.dev/using-openais-retrieval-plugin-with-llama-d2e0b6732f14)**

> **Try out https://lastmileai.dev/ for experimenting with foundation models**

## Introduction

The LLaMA Retreival Plugin repository shows how to use a similar structure to the [chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin) for augmenting the capabilities of the LLaMA large language model using a similar grounding technique. This provides a starting point for sharing plugins between LLMs, regardless of the capabilities of actual model (note that results will vary widely depending on model used).

For a full introduction to the structure of this repository, check the [ChatGPTREADME.md file](https://github.com/lastmile-ai/llama-retrieval-plugin/blob/main/ChatGPTREADME.md). This follows the same structure as the openai plugin to better support cross compatibility between different LLMs.

## Table of Contents

- [Quickstart](#quickstart)
  - [Setting up plugin server](#setting-up-plugin-server)
  - [Setup LLaMA.cpp](#setup-llamacpp)
  - [Incorporate retrieval plugin into llama](#incorporate-retrieval-plugin-into-llama)
- [About](#about)
  - [Plugins](#plugins)
  - [Retrieval Plugin](#retrieval-plugin)
  - [Memory Feature](#memory-feature)
- [Limitations](#limitations)
- [Future Directions](#future-directions)
- [Contributors](#contributors)

## Quickstart

### Setting up plugin server

This section is deliberately similar to the chatgpt-retrieval-plugin quickstart guide with some opinionated defaults on using conda & pinecone for a simpler getting started experience.

To simplify setting up the plugin environment, this repo uses [conda](https://www.anaconda.com/products/distribution) to create a python virtual environment, then follows the standard chatgpt plugin approach to install dependencies via poetry.

```shell
git clone https://github.com/lastmile-ai/llama-retrieval-plugin
cd ./llama-retrieval-plugin
conda env create -f environment.yml
conda activate llama-retrieval-plugin
poetry install
```

Set the required Environment variables (note that we only used pinecone during testing, however any of the chatgpt repo's datastores should work with their correct environment vars - see the [ChatGPTREADME.md](https://github.com/lastmile-ai/llama-retrieval-plugin/blob/main/ChatGPTREADME.md) file for more info):
```
export DATASTORE=pinecone
export BEARER_TOKEN=your_bearer_token_here
export OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment_aws_gcp
PINECONE_INDEX=your_pinecone_index_name
```

A few additional notes:
* This repo is still using Openai's embedding creation endpoint, however it should be able to migrate to a different embedding model to remove that dependency in the future.

Next, run the API locally:
```
poetry run start
```

You can still go to the API documentation at `http://0.0.0.0:8000/docs`.

### Setup LLaMA.cpp

Clone the llama.cpp repository into a separate folder:

```
git clone https://github.com/ggerganov/llama.cpp
cd ./llama.cpp
```

Follow the usage guide from [llama.cpp's README](https://github.com/ggerganov/llama.cpp#usage), also copied here:
```
# build this repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
python3 -m pip install torch numpy sentencepiece

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
python3 quantize.py 7B

# run the inference & test that it works
./main -m ./models/7B/ggml-model-q4_0.bin -n 128
```

Note that you will need to obtain the LLaMA model weights and place them in the `./models` folder. Refer to [Facebook's LLaMA repository](https://github.com/facebookresearch/llama/pull/73/files) if you need to request access to the model data.

### Incorporate retrieval plugin into llama

Go back to the root of this project's folder:
```
cd /path/to/llama-retrieval-plugin
```

First, add some data to the retrieval datastore using the upsert api endpoint via the docs `http://0.0.0.0:8000/docs` page (you will need to authenticate using the bearer token you specified on that swagger ui page).

Specify the directory that you put llama.cpp into (and tested that it works) and ensure that the `llama-retrieval-plugin` conda venv is set in your shell:
```
export LLAMA_WORKING_DIRECTORY="/my/working/dir/to/llama.cpp"
conda activate llama-retrieval-plugin
```

Then use the following command to run the llama_with_retrieval.py file:
```
python3 llama_with_retrieval.py
```

If everything was setup, you will have llama utilize the context from the document(s) that you uploaded! 🚀

## About

### Plugins

Plugins are chat extensions designed specifically for language models like ChatGPT, enabling them to access up-to-date information, run computations, or interact with third-party services in response to a user's request. They unlock a wide range of potential use cases and enhance the capabilities of language models.

Developers can create a plugin by exposing an API through their website and providing a standardized manifest file that describes the API. ChatGPT consumes these files and allows the AI models to make calls to the API defined by the developer.

A plugin consists of:

- An API
- An API schema (OpenAPI JSON or YAML format)
- A manifest (JSON file) that defines relevant metadata for the plugin

The Retrieval Plugin already contains all of these components. Read the Chat Plugins blogpost [here](https://openai.com/blog/chatgpt-plugins), and find the docs [here](https://platform.openai.com/docs/plugins/introduction).

### Retrieval Plugin

This is a plugin for ChatGPT that enables semantic search and retrieval of personal or organizational documents. It allows users to obtain the most relevant document snippets from their data sources, such as files, notes, or emails, by asking questions or expressing needs in natural language. Enterprises can make their internal documents available to their employees through ChatGPT using this plugin.

The plugin uses OpenAI's `text-embedding-ada-002` embeddings model to generate embeddings of document chunks, and then stores and queries them using a vector database on the backend. As an open-source and self-hosted solution, developers can deploy their own Retrieval Plugin and register it with ChatGPT. The Retrieval Plugin supports several vector database providers, allowing developers to choose their preferred one from a list.

A FastAPI server exposes the plugin's endpoints for upserting, querying, and deleting documents. Users can refine their search results by using metadata filters by source, date, author, or other criteria. The plugin can be hosted on any cloud platform that supports Docker containers, such as Fly.io, Heroku or Azure Container Apps. To keep the vector database updated with the latest documents, the plugin can process and store documents from various data sources continuously, using incoming webhooks to the upsert and delete endpoints. Tools like [Zapier](https://zapier.com) or [Make](https://www.make.com) can help configure the webhooks based on events or schedules.

### Memory Feature

A notable feature of the Retrieval Plugin is its capacity to provide ChatGPT with memory. By utilizing the plugin's upsert endpoint, ChatGPT can save snippets from the conversation to the vector database for later reference (only when prompted to do so by the user). This functionality contributes to a more context-aware chat experience by allowing ChatGPT to remember and retrieve information from previous conversations. Learn how to configure the Retrieval Plugin with memory [here](/examples/memory).

## Limitations

Including all the limitations of the [ChatGPT plugin](https://github.com/lastmile-ai/llama-retrieval-plugin/blob/main/ChatGPTREADME.md#limitations), using llama as the LLM for a plugin has some additional limitations since it has not been trained to parse openapi specs [like ChatGPT has](https://platform.openai.com/docs/plugins/introduction).

- **Openapi schema parsing**: Instead of parsing the openapi schema, the llama_with_retrieval.py simply expects a `/query` endpoint to retrieve additional context to ground the LLM.
- **Performance**: LLaMA has multiple model sizes and utilizing the 7B model is better for speed, however may be worse for prompt responses than the 65B model. Both will generally perform worse than gpt-turbo-3.5 or gpt-4 models.

## Future Directions

The LLaMA retrieval plugin provides a proof of concept that LLM plugins can be used across models. There is still substantial work needed to productionize this system to generalize better for other models like [Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html), [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), or other future LLMs.

We are looking to collaborate with others in helping define how LLM plugins can be standardized in the future. Feel free to reach out if you would like to collaborate by opening an issue in this repo.

## Contributors

We want to thank OpenAI & the entire ChatGPT plugins team along with everyone in the repo this was [forked from](https://github.com/openai/chatgpt-retrieval-plugin#contributors). Also want to thank the [Llama.cpp](https://github.com/ggerganov/llama.cpp) and the [LLaMA LLM](https://github.com/facebookresearch/llama) model from Facebook.

- [saqadri](https://github.com/saqadri) 
- [Flux159](http://github.com/Flux159)

## TODO
[] switch from llama.cpp to camoscio 
[] build interface
[] get embedding from llama instead of bert
[]
