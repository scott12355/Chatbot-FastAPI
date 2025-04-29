---
title: SophiaAi
emoji: 💕
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
short_description: SophiaAi API for Synaptic Project
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# SophiaAI

## Overview
Sophia AI is designed to empower refugee women by providing accessible information and support. Built on a fine-tuned version of Qwen2.5-7B-Instruct, it uses a RAG pipeline to retrieve relevant information from a knowledge base of refugee-focused resources.


Access Sophia AI API - https://huggingface.co/spaces/scott12355/SophiaAi

*Please note that the currently hosted version will be using a alternative, very small LLM, due to cloud costs to host the fine-tuned model created for this project*

Fine tuned model - https://huggingface.co/scott12355/Qwen-SophiaAI-Finetune-7B


## Installation

### Prerequisites
- Python 3.9+
- Docker
- **To use the finetuned model a nvidia GPU is required.**

You may struggle to get this model to run on your local computer, due to memory limitations or getting your python environment to recognise your CUDA device. 

Attempt to make a python virtual environment and install the requirements.txt, if your device will not support CUDA, you may swap the model argument in [`config.py`](config.py) to the smaller qwen model, which should run on CPU bound devices given enough memory. 


 To test with smaller models that will run on CPU bound systems, adjust ```config.py```

### Local Setup
1. Clone the repository
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables in [`.env`](.env):
   ```
   SUPABASE_URL=xxx
   SUPABASE_SERVICE_ROLE_KEY=xxx
   ```
4. Run the application:
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```
5. Use the API: 
    ```
    http://localhost:7860
    ```

### Docker Deployment
```sh
docker build -t sophiaai .
docker run -p 7860:7860 sophiaai
```
This will make the API avaliable on local host port 7860. To user another port you must change the docker file. 

## API Endpoints

### `/generateFromChatHistory` (POST)
Generate a response based on conversation history.

### `/test-searchRAG` (GET)
Test the RAG system with a direct query.

### `/test-generateSingleResponse` (GET)
Generate a single response without conversation history.

### `/status` (GET)
Check if the service is running.

## Architecture
- **FastAPI**: Handles API requests and response generation
- **RAG Pipeline**: Processes documents, generates embeddings, and retrieves relevant information
- **Supabase**: Stores and manages chat histories
- **Fine-tuned LLM**: Provides contextual, empathetic responses

## Document Processing
The system processes documents from the [`documents`](documents) folder, including:
- Microsoft Word documents (.docx) 
- PDF files (.pdf)


**Word documents are recommenced to achieve the best text extraction.**

Documents are chunked, embedded, and stored in ChromaDB for efficient retrieval.


## Development
- Model configuration can be adjusted in [`config.py`](config.py)
