from fastapi import FastAPI, HTTPException
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from config import MODEL_CONFIG, RAG_CONFIG
from typing import Dict, Any
from fastapi.responses import JSONResponse
import os
from PyPDF2 import PdfReader
import chromadb
from api_schemas import API_RESPONSES
from VectorDB import *

# Pick the best available device - MPS (Mac), CUDA (NVIDIA), or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

initRAG(device)


# print(search_docs("how much employment in manchester"))

# Initialize the LLM
try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_CONFIG["model_name"],
        device=device,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to initialize the model")

# Define the system prompt that sets the behavior and role of the LLM
SYSTEM_PROMPT = """You are a helpful AI assistant to educate people on the city of Manchester in England"""

# Serve the API docs as our landing page
app = FastAPI(docs_url="/", title="21312701 - Chatbot Prof of Concept", version="1")


@app.get(
    "/generateSingleResponse",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "generated_text": [
                            {
                                "role": "user",
                                "content": "hey"
                            },
                            {
                                "role": "assistant",
                                "content": "Hello! How can I assist you today? Is there something specific you'd like to talk about or learn more about?"
                            }
                        ]
                    }
                }
            }
        },
        400: API_RESPONSES[400],
        500: API_RESPONSES[500]
    }
)

async def generateSingleResponse(input: str):
    """
    Generate AI responses.

    Args:
        input (str): The user's question or prompt

    Returns:
        Dict[str, str]: Structured response containing the generated text

    Raises:
        HTTPException: If input is invalid or generation fails
    """
    # Input validation
    if not input or len(input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    if len(input) > 1000:  # Arbitrary limit, adjust as needed
        raise HTTPException(status_code=400, detail="Input text too long")

    # search Vector Database for user input.
    RAG_Results = search_docs(input, 3) 
    # print(RAG_Results)

    combined_input = f"""
     Here is the users questions: {input}.
     
    Use the following information to assist in answering the users question. Do not make anything up or guess. 
    If you don't know, simply let the user know. 
    {RAG_Results}
    """

    try:
        # Combine system prompt with user input
        content = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_input},
        ]

        # Generate response
        output = pipe(content, num_return_sequences=1, max_new_tokens=250)

        # Extract the generated text from the output

        generated_text = output[0]["generated_text"]
        print(generated_text)
        # Structure the response
        return {
            "status": "success",
            "generated_text": generated_text[-1],  # return only the input prompt and the generated response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )
