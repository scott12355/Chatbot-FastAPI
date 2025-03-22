from uuid import UUID
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch
from Supabase import initSupabase, updateSupabaseChatHistory, updateSupabaseChatStatus
from supabase import Client
from config import MODEL_CONFIG
from typing import Dict, Any, List
from api_schemas import API_RESPONSES
from VectorDB import *
from pydantic import BaseModel
from llama_cpp import Llama



# Pick the best available device - MPS (Mac), CUDA (NVIDIA), or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#print(device)

initRAG(device)
supabase: Client = initSupabase()


# Initialize the LLM
try:
    model = Llama.from_pretrained(
	    repo_id="Mungert/Qwen2.5-3B-Instruct-GGUF",
	    filename="Qwen2.5-3B-Instruct-bf16-q4_k.gguf",
        device=device,
        n_ctx=4096,  # Adjust the context window size (in tokens)
        temperature=0.3,
        do_sample=True, # Allow sampling to generate diverse responses. More conversational and human-like
        top_k=50, # Limit the top-k tokens to sample from
        top_p=0.95, # Limit the cumulative probability distribution for sampling
        device_map=device,
        max_new_tokens=MODEL_CONFIG["max_new_tokens"],)
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to initialize the model")

# Define the system prompt that sets the behavior and role of the LLM
SYSTEM_PROMPT = """Your name is SophiaAI. 
You are a friendly and empathetic assistant designed to empower refugee women and help with any questions.
You should always be friendly. Use emoji in all of your responses to be relatable. You may consider 😊😌🤗 
Once you have answered a question, you should check if the user would like more detail on a specific area.
"""
# Serve the API docs as our landing page
app = FastAPI(docs_url="/", title="SophiaAi - 21312701", version="1", description="SophiaAi is a Chatbot created for a university final project.\nDesigned to empower refugee women, there is a RAG pipeline containing resources to support refuges connected to a finetuned LLM.")
print("App Startup Complete!")



class ChatRequest(BaseModel):
    conversationHistory: List[Dict[str, str]]
    chatID: UUID
    model_config = {
        "json_schema_extra": {
            "example": {
                "conversationHistory": [
                    {
                        "role": "user",
                        "content": "hi"
                    },
                    {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?"
                    },
                    {
                        "role": "user",
                        "content": "whats the weather in MCR"
                    }
                ],
                "chatID": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }
    

@app.post(
    "/generateFromChatHistory",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "generated_text": {
                            "role": "assistant",
                            "content": "I don't have real-time weather data for Manchester. To get accurate information, please check a weather service like BBC Weather or the Met Office website."
                        }
                    }
                }
            }
        },
        400: API_RESPONSES[400],
        500: API_RESPONSES[500]
    }
)

async def generateFromChatHistory(input: ChatRequest):
    """
    Generate AI responses based on a given conversation history.
    Updates Supabase chat
    

    Args:
    input (ChatRequest): Structured request containing a list of previous responses"
    """
    # Notify database a response is being generated so the user cannot update the chat
    # Input validation
    if not input.conversationHistory or len(input.conversationHistory) == 0:
        raise HTTPException(status_code=400, detail="Conversation history cannot be empty")

    if len(input.conversationHistory) > MODEL_CONFIG["max_conversation_history_size"]:  # Arbitrary limit to avoid overloading LLM, adjust as needed
        raise HTTPException(status_code=400, detail="Conversation history too long")

    try:
        # Map Conversation history
        content = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        content.extend(
            {"role": message["role"], "content": message["content"]}
            for message in input.conversationHistory
        )
        updateSupabaseChatHistory(content[1:], input.chatID, supabase, True) # Update supabase

        # Combine system prompt with user input
        LastQuestion = input.conversationHistory[-1]["content"] # Users last question
        # Retrieve RAG results
        RAG_Results = search_docs(LastQuestion, 3)
        RagPrompt = f"""_RAG_
Use the following information to assist in answering the users question most recent question. Do not make anything up or guess. 
Relevant information retrieved: {RAG_Results}

If you don't know, simply let the user know, or ask for more detail. The user has not seen this message, it is for your reference only."""
        

        # Append RAG results with a dedicated role
        rag_message = {
            "role": "user",
            "content": RagPrompt
        }
        content.append(rag_message)
        
        # print(content)
        # Generate response
        output = model.create_chat_completion(content, max_tokens=MODEL_CONFIG["max_new_tokens"])
        generated_text = output["choices"][0]["message"]["content"]
        updateSupabaseChatHistory(generated_text, input.chatID, supabase)# Update supabase
        return {
            "status": "success",
            "generated_text": generated_text # generated_text[-1],  # return only the input prompt and the generated response
        }
    except Exception as e:
        updateSupabaseChatStatus(False, input.chatID, supabase)  # Notify database that an a chat isn't being processed
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        ) from e

@app.get(
    "/test-searchRAG",
    responses={
        200: {
            "description": "Successful RAG search results",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "results": [
                            {"content": "Example content 1", "metadata": {"source": "doc1.pdf"}},
                            {"content": "Example content 2", "metadata": {"source": "doc2.pdf"}}
                        ]
                    }
                }
            }
        },
        400: API_RESPONSES[400],
        500: API_RESPONSES[500]
    }
)
async def search_rag(query: str, limit: int = 3):
    """
    Search the RAG system directly with a query
    Args:
        query (str): The search query
        limit (int): Maximum number of results to return (default: 3
    Returns:
        Dict: Search results with relevant document
    Raises:
        HTTPException: If the query is invalid or search fails
    """
    # Input validation
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    if len(query) > 1000:  # Arbitrary limit
        raise HTTPException(status_code=400, detail="Query text too long")
    try:
        # Get results from vector database
        results = search_docs(query, limit)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching documents: {str(e)}"
        ) from e

@app.get(
    "/test-generateSingleResponse",
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
    if not input or not input.strip():
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
        output = pipe(content, num_return_sequences=1, max_new_tokens=MODEL_CONFIG["max_new_tokens"])

        # Extract the conversation text from the output
        generated_text = output[0]["generated_text"]
        print(generated_text)
        # Remove the system prompt from the generated text
        # Structure the response
        return {
            "status": "success",
            "generated_text": generated_text[-1],  # return only the input prompt and the generated response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        ) from e


@app.get(
    "/status",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Service is running"
                    }
                }
            }
        }
    }
)
async def status():
    """
    Check the service status
    """
    return {"status": "success", "message": "Service is running"}