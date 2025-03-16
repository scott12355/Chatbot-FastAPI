from uuid import UUID
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch
from Supabase import initSupabase, updateSupabaseChat
from supabase import Client
from config import MODEL_CONFIG, RAG_CONFIG
from typing import Dict, Any, List
from fastapi.responses import JSONResponse
from api_schemas import API_RESPONSES
from VectorDB import *
from pydantic import BaseModel


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
SYSTEM_PROMPT = """Your name is SophiaAI. You should always be friendly. Use emoji in your responses. """

# Serve the API docs as our landing page
app = FastAPI(docs_url="/", title="SophiaAi Chatbot API - 21312701", version="1")
print("App Startup Complete!")

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
        output = pipe(content, num_return_sequences=1, max_new_tokens=250)

        # Extract the conversation text from the output
        generated_text = output[0]["generated_text"]
        print(generated_text)
        # Remove the system prompt from the generated text
        generated_text.pop(0)
        # Structure the response
        return {
            "status": "success",
            "generated_text": generated_text[-1],  # return only the input prompt and the generated response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        ) from e




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
                "chatID": 0
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

        # Combine system prompt with user input
        LastQuestion = input.conversationHistory[-1]["content"] # Users last question
        RAG_Results = search_docs(LastQuestion, 3)  # search Vector Database for user input.

        combined_input = f"""
        Use the following information to assist in answering the users question. Do not make anything up or guess. 
        {RAG_Results}
        If you don't know, simply let the user know. 
        Your responses will be sent directly to the user
        """
        
        content.append({"role": "system", "content": combined_input})
        # print(content)
        # Generate response
        output = pipe(content, num_return_sequences=1, max_new_tokens=250)
        generated_text = output[0]["generated_text"] # Get the entire conversation history including new generated item
        generated_text.pop(0) # Remove the system prompt from the generated text
        
        updateSupabaseChat(generated_text, input.chatID, supabase)# Update supabase
        return {
            "status": "success",
            "generated_text": generated_text # generated_text[-1],  # return only the input prompt and the generated response
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        ) from e

