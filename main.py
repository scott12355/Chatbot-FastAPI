from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch
from config import MODEL_CONFIG
from typing import Dict, Any
from fastapi.responses import JSONResponse

# Pick the best available device - MPS (Mac), CUDA (NVIDIA), or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# Initialize the model with error handling
try:
    pipe = pipeline(
        "text-generation", 
        model=MODEL_CONFIG["model_name"], 
        device=device, 
        batch_size=MODEL_CONFIG["batch_size"]
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to initialize the model")

# Define the system prompt that sets the behavior and role of the LLM
SYSTEM_PROMPT = """You are an AI assistant dedicated to empowering refugee women by providing them with accurate, supportive, and culturally sensitive information. Your goal is to help them navigate challenges related to education, employment, legal rights, healthcare, mental well-being, and social integration.
Your responses should always be:
Empathetic & Encouraging: Acknowledge hardships while fostering resilience and self-confidence.
Actionable & Practical: Offer clear steps, trusted resources, and local support options where possible.
Culturally Aware & Inclusive: Respect diverse backgrounds, traditions, and sensitivities.
Safe & Ethical: Avoid legal, medical, or financial advice unless citing verified sources. Always prioritize user safety and well-being.
If a question involves sensitive topics such as legal asylum processes, domestic violence, or urgent medical concerns, direct users to relevant professional organizations or helplines in their country. When discussing education, jobs, or financial independence, focus on accessible opportunities, online learning, remote work, and community support networks.
Above all, inspire confidence, self-sufficiency, and hope in every response."""

# Serve the API docs as our landing page
app = FastAPI(docs_url="/",
              title="21312701 - Chatbot Prof of Concept",
              version="1")

@app.get("/generate", 
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
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Input text cannot be empty"
                    }
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error generating response: Model failed to generate"
                    }
                }
            }
        }
    }
)
async def generate(input: str) -> Dict[str, str]:
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
    
    try:
        # Combine system prompt with user input
        content = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input}
        ]
        
        # Generate response
        output = pipe(content, num_return_sequences=1, max_new_tokens=250)
        
        # Structure the response
        return {
            "status": "success",
            "generated_text": output[0]["generated_text"],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )
    
