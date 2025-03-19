import os
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import HTTPException
from typing import List, Dict
    
def initSupabase():
    # Load environment variables from .env file
    load_dotenv()
    
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    # Add error checking
    if not url or not key:
        raise ValueError("Supabase URL and key must be set in environment variables")
        
    supabase: Client = create_client(url, key)
    print("Supabase client initialized")
    return supabase


def updateSupabaseChatHistory(generated_text: List[Dict[str, str]], chat_id: int, supabase: Client, status: bool = False):
    """
    Updates the chat history in Supabase.

    Args:
        generated_text: The generated text to add to the chat history.
        chat_id: The ID of the chat to update.

    Raises:
        HTTPException: If there is an error updating Supabase.
    """
    try:
        response = supabase.table("Chats").update({"chat_history": generated_text, "awaiting_response": status}).eq("id", chat_id).execute()
        if hasattr(response, 'error') and response.error:
            raise HTTPException(
                status_code=500, detail=f"Error updating chat history: {response.error}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating Supabase: {str(e)}"
        ) from e
        
def updateSupabaseChatStatus(status: bool, chat_id: int, supabase: Client):
    """
    Updates the status of a chat in Supabase.

    Args:
        status: The status to update the chat to.
        chat_id: The ID of the chat to update.

    Raises:
        HTTPException: If there is an error updating Supabase.
    """
    try:
        response = supabase.table("Chats").update({"awaiting_response": status}).eq("id", chat_id).execute()
        if hasattr(response, 'error') and response.error:
            raise HTTPException(
                status_code=500, detail=f"Error updating chat status: {response.error}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating Supabase: {str(e)}"
        ) from e