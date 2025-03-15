# filepath: /Users/scotttopping/Library/CloudStorage/OneDrive-MMU/Documents/Year 4/Synaptic Project/Product/Chatbot-FastAPI/Supabase.py
def initSupabase():
    import os
    from dotenv import load_dotenv
    from supabase import create_client, Client
    
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