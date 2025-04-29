MODEL_CONFIG = {
    # "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_name": "scott12355/Qwen-SophiaAI-Finetune-7B",
    "max_new_tokens": 500,
    "num_return_sequences": 1,
    "batch_size": 8,
    "max_conversation_history_size": 100
} 

RAG_CONFIG = {
    "path": "documents"
}