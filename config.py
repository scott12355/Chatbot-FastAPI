MODEL_CONFIG = {
#    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", #"Qwen/Qwen2.5-1.5B-Instruct",
    # "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "model_name": "Mungert/Qwen2.5-3B-Instruct-GGUF",
    "max_new_tokens": 350,
    "num_return_sequences": 1,
    "batch_size": 8,
    "max_conversation_history_size": 100
} 

RAG_CONFIG = {
    "path": "documents"
}