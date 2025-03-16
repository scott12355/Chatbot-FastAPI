MODEL_CONFIG = {
#    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", #"Qwen/Qwen2.5-1.5B-Instruct",
    # "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "model_name": "qwen/Qwen2.5-1.5B-Instruct",
    "max_new_tokens": 250,
    "num_return_sequences": 1,
    "batch_size": 8,
    "max_conversation_history_size": 10
} 

RAG_CONFIG = {
    "path": "documents"
}