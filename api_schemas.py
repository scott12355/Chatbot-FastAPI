API_RESPONSES = {
    200: {
        "description": "Successful response",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "generated_text": [
                        {"role": "user", "content": "hey"},
                        {
                            "role": "assistant",
                            "content": "Hello! How can I assist you today?",
                        },
                    ],
                }
            }
        },
    },
    400: {
        "description": "Invalid input",
        "content": {
            "application/json": {"example": {"detail": "Input text cannot be empty"}}
        },
    },
    500: {
        "description": "Server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Error generating response: Model failed to generate"
                }
            }
        },
    },
}
