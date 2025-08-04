class Config:
    SEED = 42
    OLLAMA_CONTEXT_WINDOW = 4096

    class Server:
        HOST = "0.0.0.0"
        PORT = 8000
        SSE_PATH = "/sse"
        TRANSPORT = "sse"

    class Agent:
        MAX_ITERATIONS = 10