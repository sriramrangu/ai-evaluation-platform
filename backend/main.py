
from fastapi import FastAPI

app = FastAPI(
    title="AI Evaluation Platform",
    description="RAG application with AI evaluation and Langfuse observability",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "application": "AI Evaluation Platform",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
