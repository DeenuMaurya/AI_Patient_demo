from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_patient

app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "AI Patient API is running"}


@app.post("/chat")
def chat(query: Query):
    answer = ask_patient(query.question)
    return {"answer": answer}