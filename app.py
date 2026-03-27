from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working ??"}

@app.get("/search")
def search(q: str):
    return {"query": q, "result": "dummy result"}