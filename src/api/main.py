from fastapi import FastAPI

app = FastAPI()


@app.post("/segment-clouds")
async def root():
    return {"message": "Hello World"}