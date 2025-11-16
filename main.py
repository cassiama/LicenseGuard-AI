from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/")
def main():
    print("Hello from agent!")


if __name__ == "__main__":
    main()
