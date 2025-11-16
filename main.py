import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# allows the frontend to make requests to this REST API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # TODO: for production, change to be specifically the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


async def test_streaming_response():
    yield "Hello from LicenseGuard-Agent! "
    await asyncio.sleep(1.0)
    test_str = "Testing if we can stream responses to the frontend... Goodbye!"
    for word in test_str:
        yield word
        await asyncio.sleep(0.1)


@app.get("/generate/report")
async def main():
    return StreamingResponse(
        test_streaming_response(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    asyncio.run(main())
