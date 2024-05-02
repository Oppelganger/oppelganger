import uvicorn
from os import getenv
from typing import Callable, Any

import boto3
from mypy_boto3_s3.client import S3Client

from .handler import get_active_status
from .handler import create_handler
from .lipsync import load_wav2lip
from .llm import load_llm
from .tts import load_xtts
from .utils.torch import device as torch_device

import pycuda.driver as cuda
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_handler() -> Callable[..., Any]:
    s3: S3Client = boto3.Session().client(
        "s3",
        endpoint_url=getenv("S3_ENDPOINT_URL"),
        region_name=getenv("S3_REGION_NAME"),
        aws_access_key_id=getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=getenv("S3_SECRET_ACCESS_KEY"),
        aws_session_token=getenv("S3_SESSION_TOKEN"),
    )
    llm = load_llm()
    xtts = load_xtts()
    wav2lip = load_wav2lip(torch_device)

    return create_handler(s3, llm, xtts, wav2lip)


@app.post("/send_request")
async def send_request(request: dict[str, Any]):
    return handler(request)


@app.get("/request_status")
def request_status():
    return get_active_status()


@app.get("/about")
def read_root():
    return {"Hello": "World"}


@app.get("/check_gpu_driver")
def check_gpu_driver():
    supported = True

    # Проверяем наличие видеокарты и работу драйверов NVIDIA
    try:
        cuda.init()
        device_count = cuda.Device.count()
        if device_count != 0:
            return {
                "message": f"Видеокарта поддерживается: {str(cuda.Device(0).name())}",
                "supported": supported
            }
        return {
            "message": "Нет доступных CUDA-устройств",
            "supported": False
        }
    except cuda.NVIDIAError as e:
        return {
            "message": f"Не удалось проверить работу видеокарты: {str(e)}",
            "supported": False
        }


if __name__ == "__main__":
    handler = initialize_handler()
    uvicorn.run(app=app, host="0.0.0.0", port=8080)
