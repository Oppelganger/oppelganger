from typing import Any

import pycuda.driver as cuda
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .chain import handler
from .chain import s3
from .handler.stage import Stage
from .utils import strict_getenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stage = Stage()


@app.post("/send_request")
def send_request(request: dict[str, Any]):
    return handler(request, stage)


@app.get("/request_status")
def request_status():
    return {"stage": stage.stage, "last_update": stage.last_update}


@app.get("/list_files")
def read_root():
    objects = []
    response = s3.list_objects_v2(Bucket=strict_getenv('S3_BUCKET_PERSONALITIES'))

    for obj in response.get('Contents', []):
        objects.append(obj['Key'])

    return objects


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