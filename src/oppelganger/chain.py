import boto3
from os import getenv

from .handler import create_handler
from .lipsync import load_wav2lip
from .llm import load_llm
from .tts import load_xtts
from .utils.torch import device as torch_device

llm = load_llm()
xtts = load_xtts()
wav2lip = load_wav2lip(torch_device)

s3 = boto3.Session().client(
    "s3",
    endpoint_url=getenv("S3_ENDPOINT_URL"),
    region_name=getenv("S3_REGION_NAME"),
    aws_access_key_id=getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=getenv("S3_SECRET_ACCESS_KEY"),
    aws_session_token=getenv("S3_SESSION_TOKEN"),
)

handler = create_handler(s3, llm, xtts, wav2lip)
