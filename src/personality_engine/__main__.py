from os import getenv
from typing import Callable, Any

import boto3
import runpod
from mypy_boto3_s3.client import S3Client

from .handler import create_handler
from .lipsync import load_wav2lip
from .llm import load_llm
from .tts import load_xtts
from .utils.torch import device as torch_device


def initialize_handler(s3: S3Client) -> Callable[..., Any]:
	llm = load_llm()
	xtts = load_xtts()
	wav2lip = load_wav2lip(torch_device)

	return create_handler(s3, llm, xtts, wav2lip)


def main():
	with boto3.Session().client(
		"s3",
		endpoint_url=getenv("S3_ENDPOINT_URL"),
		region_name=getenv("S3_REGION_NAME"),
		aws_access_key_id=getenv("S3_ACCESS_KEY_ID"),
		aws_secret_access_key=getenv("S3_SECRET_ACCESS_KEY"),
		aws_session_token=getenv("S3_SESSION_TOKEN"),
	) as s3:
		handler = initialize_handler(s3)
		runpod.serverless.start({
			'handler': handler
		})


if __name__ == '__main__':
	main()
