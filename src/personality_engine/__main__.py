import asyncio
import os
from typing import Callable, Awaitable, Any

import aioboto3
import runpod

from .handler import create_handler
from .lipsync import load_wav2lip
from .llm import load_llm
from .personalities import load_personalities
from .tts import load_xtts
from .utils.torch import device as torch_device


async def initialize_handler() -> Callable[..., Awaitable[Any]]:
	llm = await load_llm()  # FIXME
	xtts = await load_xtts()  # FIXME
	wav2lip = await load_wav2lip(torch_device)  # FIXME

	personalities = await load_personalities(xtts)

	async with aioboto3.Session().client(
		"s3",
		endpoint_url=os.getenv("S3_ENDPOINT_URL"),
		region_name=os.getenv("S3_REGION_NAME"),
		aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
		aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
		aws_session_token=os.getenv("S3_SESSION_TOKEN"),
	) as s3:
		bucket = os.getenv("S3_BUCKET")

		async def upload_to_s3(path: str, object_name: str):
			await s3.upload_file(path, bucket, object_name)

		return create_handler(upload_to_s3, llm, xtts, wav2lip, personalities)


if __name__ == '__main__':
	handler = asyncio.run(initialize_handler())
	runpod.serverless.start({
		'handler': handler
	})
