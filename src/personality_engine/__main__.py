import asyncio
import os

import aioboto3
import runpod
from aiobotocore.config import AioConfig as S3Config

from .personalities import load_personalities
from .handler import create_handler
from .lipsync import load_wav2lip
from .llm import load_llm
from .tts import load_xtts
from .utils.torch import device as torch_device


async def main():
	llm = await load_llm()  # FIXME
	xtts = await load_xtts()  # FIXME
	wav2lip = await load_wav2lip(torch_device)  # FIXME

	personalities = await load_personalities(xtts)

	async with aioboto3.Session().resource(
		"s3",
		endpoint_url=os.getenv("S3_ENDPOINT_URL"),
		aws_access_key_id=os.getenv("S3_KEY_ID"),
		aws_secret_access_key=os.getenv("S3_ACCESS_KEY"),
		config=S3Config(signature_version='s3v4')
	) as s3:
		bucket = os.getenv("S3_BUCKET")

		async def upload_to_s3(path: str, object_name: str):
			await s3.upload_file(path, bucket, object_name)

		def start():
			runpod.serverless.start({
				'handler': create_handler(upload_to_s3, llm, xtts, wav2lip, personalities)
			})

		await asyncio.to_thread(start)


if __name__ == '__main__':
	asyncio.run(main())
