import asyncio
import uuid
from functools import partial, wraps

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from wav2lip import wav2lip as sync_wav2lip


def awaitable(func):
	@wraps(func)
	async def run(*args, loop=None, executor=None, **kwargs):
		if loop is None:
			loop = asyncio.get_running_loop()
		fn = partial(func, *args, **kwargs)
		return await loop.run_in_executor(executor, fn)

	return run


wav2lip = awaitable(sync_wav2lip)

app = FastAPI()


class Request(BaseModel):
	audio_path: str
	video_path: str
	enhance: bool
	female: bool


@app.post("/", response_class=PlainTextResponse)
async def generate(request: Request):
	out = f"/result/{uuid.uuid4()}.mp4"
	await wav2lip(
		request.audio_path,
		request.video_path,
		out,
		request.enhance,
		request.female
	)
	return out
