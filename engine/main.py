import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import aiofiles.os
import aiohttp
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from dotmap import DotMap
from fastapi import FastAPI
from llama_cpp import Llama, ChatCompletionRequestMessage
from pydantic import BaseModel

app = FastAPI()

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Torch device {torch_device}')

mixtral = Llama(
	model_path="./models/mixtral_7bx2_moe.Q5_K_M.gguf",
	chat_format="llama-2",
	n_gpu_layers=-1,
	seed=-1,
	n_threads=16,
	n_threads_batch=16,
	offload_kqv=True,
	numa=True,
	use_mlock=True,
)

model_manager = ModelManager()

tts_model_path, _, _ = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

tts_config = XttsConfig()
tts_config.load_json(f"{tts_model_path}/config.json")
tts = Xtts.init_from_config(tts_config)
tts.load_checkpoint(
	tts_config,
	checkpoint_dir=tts_model_path,
	vocab_path=f"{tts_model_path}/vocab.json",
	use_deepspeed=False
)

if torch.cuda.is_available():
	tts = tts.cuda()


def read_personalities():
	out = []
	for path in Path("/personalities").glob("*/"):
		with open(path / "personality.json") as meta:
			personality = DotMap(json.load(meta))
			personality.path = path

			voices = [str(path / audio) for audio in personality.sample_audio]
			gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(
				audio_path=voices,
				gpt_cond_len=tts.config.gpt_cond_len,
				max_ref_length=tts.config.max_ref_len,
				sound_norm_refs=tts.config.sound_norm_refs
			)

			personality.gpt_cond_latent = gpt_cond_latent.to(torch_device)
			personality.speaker_embedding = speaker_embedding.to(torch_device)

			out.append(personality)
	return out


personalities = read_personalities()

persons = {person.command.name: person for person in personalities}


def gtime():
	now = datetime.now()
	return now.strftime("%H:%M:%S")


def awaitable(func):
	@wraps(func)
	async def run(*args, loop=None, executor=None, **kwargs):
		if loop is None:
			loop = asyncio.get_running_loop()
		fn = partial(func, *args, **kwargs)
		return await loop.run_in_executor(executor, fn)
	return run


@awaitable
def create_chat_completion(messages: List[ChatCompletionRequestMessage]):
	result = mixtral.create_chat_completion(messages, max_tokens=100)
	return DotMap(result)


@awaitable
def tts_inference(
	text,
	gpt_cond_latent,
	speaker_embedding,
	path: str | os.PathLike
):
	res = tts.inference(
		text,
		"en",
		gpt_cond_latent,
		speaker_embedding,
		speed=1.1,
		enable_text_splitting=True,
		temperature=tts.config.temperature,
		length_penalty=tts.config.length_penalty,
		repetition_penalty=tts.config.repetition_penalty,
		top_k=tts.config.top_k,
		top_p=tts.config.top_p,
	)
	# noinspection PyUnresolvedReferences
	torchaudio.save(str(path), torch.tensor(res["wav"]).unsqueeze(0), 24000)


class GenerateRequest(BaseModel):
	person: str
	text: str


@app.get("/persons", response_model=dict[str, str])
async def get_persons():
	return {p.command.name: p.command.desc for p in personalities}


@app.post("/generate")
async def post_generate(req: GenerateRequest):
	print(req.text)
	user_input = req.text

	personality = persons[req.person]

	messages = [
		{"role": "system", "content": personality.prompt},
		{"role": "system", "content": "Please write text in less than 20 words"},
		{"role": "user", "content": user_input},
	]

	print(f"{gtime()}: llm start")
	result = await create_chat_completion(messages)
	print(f"{gtime()}: llm end")
	text = result.choices[0].message.content

	out = Path(f"/share/{uuid.uuid4()}")
	await aiofiles.os.mkdir(out)

	video = personality.path / random.choice(personality.sample_video)

	print(f"{gtime()}: tts start")
	await tts_inference(text, personality.gpt_cond_latent, personality.speaker_embedding, out / "audio.wav")
	print(f"{gtime()}: tts end")

	print(f"{gtime()}: lipsync start")
	async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as h:
		request = {"audio_path": str(out / "audio.wav"), "video_path": str(video)}
		async with h.post("http://lipsync:6873", json=request) as result:
			wav2lip = await result.text()
	print(f"{gtime()}: lipsync stop")
	print(f"Result: {wav2lip}")
	return {"result": wav2lip}
