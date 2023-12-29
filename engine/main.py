import os
import json
import uuid
import random
import asyncio
import requests
import subprocess

import torch
import torchaudio

from pathlib import Path
from dotmap import DotMap
from llama_cpp import Llama

from pydantic import BaseModel

from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from fastapi import FastAPI

app = FastAPI()

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Torch devide {torch_device}')

mixtral = Llama(
  model_path = "./models/mixtral_7bx2_moe.Q5_K_M.gguf",
  chat_format = "llama-2",
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
tts.load_checkpoint(tts_config, checkpoint_dir=tts_model_path, use_deepspeed=False)

if torch.cuda.is_available():
  tts.cuda()

personalities = []

for path in Path("/personalities").glob("*/"):
  with open(path / "personality.json") as metajson:
    personality = DotMap(json.load(metajson))
    personality.path = path

    voices = [str(path / audio) for audio in personality.sample_audio]
    gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=voices)

    personality.gpt_cond_latent = gpt_cond_latent.to(torch_device)
    personality.speaker_embedding = speaker_embedding.to(torch_device)

    personalities.append(personality)

persons = {person.command.name: person for person in personalities}

@app.get("/persons", response_model=dict[str, str])
async def get_persons():
  return {p.command.name: p.command.desc for p in personalities}

class GenerateRequest(BaseModel):
  person: str
  text: str

@app.post("/generate")
async def get_generate(req: GenerateRequest):
  print(req.text)
  user_input = req.text

  if user_input == "":
    return

  personality = persons[req.person]

  messages = [
    { "role": "system", "content": personality.prompt },
    { "role": "system", "content": "Please write text in less than 20 words" },
    { "role": "user", "content": user_input },
  ]

  result = DotMap(mixtral.create_chat_completion(messages, max_tokens=50))
  text = result.choices[0].message.content

  out = Path(f"/share/{uuid.uuid4()}")
  out.mkdir(parents=True, exist_ok=False)

  sample_video = random.choice(personality.sample_video)
  video = personality.path / sample_video.name

  tres = tts.inference(
    text,
    "en",
    personality.gpt_cond_latent,
    personality.speaker_embedding,
    speed=1.35,
    enable_text_splitting=True,
    temperature=tts.config.temperature,
    length_penalty=tts.config.length_penalty,
    repetition_penalty=tts.config.repetition_penalty,
    top_k=tts.config.top_k,
    top_p=tts.config.top_p,
  )

  torchaudio.save(str(out / "audio.wav"), torch.tensor(tres["wav"]).unsqueeze(0), 24000)

  wav2lip = requests.post("http://lipsync:6873", json={"audio_path": str(out / "audio.wav"), "video_path": str(video)})

  if wav2lip.status_code == 200:
    print(f"Result: {wav2lip.text}")
    file = wav2lip.text
    vo = Path(f"/result/{uuid.uuid4()}.mp4")
    subprocess.run(["ffmpeg", "-i", file, "-vf", f"crop={sample_video.crop}", "-t", "00:00:55", vo])
    return { "result": str(vo) }
  else:
    return { "result": wav2lip.text }
