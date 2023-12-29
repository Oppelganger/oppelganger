import os
import json
import uuid
import random
import asyncio
import requests

import torch

from pathlib import Path
from dotmap import DotMap
from llama_cpp import Llama

from pydantic import BaseModel

from TTS.api import TTS

from fastapi import FastAPI

app = FastAPI()

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Torch devide {torch_device}')

mixtral = Llama(
  model_path = "./models/mistral-7b-instruct-v0.2.Q6_K.gguf",
  chat_format = "llama-2"
)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(torch_device)

personalities = []

for path in Path("/personalities").glob("*/"):
  with open(path / "personality.json") as metajson:
    personalitiy = DotMap(json.load(metajson))
    personalitiy.path = path

    personalities.append(personalitiy)

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
    { "role": "user", "content": user_input },
  ]

  result = DotMap(mixtral.create_chat_completion(messages))
  text = result.choices[0].message.content

  out = Path(f"/share/{uuid.uuid4()}")
  out.mkdir(parents=True, exist_ok=False)

  video = personality.path / random.choice(personality.sample_video)

  tts.tts_to_file(
    text=text,
    language="en",
    speaker_wav=[str(personality.path / audio) for audio in personality.sample_audio],
    file_path=str(out / "audio.wav"),
    speed=1.5,
  )

  wav2lip = requests.post("http://lipsync:6873", json={"audio_path": str(out / "audio.wav"), "video_path": str(video)})

  print(f"Result: {wav2lip.text}")
  return { "result": wav2lip.text }
