import json
import uuid
import torch
import random
import requests

from pathlib import Path
from dotmap import DotMap
from TTS.api import TTS
from llama_cpp import Llama

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Torch devide {torch_device}')

mixtral = Llama(
  model_path = "./models/mistral-7b-instruct-v0.2.Q6_K.gguf",
  chat_format = "llama-2"
)

tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(torch_device)

personalities = []

for path in Path("/personalities").glob("*/"):
  with open(path / "personality.json") as metajson:
    personalitiy = DotMap(json.load(metajson))
    personalitiy.path = path

    print(f"{len(personalities)}. {personalitiy.name}")
    personalities.append(personalitiy)

while True:
  personality = personalities[int(input("Please select an personality: "))]
  user_prompt = input("Ask anything: ")

  messages = [
    { "role": "system", "content": personality.prompt },
    { "role": "user", "content": user_prompt },
  ]

  result = DotMap(mixtral.create_chat_completion(messages))
  text = " ".join(result.choices[0].message.content.split(".")[:4])

  out = Path(f"/share/{uuid.uuid4()}")
  out.mkdir(parents=True, exist_ok=False)

  voice = personality.path / random.choice(personality.sample_audio)
  video = personality.path / random.choice(personality.sample_video)

  tts.tts_to_file(text=text, speaker_wav=voice, file_path=out / "audio.wav", language="en")

  wav2lip = requests.post("http://lipsync:6873", json={"audio_path": str(out / "audio.wav"), "video_path": str(video)})

  print(f"Result: {wav2lip.text}")
