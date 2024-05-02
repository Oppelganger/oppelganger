import os
from threading import Lock

from TTS.tts.models.xtts import Xtts
from mypy_boto3_s3.client import S3Client

from .json import PersonalityJson
from .personality import Personality
from ..utils import strict_getenv

bucket_personalities = strict_getenv("S3_BUCKET_PERSONALITIES")

lock = Lock()
cache: dict[str, Personality] = {}


def load(s3: S3Client, xtts: Xtts, json: PersonalityJson) -> Personality:
	with lock:
		if json.id in cache:
			return cache[json.id]

		path = f'/tmp/personalities/{json.id}'
		os.makedirs(path, exist_ok=True)

		audios: list[str] = []
		videos: list[str] = []

		for audio_object in json.audio_objects:
			audio = f'{path}/{audio_object}'
			s3.download_file(bucket_personalities, f'{json.id}/{audio_object}', audio)
			audios.append(audio)

		for video_object in json.video_objects:
			video = f'{path}/{video_object}'
			s3.download_file(bucket_personalities, f'{json.id}/{video_object}', video)
			videos.append(video)

		gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
			audio_path=audios,
			gpt_cond_len=xtts.config.gpt_cond_len,
			max_ref_length=xtts.config.max_ref_len,
			sound_norm_refs=xtts.config.sound_norm_refs
		)

		gpt_cond_latent = gpt_cond_latent.to(xtts.device)
		speaker_embedding = speaker_embedding.to(xtts.device)

		personality = Personality(
			json.id,
			'\n'.join(json.prompt),
			videos,
			audios,
			json.language,
			json.enhance,
			json.female,

			gpt_cond_latent,
			speaker_embedding
		)

		cache[json.id] = personality

		return personality
