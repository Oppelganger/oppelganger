from typing import Tuple

from TTS.tts.models.xtts import Xtts
from mypy_boto3_s3.client import S3Client

from .json import PersonalityJson
from .personality import Personality
from ..utils import strict_getenv

bucket_personalities = strict_getenv("S3_BUCKET_PERSONALITIES")

cache: dict[str, Personality] = {}


def load(s3: S3Client, xtts: Xtts, json: PersonalityJson) -> Personality:
	if json.id in cache:
		return cache[json.id]

	path = f'/tmp/personalities/{json.id}'

	audios: list[str] = []
	videos: list[Tuple[str, Tuple[int, int, int, int]]] = []

	for audio_object in json.audio_objects:
		audio = f'{path}/{audio_object}'
		s3.download_file(bucket_personalities, f'{json.id}/{audio_object}', audio)
		audios.append(audio)

	for video_object, coords in json.video_objects.items():
		video = f'{path}/{video_object}'
		s3.download_file(bucket_personalities, f'{json.id}/{video_object}', video)
		videos.append((video, coords))

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
		json.prompt,
		videos,
		audios,
		json.enhance,
		json.female,

		gpt_cond_latent,
		speaker_embedding
	)

	cache[json.id] = personality

	return personality
