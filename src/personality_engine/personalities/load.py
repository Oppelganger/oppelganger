from os import PathLike
from pathlib import Path

from TTS.tts.models.xtts import Xtts

from .json import PersonalityJson
from .personality import Personality
from ..utils import awaitable


@awaitable
def load(
	xtts: Xtts,
	personalities_path: str | PathLike = '/personalities'
) -> dict[str, Personality]:
	out: dict[str, Personality] = {}
	for path in Path(personalities_path).glob('*/'):
		with open(path / 'personality.json') as meta:
			personality = PersonalityJson.model_validate_json(meta.read())

			voices = [str(path / audio) for audio in personality.sample_audio]
			gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
				audio_path=voices,
				gpt_cond_len=xtts.config.gpt_cond_len,
				max_ref_length=xtts.config.max_ref_len,
				sound_norm_refs=xtts.config.sound_norm_refs
			)

			gpt_cond_latent = gpt_cond_latent.to(xtts.device)
			speaker_embedding = speaker_embedding.to(xtts.device)

			out[personality.id] = Personality(
				path,

				personality.prompt,
				personality.sample_video,
				personality.sample_audio,
				personality.enhance,
				personality.female,

				gpt_cond_latent,
				speaker_embedding
			)
	return out
