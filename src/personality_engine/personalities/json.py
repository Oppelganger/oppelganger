from pydantic import BaseModel

from .types import Language


class PersonalityJson(BaseModel):
	id: str
	prompt: str
	video_objects: list[str]
	audio_objects: list[str]
	language: Language
	enhance: bool
	female: bool
