from pydantic import BaseModel


class PersonalityJson(BaseModel):
	id: str
	prompt: str
	sample_video: list[str]
	sample_audio: list[str]
	enhance: bool
	female: bool
