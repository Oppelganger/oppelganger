from typing import Tuple

from pydantic import BaseModel


class PersonalityJson(BaseModel):
	id: str
	prompt: str
	video_objects: dict[str, Tuple[int, int, int, int]]  # x1, y1, x2, y2
	audio_objects: list[str]
	enhance: bool
	female: bool
