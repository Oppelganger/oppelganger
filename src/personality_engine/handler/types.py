from enum import StrEnum

from pydantic import BaseModel


class LlmMessage(BaseModel):
	request: str
	reply: str


class ResponseType(StrEnum):
	VIDEO = 'video'
	AUDIO = 'audio'
	TEXT = 'text'
