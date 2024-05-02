from typing import Iterable

from pydantic import BaseModel

from .types import LlmMessage, ResponseType
from ..personalities import PersonalityJson as Personality


class Request(BaseModel):
	personality: Personality
	messages: Iterable[LlmMessage]
	prompt: str
	response_type: ResponseType = ResponseType.VIDEO
