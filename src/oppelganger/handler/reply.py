from typing import Optional

from pydantic import BaseModel

from .types import ResponseType


class Reply(BaseModel):
	text: str
	object: Optional[str]
	object_type: ResponseType
