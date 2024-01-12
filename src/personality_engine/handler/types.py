from pydantic import BaseModel


class LlmMessage(BaseModel):
	request: str
	reply: str
