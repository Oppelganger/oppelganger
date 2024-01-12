from pydantic import BaseModel


class Reply(BaseModel):
	text: str
	video_object: str
