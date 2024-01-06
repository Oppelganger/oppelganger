from pydantic import TypeAdapter
from typing_extensions import TypedDict  # pydantic requires using typing_extensions on <3.12


class Job(TypedDict):
	personality: str
	text: str


JobTA = TypeAdapter(Job)
