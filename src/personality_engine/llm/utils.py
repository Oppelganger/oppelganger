from typing import List, Tuple, Optional

from llama_cpp import ChatCompletionRequestMessage


def map_roles(
	messages: List[ChatCompletionRequestMessage],
	role_map: dict[str, str]
) -> List[Tuple[str, Optional[str]]]:
	"""Map the message roles."""
	output: List[Tuple[str, Optional[str]]] = []
	for message in messages:
		role = message["role"]
		if role in role_map:
			output.append((role_map[str(role)], message["content"]))
	return output


def format_add_colon_two(
	system_message: str,
	messages: List[Tuple[str, Optional[str]]],
	sep: str,
	sep2: str
) -> str:
	"""Format the prompt with the add-colon-two style."""
	seps = [sep, sep2]
	ret = system_message + seps[0]
	for i, (role, message) in enumerate(messages):
		if message:
			ret += role + ": " + message + seps[int(i) % 2]
		else:
			ret += role + ":"
	return ret
