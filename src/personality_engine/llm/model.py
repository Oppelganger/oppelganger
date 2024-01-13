from os import PathLike
from typing import List

from llama_cpp import (
	ChatCompletionRequestSystemMessage,
	ChatCompletionRequestAssistantMessage,
	ChatCompletionRequestUserMessage
)
from llama_cpp import Llama
from llama_cpp.llama_chat_format import register_chat_format, ChatFormatterResponse

from .utils import map_roles, format_add_colon_two

BasicChatCompletionRequestMessage = ChatCompletionRequestSystemMessage | \
																		ChatCompletionRequestUserMessage | \
																		ChatCompletionRequestAssistantMessage


@register_chat_format("airoboros")
def _(messages: List[BasicChatCompletionRequestMessage]) -> ChatFormatterResponse:
	roles = dict(user='USER', assistant='ASSISTANT')
	_messages = map_roles(messages, roles)
	_messages.append((roles['assistant'], None))
	return ChatFormatterResponse(prompt=format_add_colon_two(
		'\n'.join([str(msg['content']) for msg in messages if msg['role'] == 'system']),
		_messages,
		' ',
		'<\\s>'
	))


def load_llm(path: str | PathLike = '/models/llm.gguf') -> Llama:
	return Llama(
		model_path=str(path),
		chat_format='airoboros',
		n_ctx=8192,
		n_batch=1024,
		n_gpu_layers=-1,
		offload_kqv=True,
		seed=-1,
		n_threads=8,
		n_threads_batch=8,
	)
