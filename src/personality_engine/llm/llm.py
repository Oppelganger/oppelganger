from llama_cpp import Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse

from ..utils import awaitable


@awaitable
def create_chat_completion(
	llm: Llama,
	messages: list[ChatCompletionRequestMessage]
) -> CreateChatCompletionResponse:
	return llm.create_chat_completion(messages, max_tokens=100)
