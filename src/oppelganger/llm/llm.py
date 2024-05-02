from llama_cpp import Llama, ChatCompletionRequestMessage, CreateChatCompletionResponse


def create_chat_completion(
	llm: Llama,
	messages: list[ChatCompletionRequestMessage]
) -> CreateChatCompletionResponse:
	return llm.create_chat_completion(messages, max_tokens=100)
