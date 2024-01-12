from os import PathLike

from llama_cpp import Llama


def load_llm(path: str | PathLike = '/models/llm.gguf') -> Llama:
	return Llama(
		model_path=str(path),
		chat_format="llama-2",
		n_gpu_layers=-1,
		seed=-1,
		n_threads=4,
		n_threads_batch=4,
		offload_kqv=True,
	)
