import random
import uuid
from pathlib import Path
from typing import Callable, Coroutine, Any, Optional

import aiofiles.os
from TTS.tts.models.xtts import Xtts
from llama_cpp import ChatCompletionRequestMessage, Llama

from ..lipsync import Wav2Lip
from ..llm import create_chat_completion
from ..model import Job, JobTA
from ..personalities import Personality
from ..tts.xtts import inference as xtts


def create_handler(
	upload_to_s3: Callable[[str, str], Coroutine[Any, Any, None]],
	llm: Llama,
	xtts_model: Xtts,
	wav2lip: Wav2Lip,
	personalities: dict[str, Personality]
) -> Callable[[Job], Coroutine[Any, Any, str | dict[str, Any]]]:
	async def handler(job: Job) -> str | dict[str, Any]:
		JobTA.validate_python(job)

		user_input = job["text"]
		personality = personalities[job["personality"]]

		messages: list[ChatCompletionRequestMessage] = [
			{"role": "system", "content": personality.prompt},
			{"role": "system", "content": "Please write text in less than 20 words"},
			{"role": "user", "content": user_input},
		]

		result = await create_chat_completion(llm, messages)
		text: Optional[str] = result["choices"][0]["message"]["content"]

		if text is None:
			return {"error": "result from llm is null"}

		video = personality.path / random.choice(personality.sample_video)

		generated_audio = Path(f"/tmp/{uuid.uuid4()}.wav")
		await xtts(
			xtts_model,
			text,
			personality.gpt_cond_latent,
			personality.speaker_embedding,
			generated_audio
		)

		generated_video = Path(f"/tmp/{uuid.uuid4()}.mp4")

		await wav2lip.process(
			str(generated_audio),
			str(video),
			str(generated_video),
			personality.enhance,
			personality.female
		)

		object_id = str(uuid.uuid4())

		await upload_to_s3(str(generated_video), object_id)
		await aiofiles.os.remove(generated_audio)
		await aiofiles.os.remove(generated_video)

		return object_id

	return handler
