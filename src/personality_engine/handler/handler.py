import os
import random
import uuid
from pathlib import Path
from typing import Callable, Any, Optional

from TTS.tts.models.xtts import Xtts
from llama_cpp import ChatCompletionRequestMessage, Llama
from mypy_boto3_s3.client import S3Client

from .reply import Reply
from .request import Request
from ..lipsync import Wav2Lip
from ..llm import create_chat_completion
from ..personalities import load_personality
from ..tts.xtts import inference as xtts
from ..utils import strict_getenv, flatten
from .types import ResponseType

bucket_videos = strict_getenv('S3_BUCKET_VIDEOS')


def create_handler(
	s3: S3Client,
	llm: Llama,
	xtts_model: Xtts,
	wav2lip: Wav2Lip,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
	def handler(job: dict[str, Any]) -> dict[str, Any]:
		request = Request.model_validate(job['input'])
		personality = load_personality(s3, xtts_model, request.personality)

		messages: list[ChatCompletionRequestMessage] = [
			{"role": "system", "content": personality.prompt},
		] + flatten([
			[
				{"role": "user", "content": message.request},
				{"role": "assistant", "content": message.reply}
			]
			for message in request.messages
		]) + [
			{"role": "user", "content": request.prompt},
		]

		result = create_chat_completion(llm, messages)
		text: Optional[str] = result["choices"][0]["message"]["content"]

		if text is None:
			raise RuntimeError("result from llm is null")

		if request.response_type == ResponseType.TEXT:
			return Reply(
				text=text,
				object_type=request.response_type,
				object=None
			).model_dump(mode='json')

		generated_audio = Path(f"/tmp/{uuid.uuid4()}.wav")
		xtts(
			xtts_model,
			text,
			personality.gpt_cond_latent,
			personality.speaker_embedding,
			generated_audio,
			language=str(personality.language)
		)

		if request.response_type == ResponseType.AUDIO:
			object_id = str(uuid.uuid4())
			s3.upload_file(str(generated_audio), bucket_videos, object_id)
			os.remove(generated_audio)
			return Reply(
				text=text,
				object_type=request.response_type,
				object=object_id
			).model_dump(mode='json')

		video = random.choice(personality.video_objects)
		generated_video = Path(f"/tmp/{uuid.uuid4()}.mp4")

		wav2lip.process(
			str(generated_audio),
			str(video),
			str(generated_video),
			personality.enhance,
			personality.female
		)

		object_id = str(uuid.uuid4())

		s3.upload_file(str(generated_video), bucket_videos, object_id)
		os.remove(generated_audio)
		os.remove(generated_video)

		return Reply(
			text=text,
			object_type=request.response_type,
			object=object_id
		).model_dump(mode='json')

	return handler
